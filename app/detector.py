import cv2
import numpy as np
import math
import os
import json

# Default database (fallback jika database.json belum ada)
DEFAULT_DATABASE = {
    "Apel (Merah)": {
        "hsv_ranges": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))
        ],
        "rules": {
            "min_circularity": 0.60,
            "max_circularity": 0.95,
            "min_aspect_ratio": 0.90,
            "max_aspect_ratio": 1.35,
            "min_contrast": 1100,
            "max_contrast": 1800
        }
    },
    "Pisang (Kuning/Hijau)": {
        "hsv_ranges": [
            (np.array([20, 100, 100]), np.array([35, 255, 255])),
            (np.array([36, 50, 50]), np.array([50, 255, 255]))
        ],
        "rules": {
            "min_circularity": 0.10,
            "max_circularity": 0.50,
            "min_aspect_ratio": 1.20,
            "max_aspect_ratio": 3.00,
            "min_contrast": 450,
            "max_contrast": 900
        }
    },
    "Jeruk": {
        "hsv_ranges": [
            (np.array([11, 100, 100]), np.array([25, 255, 255]))
        ],
        "rules": {
            "min_circularity": 0.60,
            "max_circularity": 0.95,
            "min_aspect_ratio": 0.90,
            "max_aspect_ratio": 1.30,
            "min_contrast": 850,
            "max_contrast": 1400
        }
    },
    "Durian": {
        "hsv_ranges": [
            (np.array([15, 30, 30]), np.array([50, 255, 255]))
        ],
        "rules": {
            "min_circularity": 0.40,
            "max_circularity": 0.75,
            "min_aspect_ratio": 1.00,
            "max_aspect_ratio": 1.80,
            "min_contrast": 1500,
            "max_contrast": 2600
        }
    }
}

# Mapping ekstensibel: folder -> label (display name)
# Jika database.json ada, mapping diambil dari sana.
LABEL_MAP = {
    "apel": "Apel (Merah)",
    "pisang": "Pisang (Kuning/Hijau)",
    "jeruk": "Jeruk",
    "durian": "Durian",
}

def _load_database():
    """Load database.json jika ada, fallback ke DEFAULT_DATABASE."""
    candidates = ["app/database.json", "database.json", "data_buah/database.json"]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                # data disimpan sebagai {label: {hsv_ranges: [[[lo],[hi]]], rules: {...}}}
                db = {}
                for label, v in data.items():
                    ranges = []
                    for lo, hi in v["hsv_ranges"]:
                        ranges.append((np.array(lo, dtype=np.int32), np.array(hi, dtype=np.int32)))
                    db[label] = {"hsv_ranges": ranges, "rules": v["rules"]}
                if db:
                    return db
            except Exception as e:
                print(f"[WARN] Gagal load {p}: {e}")
    return DEFAULT_DATABASE

def get_database():
    return _load_database()

def buat_masker_warna(hsv_img, hsv_ranges):
    mask_gabungan = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for batas_bawah, batas_atas in hsv_ranges:
        mask = cv2.inRange(hsv_img, batas_bawah, batas_atas)
        mask_gabungan = cv2.bitwise_or(mask_gabungan, mask)
    kernel = np.ones((5, 5), np.uint8)
    mask_gabungan = cv2.morphologyEx(mask_gabungan, cv2.MORPH_OPEN, kernel)
    mask_gabungan = cv2.morphologyEx(mask_gabungan, cv2.MORPH_CLOSE, kernel)
    return mask_gabungan

def hitung_tekstur_glcm(gray_img, x, y, w, h, mask=None):
    roi_gray = gray_img[y:y+h, x:x+w]
    if roi_gray.size == 0:
        return 0, 0
    _ = mask
    max_side = 150
    if max(roi_gray.shape) > max_side:
        scale = max_side / max(roi_gray.shape)
        new_w = max(1, int(roi_gray.shape[1] * scale))
        new_h = max(1, int(roi_gray.shape[0] * scale))
        roi_gray = cv2.resize(roi_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    try:
        from skimage.feature import graycomatrix, graycoprops
        glcm = graycomatrix(roi_gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        kontras = graycoprops(glcm, 'contrast')[0, 0]
        homogenitas = graycoprops(glcm, 'homogeneity')[0, 0]
    except Exception:
        kontras, homogenitas = 0, 0
    return kontras, homogenitas

def ekstrak_fitur_kontur(contour, gray, mask, x, y, w, h):
    area = cv2.contourArea(contour)
    keliling = cv2.arcLength(contour, True)
    if keliling == 0:
        return None
    aspek_rasio = max(float(w)/h, float(h)/w) if h != 0 else 0
    circularity = (4 * math.pi * area) / (keliling ** 2)
    kontras, homogenitas = hitung_tekstur_glcm(gray, x, y, w, h, mask=mask)
    return {
        "area": area,
        "keliling": keliling,
        "aspect_ratio": aspek_rasio,
        "circularity": circularity,
        "kontras": kontras,
        "homogenitas": homogenitas,
        "x": x, "y": y, "w": w, "h": h,
    }

def _identifikasi_core(img_bgr, database=None, verbose=True):
    """Core logic: input BGR numpy (sudah di-resize atau belum), return result dict."""
    if database is None:
        database = get_database()

    # Resize proporsional lebar 500
    lebar_target = 500
    h0, w0 = img_bgr.shape[:2]
    rasio = lebar_target / float(w0)
    tinggi_target = int(h0 * rasio)
    img = cv2.resize(img_bgr, (lebar_target, tinggi_target))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    total_piksel = img.shape[0] * img.shape[1]
    min_area_threshold = 0.01 * total_piksel

    hasil_identifikasi = "Tidak Dikenali / Tidak Ada Buah"
    mask_terbaik = None
    data_fitur_terbaik = {}
    max_area_terdeteksi = 0
    objek_terdeteksi = False
    semua_deteksi = []
    bbox_terbaik = None

    for nama_buah, data in database.items():
        mask = buat_masker_warna(hsv, data["hsv_ranges"])
        area_piksel = cv2.countNonZero(mask)
        if area_piksel > min_area_threshold:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                kontur_terbesar = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(kontur_terbesar)
                keliling = cv2.arcLength(kontur_terbesar, True)
                if keliling > 0:
                    objek_terdeteksi = True
                    x, y, w, h = cv2.boundingRect(kontur_terbesar)
                    fitur = ekstrak_fitur_kontur(kontur_terbesar, gray, mask, x, y, w, h)
                    if fitur is None:
                        continue
                    circularity = fitur["circularity"]
                    aspek_rasio = fitur["aspect_ratio"]
                    kontras = fitur["kontras"]
                    homogenitas = fitur["homogenitas"]

                    rules = data["rules"]
                    match = True
                    alasan_gagal = []
                    if "min_circularity" in rules and circularity < rules["min_circularity"]:
                        match = False
                        alasan_gagal.append(f"Circularity ({circularity:.2f}) < Minimal ({rules['min_circularity']})")
                    if "max_circularity" in rules and circularity > rules["max_circularity"]:
                        match = False
                        alasan_gagal.append(f"Circularity ({circularity:.2f}) > Maksimal ({rules['max_circularity']})")
                    if "min_aspect_ratio" in rules and aspek_rasio < rules["min_aspect_ratio"]:
                        match = False
                        alasan_gagal.append(f"Aspect Ratio ({aspek_rasio:.2f}) < Minimal ({rules['min_aspect_ratio']})")
                    if "max_aspect_ratio" in rules and aspek_rasio > rules["max_aspect_ratio"]:
                        match = False
                        alasan_gagal.append(f"Aspect Ratio ({aspek_rasio:.2f}) > Maksimal ({rules['max_aspect_ratio']})")
                    if "min_contrast" in rules and kontras < rules["min_contrast"]:
                        match = False
                        alasan_gagal.append(f"Kontras ({kontras:.2f}) < Minimal ({rules['min_contrast']})")
                    if "max_contrast" in rules and kontras > rules["max_contrast"]:
                        match = False
                        alasan_gagal.append(f"Kontras ({kontras:.2f}) > Maksimal ({rules['max_contrast']})")

                    semua_deteksi.append({
                        "nama_buah": nama_buah,
                        "circularity": circularity,
                        "aspect_ratio": aspek_rasio,
                        "kontras": kontras,
                        "homogenitas": homogenitas,
                        "area": area,
                        "area_piksel": area_piksel,
                        "match": match,
                        "alasan_gagal": alasan_gagal,
                        "bbox": (x, y, w, h),
                    })

                    if match and area > max_area_terdeteksi:
                        max_area_terdeteksi = area
                        hasil_identifikasi = nama_buah
                        mask_terbaik = mask
                        bbox_terbaik = (x, y, w, h)
                        data_fitur_terbaik = {
                            "Area (px)": float(area),
                            "Area Mask (px)": int(area_piksel),
                            "Circularity": float(circularity),
                            "Aspect Ratio": float(aspek_rasio),
                            "Kontras": float(kontras),
                            "Homogenitas": float(homogenitas)
                        }

    return {
        "img": img,
        "hsv": hsv,
        "gray": gray,
        "hasil": hasil_identifikasi,
        "fitur": data_fitur_terbaik,
        "objek_terdeteksi": objek_terdeteksi,
        "semua_deteksi": semua_deteksi,
        "mask_terbaik": mask_terbaik,
        "bbox": bbox_terbaik,
        "scale_ratio": rasio,
    }

def identifikasi_frame(frame_bgr, verbose=False, database=None):
    """Untuk web live camera: frame BGR dari cv2 atau base64 decode."""
    return _identifikasi_core(frame_bgr, database=database, verbose=verbose)

def identifikasi_buah(image_path, verbose=True, headless=False, save_dir="output", database=None):
    """Wrapper untuk path file (CLI)."""
    img = cv2.imread(image_path)
    if img is None:
        return {"image": image_path, "hasil": "Tidak Dikenali / Tidak Ada Buah", "error": "file not found"}

    result = _identifikasi_core(img, database=database, verbose=False)

    if verbose:
        print("\n" + "="*60)
        print(f"MULAI PROSES SCANNING: {image_path}")
        print("="*60)
        for d in result["semua_deteksi"]:
            print(f"\n[?] Deteksi Warna Mirip: {d['nama_buah']}")
            print(f"     - Circularity  : {d['circularity']:.4f}")
            print(f"     - Aspect Ratio : {d['aspect_ratio']:.4f}")
            print(f"     - Kontras GLCM : {d['kontras']:.4f}")
            if d["match"]:
                print(f"    >>> COCOK")
            else:
                print(f"    >>> GAGAL: {', '.join(d['alasan_gagal'])}")
        print("\n" + "="*60)
        print(f"KESIMPULAN AKHIR: {result['hasil']}")
        print("="*60)
        if result["fitur"]:
            for k, v in result["fitur"].items():
                print(f" - {k}: {v:.4f}")
        print("="*60 + "\n")

    # Save output
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import os.path as op
        base = op.splitext(op.basename(image_path))[0]
        cv2.imwrite(os.path.join(save_dir, f"{base}_resized.png"), result["img"])
        if result["mask_terbaik"] is not None:
            cv2.imwrite(os.path.join(save_dir, f"{base}_mask_{result['hasil'].replace(' ','_').replace('/','-')}.png"), result["mask_terbaik"])
            img_kotak = result["img"].copy()
            if result["bbox"]:
                x, y, w, h = result["bbox"]
                cv2.rectangle(img_kotak, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_kotak, result["hasil"], (x, max(15, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.imwrite(os.path.join(save_dir, f"{base}_bbox.png"), img_kotak)

    if not headless and verbose:
        try:
            cv2.imshow("1. Gambar Asli", result["img"])
            if result["mask_terbaik"] is not None:
                cv2.imshow(f"2. Masker: {result['hasil']}", result["mask_terbaik"])
                if result["bbox"]:
                    img_kotak = result["img"].copy()
                    x, y, w, h = result["bbox"]
                    cv2.rectangle(img_kotak, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imshow("3. Bounding Box", img_kotak)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[WARN] imshow gagal: {e}")

    return {"image": image_path, "hasil": result["hasil"], "fitur": result["fitur"], "objek_terdeteksi": result["objek_terdeteksi"], "semua_deteksi": result["semua_deteksi"], "bbox": result["bbox"]}

def scan_data_buah(root="data_buah"):
    """Scan data_buah/*/*.{png,jpg,jpeg} -> {label: [paths]}. Label = folder name mapped via LABEL_MAP atau capitalized."""
    out = {}
    if not os.path.exists(root):
        return out
    for entry in os.listdir(root):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue
        # skip hidden
        if entry.startswith("."):
            continue
        label = LABEL_MAP.get(entry.lower(), entry.capitalize())
        # fallback: if entry is like "Apel (Merah)" folder, use as is
        if entry in DEFAULT_DATABASE:
            label = entry
        files = []
        for f in os.listdir(folder):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                files.append(os.path.join(folder, f))
        if files:
            out[label] = sorted(files)
    return out
