import cv2
import numpy as np
import math
import os
import json

# Default fallback — akan tergantikan oleh app/database.json probabilistik
DEFAULT_DATABASE = {
    "Apel (Merah)": {
        "hsv_ranges": [
            (np.array([0, 70, 70]), np.array([10, 255, 255])),
            (np.array([160, 70, 70]), np.array([180, 255, 255]))
        ],
        "stats": {
            "mean_circularity": 0.33, "std_circularity": 0.12,
            "mean_aspect_ratio": 1.32, "std_aspect_ratio": 0.25,
            "mean_contrast": 283, "std_contrast": 50,
            "min_circularity": 0.21, "max_circularity": 0.49,
            "min_aspect_ratio": 1.0, "max_aspect_ratio": 1.63,
            "min_contrast": 213, "max_contrast": 322, "n": 3
        },
        "rules": {
            "min_circularity": 0.20, "max_circularity": 0.90,
            "min_aspect_ratio": 0.85, "max_aspect_ratio": 1.85,
            "min_contrast": 150, "max_contrast": 700
        }
    },
    "Pisang (Kuning/Hijau)": {
        "hsv_ranges": [
            (np.array([20, 100, 100]), np.array([35, 255, 255])),
            (np.array([36, 50, 50]), np.array([50, 255, 255]))
        ],
        "stats": {
            "mean_circularity": 0.318, "std_circularity": 0.05,
            "mean_aspect_ratio": 1.38, "std_aspect_ratio": 0.1,
            "mean_contrast": 673, "std_contrast": 80,
            "min_circularity": 0.318, "max_circularity": 0.318,
            "min_aspect_ratio": 1.38, "max_aspect_ratio": 1.38,
            "min_contrast": 673, "max_contrast": 673, "n": 1
        },
        "rules": {
            "min_circularity": 0.10, "max_circularity": 0.50,
            "min_aspect_ratio": 1.20, "max_aspect_ratio": 3.00,
            "min_contrast": 450, "max_contrast": 900
        }
    },
    "Jeruk": {
        "hsv_ranges": [
            (np.array([11, 100, 100]), np.array([25, 255, 255]))
        ],
        "stats": {
            "mean_circularity": 0.779, "std_circularity": 0.05,
            "mean_aspect_ratio": 1.05, "std_aspect_ratio": 0.05,
            "mean_contrast": 1104, "std_contrast": 100,
            "n": 1
        },
        "rules": {
            "min_circularity": 0.60, "max_circularity": 0.95,
            "min_aspect_ratio": 0.90, "max_aspect_ratio": 1.30,
            "min_contrast": 850, "max_contrast": 1400
        }
    },
    "Durian": {
        "hsv_ranges": [
            (np.array([15, 30, 30]), np.array([50, 255, 255]))
        ],
        "stats": {
            "mean_circularity": 0.571, "std_circularity": 0.05,
            "mean_aspect_ratio": 1.39, "std_aspect_ratio": 0.05,
            "mean_contrast": 2062, "std_contrast": 200,
            "n": 1
        },
        "rules": {
            "min_circularity": 0.40, "max_circularity": 0.75,
            "min_aspect_ratio": 1.00, "max_aspect_ratio": 1.80,
            "min_contrast": 1500, "max_contrast": 2600
        }
    }
}

LABEL_MAP = {
    "apel": "Apel (Merah)",
    "pisang": "Pisang (Kuning/Hijau)",
    "jeruk": "Jeruk",
    "durian": "Durian",
}

def _load_database():
    candidates = ["app/database.json", "database.json", "data_buah/database.json"]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                db = {}
                for label, v in data.items():
                    ranges = []
                    for lo, hi in v["hsv_ranges"]:
                        ranges.append((np.array(lo, dtype=np.int32), np.array(hi, dtype=np.int32)))
                    entry = {"hsv_ranges": ranges}
                    if "stats" in v:
                        entry["stats"] = v["stats"]
                    if "rules" in v:
                        entry["rules"] = v["rules"]
                    # fallback: buat stats dari rules jika stats belum ada (backward compat)
                    if "stats" not in entry and "rules" in entry:
                        r=entry["rules"]
                        entry["stats"]={
                            "mean_circularity": (r["min_circularity"]+r["max_circularity"])/2,
                            "std_circularity": max(0.03, (r["max_circularity"]-r["min_circularity"])/4),
                            "mean_aspect_ratio": (r["min_aspect_ratio"]+r["max_aspect_ratio"])/2,
                            "std_aspect_ratio": max(0.05, (r["max_aspect_ratio"]-r["min_aspect_ratio"])/4),
                            "mean_contrast": (r["min_contrast"]+r["max_contrast"])/2,
                            "std_contrast": max(30, (r["max_contrast"]-r["min_contrast"])/4),
                            "min_circularity": r["min_circularity"],
                            "max_circularity": r["max_circularity"],
                            "min_aspect_ratio": r["min_aspect_ratio"],
                            "max_aspect_ratio": r["max_aspect_ratio"],
                            "min_contrast": r["min_contrast"],
                            "max_contrast": r["max_contrast"],
                            "n": 3
                        }
                    db[label] = entry
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

def _hitung_probabilitas(fitur, stats):
    """Hitung probabilitas Gaussian per fitur, rata-rata."""
    def gauss_sim(x, mean, std):
        if std <= 1e-6:
            std = max(0.03, abs(mean)*0.15)
        z = abs(x - mean) / std
        # Gaussian similarity: exp(-0.5*z^2)
        s = math.exp(-0.5 * z * z)
        return s, z
    s_circ, zc = gauss_sim(fitur["circularity"], stats["mean_circularity"], stats["std_circularity"])
    s_ar, za = gauss_sim(fitur["aspect_ratio"], stats["mean_aspect_ratio"], stats["std_aspect_ratio"])
    s_kon, zk = gauss_sim(fitur["kontras"], stats["mean_contrast"], stats["std_contrast"])
    # bobot equal
    prob = (s_circ + s_ar + s_kon) / 3.0
    # detail untuk debug
    detail = {"s_circ": s_circ, "s_ar": s_ar, "s_kon": s_kon, "z_circ": zc, "z_ar": za, "z_kon": zk}
    return prob, detail

def _identifikasi_core(img_bgr, database=None, verbose=True):
    if database is None:
        database = get_database()

    lebar_target = 500
    h0, w0 = img_bgr.shape[:2]
    rasio = lebar_target / float(w0)
    tinggi_target = int(h0 * rasio)
    img = cv2.resize(img_bgr, (lebar_target, tinggi_target))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    total_piksel = img.shape[0] * img.shape[1]
    min_area_threshold = 0.01 * total_piksel

    mask_terbaik = None
    data_fitur_terbaik = {}
    bbox_terbaik = None
    semua_deteksi = []
    objek_terdeteksi = False

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
                    # hitung probabilitas jika stats ada, fallback ke rules biner
                    stats = data.get("stats")
                    if stats:
                        prob, detail = _hitung_probabilitas(fitur, stats)
                        # probabilitas berbasis area juga: jika area sangat kecil, kurangi
                        # tapi tidak hard-threshold
                    else:
                        # fallback biner -> prob 1 atau 0
                        rules = data.get("rules", {})
                        prob = 1.0
                        detail = {}
                        if "min_circularity" in rules and fitur["circularity"] < rules["min_circularity"]:
                            prob = 0.0
                        if "max_circularity" in rules and fitur["circularity"] > rules["max_circularity"]:
                            prob = 0.0
                        # sederhanakan: jika ada rules fail, prob 0
                    semua_deteksi.append({
                        "nama_buah": nama_buah,
                        "circularity": fitur["circularity"],
                        "aspect_ratio": fitur["aspect_ratio"],
                        "kontras": fitur["kontras"],
                        "homogenitas": fitur["homogenitas"],
                        "area": area,
                        "area_piksel": area_piksel,
                        "bbox": (x, y, w, h),
                        "probabilitas": float(prob),
                        "detail": detail,
                        "mask": mask,
                    })

    # sorting probabilitas tertinggi
    semua_deteksi.sort(key=lambda d: d["probabilitas"], reverse=True)

    if semua_deteksi:
        top = semua_deteksi[0]
        prob = top["probabilitas"]
        # threshold rendah: jika prob <0.30, anggap tidak ada buah
        if prob < 0.30:
            hasil_identifikasi = f"Tidak ada buah terdeteksi (tertinggi {top['nama_buah']} {prob*100:.1f}%)"
            mask_terbaik = None
            bbox_terbaik = None
            data_fitur_terbaik = {}
        else:
            hasil_identifikasi = f"{top['nama_buah']} ({prob*100:.1f}%)"
            mask_terbaik = top["mask"]
            bbox_terbaik = top["bbox"]
            data_fitur_terbaik = {
                "Area (px)": float(top["area"]),
                "Area Mask (px)": int(top["area_piksel"]),
                "Circularity": float(top["circularity"]),
                "Aspect Ratio": float(top["aspect_ratio"]),
                "Kontras": float(top["kontras"]),
                "Homogenitas": float(top["homogenitas"]),
                "Probabilitas": float(prob),
            }
    else:
        hasil_identifikasi = "Tidak ada buah terdeteksi (tidak ada warna cocok)"
        if not objek_terdeteksi:
            hasil_identifikasi = "Tidak ada buah terdeteksi (tidak ada warna cocok)"

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
    return _identifikasi_core(frame_bgr, database=database, verbose=verbose)

def identifikasi_buah(image_path, verbose=True, headless=False, save_dir="output", database=None):
    img = cv2.imread(image_path)
    if img is None:
        return {"image": image_path, "hasil": "Tidak ada buah terdeteksi (file tidak ditemukan)", "error": "file not found"}

    result = _identifikasi_core(img, database=database, verbose=False)

    if verbose:
        print("\n" + "="*60)
        print(f"MULAI PROSES SCANNING (probabilistik): {image_path}")
        print("="*60)
        for d in result["semua_deteksi"]:
            print(f"\n[?] {d['nama_buah']}: prob={d['probabilitas']*100:.1f}% | circ={d['circularity']:.3f} ar={d['aspect_ratio']:.2f} kontr={d['kontras']:.0f}")
            if "detail" in d and d["detail"]:
                print(f"     detail: s_circ={d['detail']['s_circ']:.2f} s_ar={d['detail']['s_ar']:.2f} s_kon={d['detail']['s_kon']:.2f}")
        print("\n" + "="*60)
        print(f"KESIMPULAN AKHIR: {result['hasil']}")
        print("="*60)
        if result["fitur"]:
            for k, v in result["fitur"].items():
                print(f" - {k}: {v:.4f}" if isinstance(v, float) else f" - {k}: {v}")
        print("="*60 + "\n")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import os.path as op
        base = op.splitext(op.basename(image_path))[0]
        cv2.imwrite(os.path.join(save_dir, f"{base}_resized.png"), result["img"])
        if result["mask_terbaik"] is not None:
            safe_label = result["hasil"].split(" (")[0].replace(" ","_").replace("/","-")
            cv2.imwrite(os.path.join(save_dir, f"{base}_mask_{safe_label}.png"), result["mask_terbaik"])
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
    out = {}
    if not os.path.exists(root):
        return out
    for entry in os.listdir(root):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue
        if entry.startswith("."):
            continue
        label = LABEL_MAP.get(entry.lower(), entry.capitalize())
        if entry in DEFAULT_DATABASE:
            label = entry
        files = []
        for f in os.listdir(folder):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                files.append(os.path.join(folder, f))
        if files:
            out[label] = sorted(files)
    return out
