import cv2
import numpy as np
import math
import os
import argparse
import csv
from skimage.feature import graycomatrix, graycoprops

# ==========================================
# 1. DATABASE KARAKTERISTIK BUAH
# (Nilai akan dikalibrasi ulang via kalibrasi.py)
# ==========================================
# Kalibrasi 2026-08-31: berdasarkan kalibrasi.py unmasked GLCM (a1=1453, p1=673, j1=1104, d1=2062)
# margin 25% + validasi manual agar tidak overlap. Threshold area dinormalisasi 1%.
DATABASE_BUAH = {
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

# Mapping file -> label ground truth untuk evaluasi
LABEL_GROUND_TRUTH = {
    "a1.png": "Apel (Merah)",
    "d1.png": "Durian",
    "j1.png": "Jeruk",
    "p1.png": "Pisang (Kuning/Hijau)",
}

# ==========================================
# 2. FUNGSI EKSTRAKSI FITUR
# ==========================================
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
    """Hitung GLCM pada ROI. Jika mask diberikan, hanya untuk logging; GLCM dihitung pada ROI penuh (unmasked)
    karena zeroing background terbukti menginflate kontras (950->2900). Kalibrasi dilakukan unmasked."""
    roi_gray = gray_img[y:y+h, x:x+w]
    if roi_gray.size == 0:
        return 0, 0
    # Catatan: mask-aware zeroing dinonaktifkan - kalibrasi 2026-08-31 menunjukkan masked_zero inflate +50-200%
    # Jika butuh masked, gunakan pendekatan crop ketat tanpa zeroing, bukan set 0.
    _ = mask  # suppress unused
    # Optimasi: downscale jika ROI terlalu besar (j1.png 3000px beratkan GLCM 256 levels)
    max_side = 150
    if max(roi_gray.shape) > max_side:
        scale = max_side / max(roi_gray.shape)
        new_w = max(1, int(roi_gray.shape[1] * scale))
        new_h = max(1, int(roi_gray.shape[0] * scale))
        roi_gray = cv2.resize(roi_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    try:
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

# ==========================================
# 3. FUNGSI UTAMA IDENTIFIKASI & KALIBRASI
# ==========================================
def identifikasi_buah(image_path, headless=False, save_dir="output", verbose=True, return_all=False):
    img = cv2.imread(image_path)
    if img is None:
        msg = f"[ERROR] Gambar '{image_path}' tidak ditemukan!"
        if verbose:
            print(f"\n{msg}")
        return {"image": image_path, "hasil": "Tidak Dikenali / Tidak Ada Buah", "error": msg}

    # Resize proporsional lebar 500
    lebar_target = 500
    tinggi_asli, lebar_asli = img.shape[:2]
    rasio = lebar_target / float(lebar_asli)
    tinggi_target = int(tinggi_asli * rasio)
    img = cv2.resize(img, (lebar_target, tinggi_target))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    total_piksel = img.shape[0] * img.shape[1]
    min_area_threshold = 0.01 * total_piksel  # 1% dinormalisasi (Fase 2)

    hasil_identifikasi = "Tidak Dikenali / Tidak Ada Buah"
    mask_terbaik = None
    data_fitur_terbaik = {}
    max_area_terdeteksi = 0
    objek_terdeteksi = False
    semua_deteksi = []

    if verbose:
        print("\n" + "="*60)
        print(f"MULAI PROSES SCANNING & KALIBRASI: {image_path} (total={total_piksel}px, thr={min_area_threshold:.0f})")
        print("="*60)

    for nama_buah, data in DATABASE_BUAH.items():
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

                    if verbose:
                        print(f"\n[?] Deteksi Warna Mirip: {nama_buah}")
                        print(f"    Data Mentah Sensor:")
                        print(f"     - Circularity  : {circularity:.4f}  (Mendekati 1 = Bulat)")
                        print(f"     - Aspect Ratio : {aspek_rasio:.4f}  (Mendekati 1 = Persegi/Proporsional)")
                        print(f"     - Kontras GLCM : {kontras:.4f} (unmasked, calibrated)")
                        print(f"     - Area kontur  : {area:.0f} px / mask {area_piksel} px")

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
                    })

                    if verbose:
                        if match:
                            print(f"    >>> STATUS: COCOK (Kriteria Terpenuhi)")
                        else:
                            print(f"    >>> STATUS: GAGAL KARENA:")
                            for alasan in alasan_gagal:
                                print(f"        - {alasan}")

                    if match and area > max_area_terdeteksi:
                        max_area_terdeteksi = area
                        hasil_identifikasi = nama_buah
                        mask_terbaik = mask
                        data_fitur_terbaik = {
                            "Area (px)": area,
                            "Area Mask (px)": area_piksel,
                            "Circularity": circularity,
                            "Aspect Ratio": aspek_rasio,
                            "Kontras": kontras,
                            "Homogenitas": homogenitas
                        }

    if not objek_terdeteksi and verbose:
        print("\n[!] TIDAK ADA WARNA yang terdeteksi cocok dengan database.")
        print("    Saran: Sesuaikan nilai 'hsv_ranges' jika buah ada di dalam gambar namun tidak terlacak.")

    if verbose:
        print("\n" + "="*60)
        print(f"KESIMPULAN AKHIR: {hasil_identifikasi}")
        print("="*60)
        if data_fitur_terbaik:
            print("Detail Fitur Objek yang Terpilih:")
            for kunci, nilai in data_fitur_terbaik.items():
                print(f" - {kunci:<15}: {nilai:.4f}")
        else:
            print("Gunakan 'Data Mentah Sensor' di atas untuk mengkalibrasi ulang nilai di DATABASE_BUAH.")
        print("="*60 + "\n")

    # Visualisasi / Save
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    # Selalu save output untuk jejak, bahkan saat demo live
    try:
        cv2.imwrite(os.path.join(save_dir, f"{base}_resized.png"), img)
        if mask_terbaik is not None:
            cv2.imwrite(os.path.join(save_dir, f"{base}_mask_{hasil_identifikasi.replace(' ','_').replace('/','-')}.png"), mask_terbaik)
            img_kotak = img.copy()
            contours, _ = cv2.findContours(mask_terbaik, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                cv2.rectangle(img_kotak, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_kotak, hasil_identifikasi, (x, max(15, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.imwrite(os.path.join(save_dir, f"{base}_bbox.png"), img_kotak)
    except Exception as e:
        if verbose:
            print(f"[WARN] Gagal save output: {e}")

    if not headless:
        try:
            cv2.imshow("1. Gambar Asli", img)
            if mask_terbaik is not None:
                cv2.imshow(f"2. Masker Identifikasi: {hasil_identifikasi}", mask_terbaik)
                img_kotak = img.copy()
                contours, _ = cv2.findContours(mask_terbaik, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    cv2.rectangle(img_kotak, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imshow("3. Bounding Box", img_kotak)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            if verbose:
                print(f"[WARN] imshow gagal (headless env), output sudah disimpan di {save_dir}: {e}")
    else:
        if verbose:
            print(f"[INFO] Headless mode: output disimpan di {save_dir}/")

    result = {
        "image": image_path,
        "hasil": hasil_identifikasi,
        "fitur": data_fitur_terbaik,
        "objek_terdeteksi": objek_terdeteksi,
        "semua_deteksi": semua_deteksi,
    }
    if return_all:
        result["img"] = img
        result["mask_terbaik"] = mask_terbaik
    return result

def evaluasi_semua(headless=True, save_dir="output"):
    files = [f for f in os.listdir(".") if f.lower().endswith(".png") and f in LABEL_GROUND_TRUTH]
    if not files:
        print("[WARN] Tidak ada file png ground truth ditemukan untuk evaluasi.")
        return
    benar = 0
    print("\n" + "="*60)
    print("EVALUASI AKURASI SEMUA GAMBAR")
    print("="*60)
    for f in sorted(files):
        gt = LABEL_GROUND_TRUTH[f]
        res = identifikasi_buah(f, headless=headless, save_dir=save_dir, verbose=False)
        pred = res["hasil"]
        ok = (pred == gt)
        benar += int(ok)
        status = "BENAR" if ok else "SALAH"
        print(f" {f:8} | GT: {gt:22} | Pred: {pred:22} | {status}")
        if not ok and res["semua_deteksi"]:
            for d in res["semua_deteksi"]:
                if d["match"]:
                    print(f"   -> match lain: {d['nama_buah']}")
    acc = benar / len(files) * 100 if files else 0
    print("-"*60)
    print(f"Akurasi: {benar}/{len(files)} ({acc:.1f}%)")
    print("="*60 + "\n")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identifikasi Buah - Demo UAS Kecerdasan Buatan")
    parser.add_argument("--image", type=str, default="d1.png", help="Path gambar (default d1.png)")
    parser.add_argument("--headless", action="store_true", help="Tanpa imshow, hanya save ke output/")
    parser.add_argument("--eval", action="store_true", help="Evaluasi semua gambar ground truth")
    parser.add_argument("--save-dir", type=str, default="output", help="Folder output")
    parser.add_argument("--all", action="store_true", help="Proses semua png di folder")
    args = parser.parse_args()

    if args.eval:
        evaluasi_semua(headless=args.headless, save_dir=args.save_dir)
    elif args.all:
        for f in sorted([x for x in os.listdir(".") if x.lower().endswith(".png")]):
            identifikasi_buah(f, headless=args.headless, save_dir=args.save_dir)
    else:
        identifikasi_buah(args.image, headless=args.headless, save_dir=args.save_dir)
