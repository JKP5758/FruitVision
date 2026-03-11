import cv2
import numpy as np
import math
from skimage.feature import graycomatrix, graycoprops

# ==========================================
# 1. DATABASE KARAKTERISTIK BUAH
# ==========================================
DATABASE_BUAH = {
    "Apel (Merah)": {
        "hsv_ranges": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))
        ],
        "rules": {
            "min_circularity": 0.3,
            "max_circularity": 1.0,   # Ditambah: Batas atas bulat sempurna
            "min_aspect_ratio": 1.0,  # Ditambah: Batas bawah persegi/lingkaran
            "max_aspect_ratio": 1.3,
            "min_contrast": 0,        # Ditambah: Batas bawah tekstur halus
            "max_contrast": 500
        }
    },
    "Pisang (Kuning/Hijau)": {
        "hsv_ranges": [
            (np.array([20, 100, 100]), np.array([35, 255, 255])),
            (np.array([36, 50, 50]), np.array([50, 255, 255]))
        ],
        "rules": {
            "min_circularity": 0.0,   # Ditambah: Pisang tidak bulat
            "max_circularity": 0.5,
            "min_aspect_ratio": 1.3,
            "max_aspect_ratio": 5.0,  # Ditambah: Batas atas untuk bentuk sangat panjang
            "min_contrast": 0,        # Ditambah: Batas bawah tekstur halus
            "max_contrast": 400
        }
    },
    "Jeruk": {
        "hsv_ranges": [
            (np.array([11, 100, 100]), np.array([25, 255, 255]))
        ],
        "rules": {
            "min_circularity": 0.75,
            "max_circularity": 1.0,   # Ditambah: Jeruk sangat bulat
            "min_aspect_ratio": 1.0,  # Ditambah: Batas bawah bentuk proporsional
            "max_aspect_ratio": 1.25,
            "min_contrast": 300,
            "max_contrast": 800
        }
    },
    "Durian": {
        "hsv_ranges": [
            (np.array([15, 30, 30]), np.array([50, 255, 255]))
        ],
        "rules": {
            "min_circularity": 0.1,
            "max_circularity": 0.7,   # Ditambah: Durian jarang bulat sempurna (karena duri)
            "min_aspect_ratio": 1.0,  # Ditambah: Batas bawah bentuk
            "max_aspect_ratio": 1.5,
            "min_contrast": 1300,
            "max_contrast": 5000      # Ditambah: Batas atas untuk tekstur sangat kasar
        }
    }
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

def hitung_tekstur_glcm(gray_img, x, y, w, h):
    roi_gray = gray_img[y:y+h, x:x+w]
    if roi_gray.size == 0:
        return 0, 0
    glcm = graycomatrix(roi_gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    kontras = graycoprops(glcm, 'contrast')[0, 0]
    homogenitas = graycoprops(glcm, 'homogeneity')[0, 0]
    return kontras, homogenitas

# ==========================================
# 3. FUNGSI UTAMA IDENTIFIKASI & KALIBRASI
# ==========================================
def identifikasi_buah(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"\n[ERROR] Gambar '{image_path}' tidak ditemukan!")
        return

    # Resize gambar
    lebar_target = 500
    tinggi_asli, lebar_asli = img.shape[:2]
    rasio = lebar_target / float(lebar_asli)
    tinggi_target = int(tinggi_asli * rasio)
    img = cv2.resize(img, (lebar_target, tinggi_target))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hasil_identifikasi = "Tidak Dikenali / Tidak Ada Buah"
    mask_terbaik = None
    data_fitur_terbaik = {}
    max_area_terdeteksi = 0 
    objek_terdeteksi = False 
    
    print("\n" + "="*60)
    print(f"MULAI PROSES SCANNING & KALIBRASI: {image_path}")
    print("="*60)

    for nama_buah, data in DATABASE_BUAH.items():
        mask = buat_masker_warna(hsv, data["hsv_ranges"])
        area_piksel = cv2.countNonZero(mask)

        # Jika mendeteksi warna yang cukup luas
        if area_piksel > 1000:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                kontur_terbesar = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(kontur_terbesar)
                keliling = cv2.arcLength(kontur_terbesar, True)
                
                if keliling > 0:
                    objek_terdeteksi = True
                    x, y, w, h = cv2.boundingRect(kontur_terbesar)
                    aspek_rasio = max(float(w)/h, float(h)/w) 
                    circularity = (4 * math.pi * area) / (keliling ** 2)
                    kontras, homogenitas = hitung_tekstur_glcm(gray, x, y, w, h)

                    # --- TAMPILKAN DATA MENTAH SENSOR UNTUK KALIBRASI ---
                    print(f"\n[?] Deteksi Warna Mirip: {nama_buah}")
                    print(f"    Data Mentah Sensor:")
                    print(f"     - Circularity  : {circularity:.4f}  (Mendekati 1 = Bulat)")
                    print(f"     - Aspect Ratio : {aspek_rasio:.4f}  (Mendekati 1 = Persegi/Proporsional)")
                    print(f"     - Kontras GLCM : {kontras:.4f}")
                    
                    rules = data["rules"]
                    match = True
                    alasan_gagal = []
                    
                    # Evaluasi Rules dan Simpan Alasan jika Gagal
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

                    # Penentuan Status
                    if match:
                        print(f"    >>> STATUS: ✅ COCOK (Kriteria Terpenuhi)")
                        if area > max_area_terdeteksi:
                            max_area_terdeteksi = area
                            hasil_identifikasi = nama_buah
                            mask_terbaik = mask
                            data_fitur_terbaik = {
                                "Area (px)": area,
                                "Circularity": circularity,
                                "Aspect Ratio": aspek_rasio,
                                "Kontras": kontras,
                                "Homogenitas": homogenitas
                            }
                    else:
                        print(f"    >>> STATUS: ❌ GAGAL KARENA:")
                        for alasan in alasan_gagal:
                            print(f"        - {alasan}")

    # Peringatan jika warna sama sekali tidak ada yang cocok
    if not objek_terdeteksi:
        print("\n[!] TIDAK ADA WARNA yang terdeteksi cocok dengan database.")
        print("    Saran: Sesuaikan nilai 'hsv_ranges' jika buah ada di dalam gambar namun tidak terlacak.")

    # --- MENAMPILKAN HASIL AKHIR ---
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

    # Visualisasi
    cv2.imshow("1. Gambar Asli", img)
    if mask_terbaik is not None:
        cv2.imshow(f"2. Masker Identifikasi: {hasil_identifikasi}", mask_terbaik)
        img_kotak = img.copy()
        contours, _ = cv2.findContours(mask_terbaik, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(img_kotak, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("3. Bounding Box", img_kotak)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    identifikasi_buah('p1.png')