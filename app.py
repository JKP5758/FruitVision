import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def identifikasi_buah(image_path):
    # --- TAHAP 1: MEMBACA GAMBAR ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"\n[ERROR] Gambar '{image_path}' tidak ditemukan!")
        print("Pastikan nama file benar dan berada di folder yang sama.\n")
        return

    # Mengecilkan gambar 50% agar sesuai layar dan proses lebih ringan 
    # (Sesuai dengan metode di jurnal: resize_factor = 0.5)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # --- TAHAP 2: PRA-PEMROSESAN (KONVERSI WARNA) ---
    # Mengubah ke HSV untuk deteksi warna, dan Grayscale untuk tekstur/bentuk
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- TAHAP 3: EKSTRAKSI FITUR WARNA (Berdasarkan Artikel Jurnal) ---
    # Rentang warna merah untuk Apel
    batas_bawah_merah = np.array([0, 100, 100])
    batas_atas_merah = np.array([10, 255, 255])
    mask_merah = cv2.inRange(hsv, batas_bawah_merah, batas_atas_merah)

    # Rentang warna kuning untuk Pisang
    batas_bawah_kuning = np.array([20, 100, 100])
    batas_atas_kuning = np.array([30, 255, 255])
    mask_kuning = cv2.inRange(hsv, batas_bawah_kuning, batas_atas_kuning)

    # Menghitung jumlah piksel warna yang terdeteksi
    piksel_merah = cv2.countNonZero(mask_merah)
    piksel_kuning = cv2.countNonZero(mask_kuning)

    # --- TAHAP 4: LOGIKA KLASIFIKASI ---
    # Jika piksel warna lebih dari 500, maka buah teridentifikasi
    jenis_buah = "Tidak Dikenali / Tidak Ada Buah"
    mask_aktif = None

    if piksel_merah > 500:
        jenis_buah = "Apel (Merah)"
        mask_aktif = mask_merah
    elif piksel_kuning > 500:
        jenis_buah = "Pisang (Kuning)"
        mask_aktif = mask_kuning

    # --- TAHAP 5: EKSTRAKSI FITUR BENTUK ---
    area = 0
    keliling = 0
    if mask_aktif is not None:
        # Mencari garis tepi (kontur) dari warna yang terdeteksi
        contours, _ = cv2.findContours(mask_aktif, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            kontur_terbesar = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(kontur_terbesar)          # Luas bentuk buah
            keliling = cv2.arcLength(kontur_terbesar, True)  # Keliling bentuk buah

    # --- TAHAP 6: EKSTRAKSI FITUR TEKSTUR (Metode GLCM) ---
    # Menganalisis tingkat kekasaran/kehalusan permukaan buah dari gambar hitam-putih
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    kontras = graycoprops(glcm, 'contrast')[0, 0]
    homogenitas = graycoprops(glcm, 'homogeneity')[0, 0]

    # --- TAHAP 7: MENAMPILKAN HASIL DI TERMINAL ---
    print("\n" + "="*50)
    print(f"HASIL ANALISIS GAMBAR: {image_path}")
    print("="*50)
    print(f"[!] KESIMPULAN     : {jenis_buah}")
    print("-" * 50)
    print(f"1. Analisis Warna  : Merah ({piksel_merah} px), Kuning ({piksel_kuning} px)")
    if mask_aktif is not None:
        print(f"2. Analisis Bentuk : Luas Area ({area} px), Keliling ({keliling:.2f} px)")
    print(f"3. Analisis Tekstur: Kontras ({kontras:.2f}), Homogenitas ({homogenitas:.2f})")
    print("="*50 + "\n")

    # --- TAHAP 8: MENAMPILKAN GAMBAR VISUAL ---
    cv2.imshow("Gambar Asli (Diperkecil)", img)
    if mask_aktif is not None:
        cv2.imshow("Hasil Isolasi Bentuk & Warna", mask_aktif)
    
    print("Tekan tombol 'Q' atau tombol apapun pada jendela gambar untuk menutup program...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================
# AREA UNTUK MENJALANKAN PROGRAM
# ==========================================
if __name__ == "__main__":
    # UBAH 'apel.jpg' DENGAN NAMA FILE GAMBARMU!
    nama_file_gambar = 'apel.jpg' 
    identifikasi_buah(nama_file_gambar)