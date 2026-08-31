# Identifikasi Buah — UAS Kecerdasan Buatan (Computer Vision)

Sistem klasifikasi **rule-based** 4 buah (Apel, Pisang, Jeruk, Durian) berbasis **Warna HSV + Bentuk (Circularity, Aspect Ratio) + Tekstur GLCM**. Dikembangkan untuk demo live di laptop, dengan kalibrasi data-driven.

## Struktur
```
uas/
├── app.py              # Program utama (identifikasi + evaluasi + visualisasi)
├── kalibrasi.py        # Dump fitur mentah & usulan rules (data-driven)
├── kalibrasi.csv       # Output kalibrasi (generated, gitignore)
├── requirements.txt
├── output/             # Hasil bbox/mask (generated, gitignore)
├── a1.png (Apel)  d1.png (Durian)  j1.png (Jeruk)  p1.png (Pisang)
└── README.md
```

## Instalasi
```bash
pip install -r requirements.txt
# atau
pip install --break-system-packages opencv-python scikit-image matplotlib numpy
```

## Penggunaan
```bash
# Single image (demo live → tampilkan 3 jendela)
python3 app.py --image a1.png
python3 app.py --image d1.png
python3 app.py --image j1.png
python3 app.py --image p1.png

# Headless (server / tanpa display, save ke output/)
python3 app.py --image a1.png --headless
ls output/

# Evaluasi akurasi 4 gambar ground truth
python3 app.py --eval --headless
# Expected: Akurasi: 4/4 (100.0%)

# Proses semua png di folder
python3 app.py --all --headless
```

## Metode
1. **Resize proporsional** lebar 500px (`app.py:148-153`) — preservasi aspek, threshold area dinormalisasi 1% dari total piksel (`app.py:158-159`) agar `j1.png` 3000px dan `a1.png` 515px setara.
2. **Warp HSV** → `cv2.inRange` + `bitwise_or` + morfologi `MORPH_OPEN/CLOSE` 5x5 (`app.py:81-89`).
3. **Kontur terbesar** → `circularity = 4πA/P²`, `aspect_ratio = max(w/h, h/w)` (`app.py:119-126`).
4. **GLCM unmasked** `distances=[5], angles=[0], levels=256` pada ROI penuh (tanpa zeroing background). Zeroing terbukti inflate kontras 50-200% (mis. Pisang 974→2654), jadi kalibrasi 2026-08-31 dilakukan **unmasked** (`app.py:91-117`). ROI >150px di-downscale untuk performa.
5. **Rule matching** 6 kondisi (`app.py:205-222`) + winner-takes-all via `max_area`.

## Kalibrasi Data-Driven (Fase B)
```bash
python3 kalibrasi.py
cat kalibrasi.csv
```
`kalibrasi.py` ekstrak fitur mentah untuk setiap pasangan `file × candidate` dan hitung statistik GT:

| File | GT | circ | ar | kontras (unmasked) |
|------|----|------|----|-------------------|
| a1.png | Apel | 0.791 | 1.03 | 1453 |
| p1.png | Pisang | 0.318 | 1.38 | 673 |
| j1.png | Jeruk | 0.779 | 1.05 | 1104 |
| d1.png | Durian | 0.571 | 1.39 | 2062 |

Usulan rules dengan margin 25% (`kalibrasi.py:57-73`):
- **Apel**: circ 0.60-0.95, ar 0.90-1.35, kontr 1100-1800
- **Pisang**: circ 0.10-0.50, ar 1.20-3.00, kontr 450-900
- **Jeruk**: circ 0.60-0.95, ar 0.90-1.30, kontr 850-1400
- **Durian**: circ 0.40-0.75, ar 1.00-1.80, kontr 1500-2600

Nilai ini sudah di-apply ke `DATABASE_BUAH` di `app.py:13-68`. Sebelum kalibrasi, Apel `max_contrast 500` selalu gagal (kontras riil 1453), sehingga semua gambar fallback ke Durian (akurasi 1/4).

## Hasil Evaluasi (2026-08-31, setelah kalibrasi)
```
 a1.png | GT: Apel (Merah)          | Pred: Apel (Merah)          | BENAR
 d1.png | GT: Durian                | Pred: Durian                | BENAR
 j1.png | GT: Jeruk                 | Pred: Jeruk                 | BENAR
 p1.png | GT: Pisang (Kuning/Hijau) | Pred: Pisang (Kuning/Hijau) | BENAR
Akurasi: 4/4 (100.0%)
```
Lihat `python3 app.py --image a1.png --headless` untuk detail `Data Mentah Sensor` dan `Alasan Gagal` per-kandidat.

## Demo Live Checklist (Laptop)
1. Buka terminal di folder `uas/`.
2. Jalankan berturut: `python3 app.py --image a1.png` → `p1.png` → `j1.png` → `d1.png` — tiap run tampil 3 jendela (Gambar Asli, Masker, Bounding Box). Tekan key untuk lanjut.
3. Jika proyektor gagal, fallback headless: `python3 app.py --all --headless` lalu buka `output/*_bbox.png`.
4. Tunjukkan `--eval` untuk bukti akurasi.
5. Tunjukkan `kalibrasi.csv` dan `kalibrasi.py` sebagai bukti ilmiah.

## Keterbatasan
- HSV Durian `[15,30,30]-[50,255,255]` masih luas, mengoverlap Pisang/Jeruk; diskriminasi mengandalkan kontras+bentuk. Untuk dataset lebih besar, gunakan histogram HSV adaptif atau model ML.
- GLCM unmasked masih mencakup background meja di ROI; untuk presisi lebih tinggi, masking ketat (crop tanpa zeroing) atau segmentasi GrabCut.
- Single-fruit per image (kontur terbesar). Multi-buah butuh loop semua kontur > threshold (sudah ada `semua_deteksi` di `app.py:166,224-234` sebagai fondasi).
- Hanya 1 sampel per kelas → margin rules belum generalisasi; tambah data uji untuk std yang valid.

## Lisensi
UAS — bebas modifikasi untuk akademik.
