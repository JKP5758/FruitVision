# Identifikasi Buah — UAS Kecerdasan Buatan (Computer Vision + Web Live + Docker)

Sistem klasifikasi **rule-based** ekstensibel berbasis **Warna HSV + Bentuk (Circularity, Aspect Ratio) + Tekstur GLCM**, dengan **dataset terstruktur**, **auto-kalibrasi**, dan **web live camera**.

## Struktur Baru
```
uas/
├── data_buah/                  # Dataset terstruktur (VOLUME Docker)
│   ├── apel/a1.png             # 1+ gambar, minimal 5/kelas disarankan
│   ├── pisang/p1.png
│   ├── jeruk/j1.png
│   └── durian/d1.png           # tambah folder = tambah kelas otomatis (mis. mangga/)
├── app/
│   ├── detector.py             # Core CV (frame-ready, load database.json)
│   └── database.json           # Generated oleh kalibrasi.py (jangan edit manual)
├── kalibrasi.py                # Scan data_buah/ → stats → database.json + kalibrasi.csv
├── web.py                      # Flask server (kamera live + upload)
├── templates/index.html        # getUserMedia + canvas overlay
├── app.py                      # Wrapper CLI (backward compat)
├── requirements.txt            # + flask, watchdog
├── Dockerfile & docker-compose.yml
├── output/                     # Hasil bbox (gitignore)
└── a1.png, ... (legacy seed, juga ada di data_buah/)
```

## Instalasi Lokal (tanpa Docker)
```bash
pip install -r requirements.txt  # atau --break-system-packages
# CLI
python app.py --image data_buah/apel/a1.png --headless
python kalibrasi.py --data data_buah
python app.py --eval --data data_buah --headless  # 4/4 (100%)
# Web
python web.py  # http://localhost:5000
```

## Instalasi Docker (praktis untuk deploy)
```bash
docker compose up --build -d
# buka http://localhost:5000
# log
docker compose logs -f
# stop
docker compose down
```
Volume `data_buah/` & `output/` ter-mount, jadi foto di host langsung terlihat di container tanpa rebuild.

## Alur Dataset Baru
1. Buat folder per buah: `mkdir -p data_buah/mangga`
2. Isi minimal 5 gambar: `cp foto*.jpg data_buah/mangga/`
3. Auto-kalibrasi terpicu otomatis via `watchdog` (2s debounce) atau manual:
   ```bash
   python kalibrasi.py --data data_buah
   # atau via web: tombol 🔄 Kalibrasi Ulang / POST /api/calibrate
   curl -X POST http://localhost:5000/api/calibrate
   ```
   Output: `kalibrasi.csv` (14 baris untuk 4 kelas×1 gambar, 20+ untuk 5/kelas) + `app/database.json`.

4. Verifikasi:
   ```bash
   curl http://localhost:5000/api/classes | jq
   # harus muncul kelas baru, counts, rules
   python app.py --eval --data data_buah --headless
   ```

**Catatan:** Kelas baru (mis. `mangga`) akan pakai HSV seed Durian — perlu tuning `hsv_ranges` di `kalibrasi.py` jika warna jauh berbeda. Auto-kalibrasi hanya tune `rules` (circularity/aspect/contrast), bukan HSV.

## Web Live — Dua Mode Tetap Tersedia
- **Kamera Live:** `http://localhost:5000` → Izinkan kamera → `▶️ Mulai Kamera` → deteksi tiap 400ms via `POST /predict` (base64). Overlay bbox hijau + label.
- **Upload Gambar:** Panel kanan → `📁 Pilih Gambar` → `Upload & Deteksi` → `POST /upload` (multipart) → tampil bbox + fitur. Tetap berfungsi jika kamera ditolak.

API:
- `GET /api/classes` → `{classes, counts, database}`
- `POST /predict` → `{image: "data:image/jpeg;base64,..."}` → `{hasil, bbox, fitur, latency_ms}`
- `POST /upload` → form file → `{hasil, bbox_image (base64), fitur}`

## CLI Tetap Kompatibel
```bash
python app.py --image a1.png                    # jendela cv2
python app.py --image data_buah/jeruk/j1.png --headless
python app.py --eval --headless                 # legacy a1/d1/j1/p1
python app.py --eval --data data_buah --headless # baru, dari folder
python app.py --all --headless
```

## Metode (sinkron dengan app/detector.py)
1. Resize proporsional lebar 500px, threshold area 1% (`detector.py:45-55`)
2. HSV `inRange` + morfologi 5x5
3. Kontur terbesar → `circularity`, `aspect_ratio`
4. GLCM unmasked `distances=[5]` pada ROI 150px max
5. Rule matching 6 kondisi + winner `max_area`
6. Kalibrasi: `mean±std*1.2` jika n≥3, else `min/max+25%` (`kalibrasi.py:60-80`)

## Hasil Evaluasi (4 kelas seed)
```
a1.png | Apel (Merah)          | BENAR
d1.png | Durian                | BENAR
j1.png | Jeruk                 | BENAR
p1.png | Pisang                | BENAR
Akurasi: 4/4 (100.0%) — via data_buah/ maupun root
```

## Demo Checklist (Laptop + Docker)
1. `docker compose up --build -d` → `http://localhost:5000`
2. Tunjukkan `Upload` dengan `j1.png` → `Jeruk` + bbox
3. Tunjukkan `Live Camera` dengan buah asli di depan kamera
4. Tunjukkan `GET /api/classes` + `kalibrasi.csv` sebagai bukti ilmiah
5. Jika tambah buah, demo `mkdir data_buah/mangga && cp 5 foto` → auto-kalibrasi → kelas baru muncul

## Keterbatasan
- HSV belum auto-estimasi untuk kelas baru
- Single-fruit per frame (kontur terbesar)
- GLCM unmasked masih termasuk background

## Lisensi
UAS — bebas modifikasi akademik.
