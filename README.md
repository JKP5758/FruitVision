# Identifikasi Buah — UAS Kecerdasan Buatan (Computer Vision + Web Live + Docker + Cloudflare Tunnel)

Sistem klasifikasi **probabilistik** ekstensibel berbasis **Warna HSV + Bentuk (Circularity, Aspect Ratio) + Tekstur GLCM**, dengan **dataset terstruktur** (`data_buah/(nama_buah)/`), **auto-kalibrasi** (reset sebelum kalibrasi, fleksibel, tanpa batas 5 mutlak), **web live camera + upload**, dan **probabilitas % kemiripan**.

## Struktur
```
uas/
├── data_buah/                  # Dataset terstruktur (VOLUME Docker) — fleksibel
│   ├── apel/P_...039.jpg       # 3 gambar (contoh)
│   └── pisang/P_...355.jpg     # 5 gambar — tambah folder = tambah kelas (mangga/)
├── app/
│   ├── detector.py             # Core CV probabilistik (tanpa hard-code, load database.json)
│   └── database.json           # Generated kalibrasi.py (stats mean/std, jangan edit manual)
├── kalibrasi.py                # Scan data_buah/ → stats → database.json + kalibrasi.csv (--reset)
├── web.py                      # Flask (kamera live + upload + auto-kalibrasi reset)
├── templates/index.html        # getUserMedia + overlay + Deteksi Sekali auto-stop + bar probabilitas
├── app.py                      # Wrapper CLI (eval probabilistik)
├── requirements.txt            # + flask, watchdog
├── Dockerfile & docker-compose.yml # + cloudflared sidecar
├── .env.example / .env         # TUNNEL_TOKEN (jangan commit .env)
├── output/                     # Hasil bbox (gitignore)
└── README.md
```

## Fitur
- **Dataset fleksibel:** `mkdir data_buah/mangga && cp foto.jpg` → auto terdeteksi, `GET /api/classes` muncul `Mangga`. Tidak hard-code 4 buah.
- **Reset kalibrasi:** sebelum kalibrasi ulang `rm app/database.json` (via `kalibrasi.py --reset`, `POST /api/calibrate`, `docker start`) — hapus ghost class jika foto/folder dihapus.
- **Probabilistik:** bukan `Tidak Dikenali` biner, tapi `Apel (Merah) (93.5%)` + bar `Apel 93% | Pisang 1%` (Gaussian `exp(-0.5*z²)`). Jika `max<30%` → `Tidak ada buah terdeteksi (tertinggi Apel 24%)`.
- **Dua mode web:** Kamera Live (400ms) + Upload — keduanya `/predict` (base64) & `/upload` (multipart).
- **Deteksi Sekali auto-stop:** `📸 Deteksi Sekali` → 1 jepret → kamera mati (hemat).
- **Docker + Cloudflare Tunnel:** `web:5000` lokal tetap + `https://<subdomain>` via `cloudflared` sidecar.

## Instalasi Lokal (tanpa Docker)
```bash
pip install -r requirements.txt  # atau --break-system-packages
python kalibrasi.py --data data_buah --reset  # 8/8 (3 Apel 40-93% +5 Pisang) tanpa a1
python app.py --eval --data data_buah --headless
python web.py  # http://localhost:5000
```

## Instalasi Docker (praktis)
```bash
docker compose up --build -d
# buka http://localhost:5000
docker compose logs -f
docker compose down
```
Volume `data_buah/` & `output/` & `app/database.json` ter-mount.

## Alur Dataset
1. `mkdir -p data_buah/mangga && cp foto*.jpg data_buah/mangga/` (minimal 1, saran 5)
2. Auto-kalibrasi via `watchdog` 2s debounce (file/folder create/delete) + `docker start` + tombol `🔄 Kalibrasi Ulang` — semua dengan `reset`.
   ```bash
   python kalibrasi.py --data data_buah --reset
   curl -X POST http://localhost:5000/api/calibrate
   ```
3. Verifikasi: `curl http://localhost:5000/api/classes | jq` + `python app.py --eval --data data_buah`.

## Deploy via Cloudflare Tunnel + Subdomain

### Prasyarat
- Domain sudah di Cloudflare (nameserver aktif).
- Subdomain sudah Anda siapkan (mis. `buah.example.com`) — ganti `TUNNEL_HOSTNAME` di bawah.
- Token sudah Anda siapkan dari **Cloudflare Zero Trust > Networks > Tunnels > Create Tunnel (Cloudflared) > Copy Token** — token `ey...`.

### Setup Token & Service URL
- **Token placement:** `cp .env.example .env` → edit `.env`:
  ```
  TUNNEL_TOKEN=eyJhIjoi... (token Anda)
  TUNNEL_HOSTNAME=buah.example.com
  PORT=5000
  ```
  **Jangan commit `.env`** (sudah di `.gitignore` + `.dockerignore`). `.env.example` sebagai template.
- **Service URL:** di **Dashboard Tunnel > Public Hostname > Add**:
  - `Hostname: <subdomain Anda>` (mis. `buah.example.com`)
  - `Service Type: HTTP` `URL: http://web:5000`  **(bukan localhost!)** — `web` adalah nama service Docker, resolve via network `uas_default`. Jika pakai host install `cloudflared` (tanpa Docker), baru pakai `http://localhost:5000`.

### Menjalankan
```bash
cp .env.example .env  # isi TUNNEL_TOKEN
docker compose up --build -d
docker compose logs -f cloudflared  # harus Registered tunnel connection, 502 jika Service URL salah
docker compose logs -f web
# lokal tetap: http://localhost:5000
# via tunnel: https://<subdomain> (mis. https://buah.example.com)
curl -I https://buah.example.com/api/classes  # 200 + cf-ray
```

### Compose
`docker-compose.yml` sekarang 2 service:
```yaml
services:
  web:
    build: .
    ports: ["5000:5000"]  # keep lokal
    volumes: [...]
  cloudflared:
    image: cloudflare/cloudflared:latest
    command: tunnel --no-autoupdate run --token ${TUNNEL_TOKEN}
    depends_on: [web]
    restart: unless-stopped
    env_file: [.env]
```

### Troubleshooting
- `502 Bad Gateway` → Service URL salah (pakai `localhost` di sidecar) → ganti `http://web:5000`.
- `connection registered` tidak muncul → `TUNNEL_TOKEN` salah/kadaluarsa → re-copy dari dashboard.
- `Access Zero Trust` jika ingin proteksi login → dashboard `Access > Add application`.

## Web Live
- **Kamera Live:** `http://localhost:5000` atau `https://<subdomain>` → `▶️ Mulai Kamera` (live 400ms) atau `📸 Deteksi Sekali` (1 jepret auto-stop).
- **Upload:** `📁 Pilih Gambar` → `Upload & Deteksi` → bbox + bar probabilitas.

API: `GET /api/classes`, `POST /predict` (base64), `POST /upload` (multipart), `POST /api/calibrate` (reset).

## CLI
```bash
python app.py --image data_buah/apel/P_...039.jpg --headless
python app.py --eval --data data_buah --headless  # 8/8
```

## Metode
1. Resize 500px, threshold 1%
2. HSV inRange + morfologi 5x5 (Apel S/V 70)
3. Kontur → circularity, aspect
4. GLCM (150px max) → contrast
5. **Probabilistik Gaussian** `prob = mean(exp(-0.5*z²))` per fitur, bukan min/max hard-code. `z=|x-mean|/std`, `std` clamp minimal. `max<30%` → Tidak ada buah.

## Hasil Evaluasi (tanpa a1, 3 Apel +5 Pisang)
```
P_039 → Apel 93.5% BENAR, P_054 → 60.1% BENAR, P_145 → 40.0% BENAR
Pisang 5 → 51-88% BENAR
Akurasi: 8/8 (100%)
# Sebelum probabilistik: 1/3 Apel (hanya P_054) karena mean±1.2std sempit
```

## Keterbatasan
- HSV untuk kelas baru seed Durian — perlu tuning jika warna jauh.
- Single-fruit per frame.

## Lisensi
UAS — bebas modifikasi akademik.
