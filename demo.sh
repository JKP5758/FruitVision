#!/bin/bash
set -e
echo "=== Demo UAS — Identifikasi Buah (Docker-ready) ==="
echo "[1/4] Kalibrasi dari data_buah/..."
python3 kalibrasi.py --data data_buah
echo ""
echo "[2/4] Evaluasi CLI (data_buah)..."
python3 app.py --eval --data data_buah --headless
echo ""
echo "[3/4] Evaluasi CLI (legacy root)..."
python3 app.py --eval --headless
echo ""
echo "[4/4] Test Web API (butuh web.py / docker)..."
if curl -s http://localhost:5000/api/classes > /dev/null 2>&1; then
  echo "  Web up — test upload & predict..."
  curl -s http://localhost:5000/api/classes | python3 -m json.tool | head -20
  curl -s -X POST -F "file=@data_buah/apel/a1.png" http://localhost:5000/upload | python3 -m json.tool | head -20
else
  echo "  Web tidak running — jalankan: python3 web.py  atau  docker compose up -d"
  echo "  Lalu buka http://localhost:5000"
fi
echo ""
echo "Output ada di output/:"
ls -lh output/ 2>&1 | head -20
echo ""
echo "Docker:"
echo "  docker compose up --build -d"
echo "  docker compose logs -f"
