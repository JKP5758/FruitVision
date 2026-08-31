#!/bin/bash
set -e
echo "=== Demo UAS Kecerdasan Buatan - Identifikasi Buah ==="
echo "[1/3] Kalibrasi..."
python3 kalibrasi.py
echo ""
echo "[2/3] Evaluasi akurasi..."
python3 app.py --eval --headless
echo ""
echo "[3/3] Proses semua gambar (headless)..."
python3 app.py --all --headless
echo ""
echo "Output ada di output/:"
ls -lh output/
echo ""
echo "Untuk demo live (dengan jendela):"
echo "  python3 app.py --image a1.png"
echo "  python3 app.py --image p1.png"
echo "  python3 app.py --image j1.png"
echo "  python3 app.py --image d1.png"
