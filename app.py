"""Wrapper CLI — backward compat. Core logic ada di app/detector.py"""
import os
import argparse
import sys
# agar bisa import app.detector saat dijalankan langsung
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.detector import (
    get_database, identifikasi_buah, identifikasi_frame,
    scan_data_buah, DEFAULT_DATABASE
)

# expose DATABASE_BUAH untuk kalibrasi lama
DATABASE_BUAH = get_database()
LABEL_GROUND_TRUTH = {
    "a1.png": "Apel (Merah)",
    "d1.png": "Durian",
    "j1.png": "Jeruk",
    "p1.png": "Pisang (Kuning/Hijau)",
}

def evaluasi_semua(headless=True, save_dir="output", data_root=None):
    db = get_database()
    # jika data_root diberikan, evaluasi berdasarkan data_buah/
    if data_root:
        scan = scan_data_buah(data_root)
        # buat mapping file -> gt dari folder
        files_gt = []
        for label, paths in scan.items():
            for p in paths:
                files_gt.append((p, label))
        if not files_gt:
            print(f"[WARN] Tidak ada data di {data_root}")
            return 0
        benar = 0
        print("\n" + "="*60)
        print(f"EVALUASI AKURASI — data: {data_root}")
        print("="*60)
        for f, gt in sorted(files_gt):
            res = identifikasi_buah(f, verbose=False, headless=headless, save_dir=save_dir, database=db)
            pred = res["hasil"]
            ok = (pred == gt)
            benar += int(ok)
            status = "BENAR" if ok else "SALAH"
            print(f" {os.path.basename(f):12} | GT: {gt:22} | Pred: {pred:22} | {status}")
        acc = benar / len(files_gt) * 100 if files_gt else 0
        print("-"*60)
        print(f"Akurasi: {benar}/{len(files_gt)} ({acc:.1f}%)")
        print("="*60 + "\n")
        return acc
    # fallback legacy: eval a1/j1/p1/d1 di root
    files = [f for f in os.listdir(".") if f.lower().endswith(".png") and f in LABEL_GROUND_TRUTH]
    if not files:
        print("[WARN] Tidak ada file png ground truth di root.")
        return 0
    benar = 0
    print("\n" + "="*60)
    print("EVALUASI AKURASI SEMUA GAMBAR (legacy root)")
    print("="*60)
    for f in sorted(files):
        gt = LABEL_GROUND_TRUTH[f]
        res = identifikasi_buah(f, verbose=False, headless=headless, save_dir=save_dir, database=db)
        pred = res["hasil"]
        ok = (pred == gt)
        benar += int(ok)
        status = "BENAR" if ok else "SALAH"
        print(f" {f:8} | GT: {gt:22} | Pred: {pred:22} | {status}")
    acc = benar / len(files) * 100 if files else 0
    print("-"*60)
    print(f"Akurasi: {benar}/{len(files)} ({acc:.1f}%)")
    print("="*60 + "\n")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identifikasi Buah - Demo UAS Kecerdasan Buatan")
    parser.add_argument("--image", type=str, default="d1.png", help="Path gambar")
    parser.add_argument("--headless", action="store_true", help="Tanpa imshow, hanya save ke output/")
    parser.add_argument("--eval", action="store_true", help="Evaluasi ground truth")
    parser.add_argument("--data", type=str, default=None, help="Root data_buah untuk eval (mis. data_buah)")
    parser.add_argument("--save-dir", type=str, default="output", help="Folder output")
    parser.add_argument("--all", action="store_true", help="Proses semua png di folder")
    args = parser.parse_args()

    if args.eval:
        evaluasi_semua(headless=args.headless, save_dir=args.save_dir, data_root=args.data)
    elif args.all:
        for f in sorted([x for x in os.listdir(".") if x.lower().endswith(".png")]):
            identifikasi_buah(f, verbose=True, headless=args.headless, save_dir=args.save_dir)
    else:
        identifikasi_buah(args.image, verbose=True, headless=args.headless, save_dir=args.save_dir)
