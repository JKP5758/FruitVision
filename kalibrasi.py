"""Kalibrasi probabilistik — tanpa hard-code, tanpa threshold 5.

- Scan data_buah/(nama_buah)/*.{jpg,png} (ekstensibel)
- Untuk tiap kelas: ekstrak HSV mean/std + fitur circ/aspect/contrast
- Simpan stats (mean, std, min, max, n) + hsv_ranges auto (mean±2std)
- Output: kalibrasi.csv + app/database.json (probabilistik)
"""
import os, csv, json, math, argparse
from collections import defaultdict
import cv2
import numpy as np

from app.detector import buat_masker_warna, hitung_tekstur_glcm, get_database, DEFAULT_DATABASE, scan_data_buah

def extract_for_image(img_path, database):
    img=cv2.imread(img_path)
    if img is None:
        return None
    img=cv2.resize(img,(500,int(img.shape[0]*500/img.shape[1])))
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    results=[]
    for label, data in database.items():
        mask=buat_masker_warna(hsv,data["hsv_ranges"])
        area=cv2.countNonZero(mask)
        if area < 0.01*img.shape[0]*img.shape[1]:
            continue
        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c=max(cnts,key=cv2.contourArea)
        area_c=cv2.contourArea(c)
        peri=cv2.arcLength(c,True)
        if peri==0:
            continue
        x,y,w,h=cv2.boundingRect(c)
        circ=(4*math.pi*area_c)/(peri**2)
        ar=max(w/h,h/w) if h else 0
        kontras,_=hitung_tekstur_glcm(gray,x,y,w,h, mask)
        # HSV stats di dalam mask
        hsv_pixels = hsv[mask>0]
        if len(hsv_pixels)>0:
            mean_hsv = np.mean(hsv_pixels, axis=0)
            std_hsv = np.std(hsv_pixels, axis=0)
        else:
            mean_hsv = np.array([0,0,0], dtype=float)
            std_hsv = np.array([0,0,0], dtype=float)
        results.append((label, area, area_c, circ, ar, kontras, mean_hsv, std_hsv))
    return results

def main():
    parser=argparse.ArgumentParser(description="Kalibrasi probabilistik")
    parser.add_argument("--data", type=str, default="data_buah", help="Root data_buah")
    parser.add_argument("--out-csv", type=str, default="kalibrasi.csv")
    parser.add_argument("--out-json", type=str, default="app/database.json")
    args=parser.parse_args()

    db = get_database()
    scan = scan_data_buah(args.data)
    if not scan:
        print(f"[ERROR] Tidak ada data di {args.data}/")
        return

    rows=[]
    stats_raw=defaultdict(list)
    hsv_raw=defaultdict(list)  # label -> list[mean_hsv]

    for label, paths in scan.items():
        n=len(paths)
        print(f"\n=== {label} : {n} gambar ===")
        if n < 5:
            print(f"  [INFO] n={n} (<5 saran, tapi tetap diproses — tanpa batas mutlak)")
        for p in sorted(paths):
            res=extract_for_image(p, db)
            if not res:
                print(f"  {os.path.basename(p)} : tidak terdeteksi warna")
                continue
            for cand_label, area, area_c, circ, ar, kontras, mean_hsv, std_hsv in res:
                is_gt = int(cand_label==label)
                print(f"  {os.path.basename(p):15} cand={cand_label:22} circ={circ:.3f} ar={ar:.2f} kontr={kontras:.0f} hsv_mean={mean_hsv.astype(int)} {'<<GT' if is_gt else ''}")
                rows.append([p,label,cand_label,area,area_c,circ,ar,kontras,mean_hsv[0],mean_hsv[1],mean_hsv[2],is_gt])
                if is_gt:
                    stats_raw[label].append((circ, ar, kontras))
                    hsv_raw[label].append(mean_hsv)

    with open(args.out_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["file","gt","candidate","mask_area","contour_area","circularity","aspect_ratio","contrast","h_mean","s_mean","v_mean","is_gt"])
        w.writerows(rows)
    print(f"\n[INFO] {args.out_csv} {len(rows)} baris")

    new_db={}
    print("\n=== STATISTIK PROBABILISTIK ===")
    for label in scan.keys():
        vals=stats_raw.get(label, [])
        hsv_vals=hsv_raw.get(label, [])
        if not vals:
            print(f"{label}: TIDAK ADA DATA GT -> pakai default seed")
            new_db[label]=DEFAULT_DATABASE.get(label, list(DEFAULT_DATABASE.values())[0])
            # convert to stats format
            if "stats" not in new_db[label]:
                # buat stats dummy dari rules
                r=new_db[label]["rules"]
                new_db[label]["stats"]={
                    "mean_circularity": (r["min_circularity"]+r["max_circularity"])/2,
                    "std_circularity": (r["max_circularity"]-r["min_circularity"])/4,
                    "mean_aspect_ratio": (r["min_aspect_ratio"]+r["max_aspect_ratio"])/2,
                    "std_aspect_ratio": (r["max_aspect_ratio"]-r["min_aspect_ratio"])/4,
                    "mean_contrast": (r["min_contrast"]+r["max_contrast"])/2,
                    "std_contrast": (r["max_contrast"]-r["min_contrast"])/4,
                    "min_circularity": r["min_circularity"],
                    "max_circularity": r["max_circularity"],
                    "min_aspect_ratio": r["min_aspect_ratio"],
                    "max_aspect_ratio": r["max_aspect_ratio"],
                    "min_contrast": r["min_contrast"],
                    "max_contrast": r["max_contrast"],
                    "n": 1
                }
            continue
        circs=[v[0] for v in vals]; ars=[v[1] for v in vals]; cons=[v[2] for v in vals]
        mean_circ=float(np.mean(circs)); std_circ=float(np.std(circs) if len(circs)>1 else 0.15*mean_circ if mean_circ>0 else 0.05)
        mean_ar=float(np.mean(ars)); std_ar=float(np.std(ars) if len(ars)>1 else 0.15*mean_ar)
        mean_k=float(np.mean(cons)); std_k=float(np.std(cons) if len(cons)>1 else 0.2*mean_k if mean_k>0 else 50)
        # clamp std minimal
        std_circ=max(std_circ, 0.03); std_ar=max(std_ar, 0.05); std_k=max(std_k, 30)

        # HSV: keep existing ranges (70) — auto HSV disabled to avoid drift (user wants stable)
        # Jika ingin auto, quality data_buah harus bersih; untuk 3 sampel riil, keep 70 stabil
        if label in db:
            hsv_ranges=db[label]["hsv_ranges"]
        elif label in DEFAULT_DATABASE:
            hsv_ranges=DEFAULT_DATABASE[label]["hsv_ranges"]
        else:
            hsv_ranges=DEFAULT_DATABASE["Durian"]["hsv_ranges"]
        # Convert to list for json later, but keep numpy for now

        # Simpan stats
        print(f"{label} (n={len(vals)}): circ {mean_circ:.3f}±{std_circ:.3f} range [{min(circs):.3f},{max(circs):.3f}]")
        print(f"  ar {mean_ar:.3f}±{std_ar:.3f} range [{min(ars):.3f},{max(ars):.3f}]")
        print(f"  kontr {mean_k:.0f}±{std_k:.0f} range [{min(cons):.0f},{max(cons):.0f}]")
        print(f"  hsv mean {mean_hsv.astype(int)} std {std_hsv.astype(int)} -> ranges {hsv_ranges}")

        stats={
            "mean_circularity": mean_circ,
            "std_circularity": std_circ,
            "mean_aspect_ratio": mean_ar,
            "std_aspect_ratio": std_ar,
            "mean_contrast": mean_k,
            "std_contrast": std_k,
            "min_circularity": float(min(circs)),
            "max_circularity": float(max(circs)),
            "min_aspect_ratio": float(min(ars)),
            "max_aspect_ratio": float(max(ars)),
            "min_contrast": float(min(cons)),
            "max_contrast": float(max(cons)),
            "n": len(vals),
            "hsv_mean": mean_hsv.tolist(),
            "hsv_std": std_hsv.tolist()
        }
        # rules tetap dibuat untuk backward compat (min/max + margin), tapi tidak dipakai utama
        margin=0.25
        def mrange(vs):
            lo=min(vs); hi=max(vs); m=(hi-lo)*margin if hi!=lo else abs(hi)*margin
            return lo-m, hi+m
        c_lo,c_hi=mrange(circs); a_lo,a_hi=mrange(ars); k_lo,k_hi=mrange(cons)
        rules={"min_circularity": round(c_lo,2),"max_circularity": round(min(1.0,c_hi),2),
               "min_aspect_ratio": round(a_lo,2),"max_aspect_ratio": round(a_hi,2),
               "min_contrast": int(k_lo),"max_contrast": int(k_hi)}
        new_db[label]={"hsv_ranges": hsv_ranges, "stats": stats, "rules": rules}

    out_json={}
    for label, v in new_db.items():
        lst=[]
        for lo, hi in v["hsv_ranges"]:
            lst.append([lo.tolist() if hasattr(lo, "tolist") else list(lo),
                        hi.tolist() if hasattr(hi, "tolist") else list(hi)])
        out_json[label]={"hsv_ranges": lst, "stats": v["stats"], "rules": v["rules"]}

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json,"w") as f:
        json.dump(out_json,f,indent=2)
    print(f"\n[INFO] {args.out_json} {len(out_json)} kelas (probabilistik)")
    # also save root
    if args.out_json != "database.json":
        with open("database.json","w") as f:
            json.dump(out_json,f,indent=2)

if __name__=="__main__":
    main()
