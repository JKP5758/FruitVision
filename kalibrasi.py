"""Kalibrasi otomatis dari data_buah/(nama_buah)/*.{png,jpg}.

- Scan semua folder di data_buah/ (ekstensibel, tambah buah = tambah folder).
- Minimal 5 gambar/kelas (warning jika kurang).
- Ekstrak fitur cir/aspect/contrast untuk SETIAP gambar di folder GT-nya.
- Hitung mean/std/min/max → usulan rules (mean ± std atau min/max+20% jika n<5).
- Tulis kalibrasi.csv + app/database.json (otomatis reload oleh web.py via watchdog).
- Dukung --data dan --out.

Contoh:
  python kalibrasi.py --data data_buah
  python kalibrasi.py --data data_buah --legacy  # juga evaluasi a1.png legacy
"""
import os, csv, json, math, argparse
from collections import defaultdict
import cv2
import numpy as np

# import setelah ensure app ada
from app.detector import buat_masker_warna, hitung_tekstur_glcm, get_database, DEFAULT_DATABASE, scan_data_buah

def glcm_unmasked(gray, x,y,w,h):
    roi = gray[y:y+h, x:x+w]
    if roi.size==0:
        return 0,0
    if max(roi.shape)>150:
        scale=150/max(roi.shape)
        roi=cv2.resize(roi,(int(roi.shape[1]*scale),int(roi.shape[0]*scale)))
    from skimage.feature import graycomatrix, graycoprops
    g=graycomatrix(roi, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    return graycoprops(g,'contrast')[0,0], graycoprops(g,'homogeneity')[0,0]

def extract_for_image(img_path, database):
    img=cv2.imread(img_path)
    if img is None:
        return None
    img=cv2.resize(img,(500,int(img.shape[0]*500/img.shape[1])))
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # kita ekstrak hanya untuk kandidat yang sesuai label? Tapi untuk kalibrasi,
    # kita ekstrak untuk SEMUA kandidat untuk lihat confusion, tapi stats hanya dari GT.
    # Disini untuk efficiency, kita ekstrak untuk database keys saja.
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
        kontras,_=glcm_unmasked(gray,x,y,w,h)
        results.append((label, area, area_c, circ, ar, kontras))
    return results

def main():
    parser=argparse.ArgumentParser(description="Kalibrasi otomatis data_buah")
    parser.add_argument("--data", type=str, default="data_buah", help="Root data_buah")
    parser.add_argument("--out-csv", type=str, default="kalibrasi.csv", help="Output CSV")
    parser.add_argument("--out-json", type=str, default="app/database.json", help="Output database.json")
    parser.add_argument("--legacy", action="store_true", help="Juga kalibrasi a1.png legacy (untuk demo)")
    args=parser.parse_args()

    db = get_database()  # fallback DEFAULT kalau belum ada json
    scan = scan_data_buah(args.data)
    if not scan:
        print(f"[ERROR] Tidak ada data di {args.data}/. Buat folder per buah: data_buah/apel/*.jpg")
        # fallback legacy jika diminta
        if args.legacy:
            from app import LABEL_GROUND_TRUTH
            scan={}
            for f, label in LABEL_GROUND_TRUTH.items():
                if os.path.exists(f):
                    scan.setdefault(label, []).append(f)
        else:
            return

    rows=[]
    # kumpulkan stats per GT label (hanya dari gambar di folder GT itu, bukan semua kandidat)
    stats_raw=defaultdict(list)  # label -> list[(circ,ar,kontras)]

    for label, paths in scan.items():
        n=len(paths)
        print(f"\n=== {label} : {n} gambar ===")
        if n < 5:
            print(f"  [WARN] Minimal 5 gambar disarankan, baru {n}. Margin akan lebih lebar.")
        for p in sorted(paths):
            res=extract_for_image(p, db)
            if not res:
                print(f"  {os.path.basename(p)} : tidak terdeteksi warna")
                continue
            # cari entry yang match label (GT)
            gt_entry = next((x for x in res if x[0]==label), None)
            # jika GT tidak terdeteksi tapi kandidat lain terdeteksi, tetap log semua untuk CSV
            for cand_label, area, area_c, circ, ar, kontras in res:
                is_gt = int(cand_label==label)
                print(f"  {os.path.basename(p):15} cand={cand_label:22} circ={circ:.3f} ar={ar:.2f} kontr={kontras:.0f} {'<<GT' if is_gt else ''}")
                rows.append([p,label,cand_label,area,area_c,circ,ar,kontras,is_gt])
                if is_gt:
                    stats_raw[label].append((circ, ar, kontras))

    # tulis csv
    with open(args.out_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["file","gt","candidate","mask_area","contour_area","circularity","aspect_ratio","contrast","is_gt"])
        w.writerows(rows)
    print(f"\n[INFO] {args.out_csv} tertulis {len(rows)} baris")

    # hitung stats & usulan rules
    new_db={}
    print("\n=== STATISTIK GT ===")
    for label in scan.keys():
        vals=stats_raw.get(label, [])
        if not vals:
            print(f"{label}: TIDAK ADA DATA GT TERDETEKSI -> pakai default")
            # pakai default
            new_db[label]=DEFAULT_DATABASE.get(label, list(DEFAULT_DATABASE.values())[0])
            # tapi ganti label tetap
            if label not in DEFAULT_DATABASE:
                # buat default baru untuk buah baru: pakai HSV default durian sebagai seed (akan perlu tuning manual)
                new_db[label]={
                    "hsv_ranges": DEFAULT_DATABASE["Durian"]["hsv_ranges"],
                    "rules": {"min_circularity":0.3,"max_circularity":0.9,"min_aspect_ratio":1.0,"max_aspect_ratio":2.0,"min_contrast":500,"max_contrast":2500}
                }
            continue
        circs=[v[0] for v in vals]
        ars=[v[1] for v in vals]
        cons=[v[2] for v in vals]
        mean_circ=np.mean(circs); std_circ=np.std(circs)
        mean_ar=np.mean(ars); std_ar=np.std(ars)
        mean_k=np.mean(cons); std_k=np.std(cons)
        print(f"{label} (n={len(vals)}):")
        print(f"  circ {mean_circ:.3f}±{std_circ:.3f} range [{min(circs):.3f},{max(circs):.3f}]")
        print(f"  ar   {mean_ar:.3f}±{std_ar:.3f} range [{min(ars):.3f},{max(ars):.3f}]")
        print(f"  kontr {mean_k:.0f}±{std_k:.0f} range [{min(cons):.0f},{max(cons):.0f}]")

        # usulan rules
        # jika n>=3 pakai mean±std, jika n<3 pakai min/max + margin 25%
        if len(vals) >= 3 and std_circ>0:
            c_lo = max(0, mean_circ - std_circ*1.2)
            c_hi = min(1.0, mean_circ + std_circ*1.2)
            a_lo = max(0.5, mean_ar - std_ar*1.2)
            a_hi = mean_ar + std_ar*1.2
            k_lo = max(0, mean_k - std_k*1.2)
            k_hi = mean_k + std_k*1.2
        else:
            # margin 25%
            def margin_range(vs, pct=0.25):
                lo=min(vs); hi=max(vs)
                m=(hi-lo)*pct if hi!=lo else abs(hi)*pct
                if m==0: m=0.05
                return max(0,lo-m), hi+m
            c_lo,c_hi=margin_range(circs,0.25)
            a_lo,a_hi=margin_range(ars,0.25)
            k_lo,k_hi=margin_range(cons,0.25)
            c_hi=min(1.0,c_hi)

        # bulatkan & clamp
        c_lo=round(float(c_lo),2); c_hi=round(float(c_hi),2)
        a_lo=round(float(a_lo),2); a_hi=round(float(a_hi),2)
        k_lo=int(max(0, round(k_lo))); k_hi=int(round(k_hi))

        # HSV ranges tetap dari db lama (atau default)
        if label in db:
            hsv_ranges=db[label]["hsv_ranges"]
        elif label in DEFAULT_DATABASE:
            hsv_ranges=DEFAULT_DATABASE[label]["hsv_ranges"]
        else:
            hsv_ranges=DEFAULT_DATABASE["Durian"]["hsv_ranges"]
            print(f"  [WARN] Buah baru {label} pakai HSV seed Durian — perlu tuning manual hsv_ranges")

        new_db[label]={
            "hsv_ranges": hsv_ranges,
            "rules": {
                "min_circularity": c_lo,
                "max_circularity": c_hi,
                "min_aspect_ratio": a_lo,
                "max_aspect_ratio": a_hi,
                "min_contrast": k_lo,
                "max_contrast": k_hi,
            }
        }
        print(f"  -> rules: circ {c_lo}-{c_hi} ar {a_lo}-{a_hi} kontr {k_lo}-{k_hi}")

    # simpan database.json (convert np.array -> list)
    out_json={}
    for label, v in new_db.items():
        # hsv_ranges sudah numpy, convert
        lst=[]
        for lo, hi in v["hsv_ranges"]:
            # lo/hi mungkin sudah np.array
            lst.append([lo.tolist() if hasattr(lo, "tolist") else list(lo),
                        hi.tolist() if hasattr(hi, "tolist") else list(hi)])
        out_json[label]={"hsv_ranges": lst, "rules": v["rules"]}

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json,"w") as f:
        json.dump(out_json,f,indent=2)
    print(f"\n[INFO] {args.out_json} tertulis ({len(out_json)} kelas)")
    print(json.dumps({k:v["rules"] for k,v in out_json.items()}, indent=2))

    # also simpan ke database.json di root untuk fallback
    if args.out_json != "database.json":
        try:
            with open("database.json","w") as f:
                json.dump(out_json,f,indent=2)
        except: pass

if __name__=="__main__":
    main()
