"""Kalibrasi dinamis multi-buah — GrabCut + augmentasi + templates hist/Hu.

- Scan data_buah/(nama_buah)/*
- Untuk tiap foto: GrabCut foreground -> hist HSV/LAB + Hu + LBP + GLCM
- Augmentasi n=2 -> n~10 via albumentations (rotate, flip, brightness, HSV)
- Simpan templates per kelas (bukan mean/std saja) + hsv_ranges untuk fallback
- Output: kalibrasi.csv + app/database.json
"""
import os, csv, json, math, argparse, random
from collections import defaultdict
import cv2
import numpy as np

from app.detector import hitung_tekstur_glcm, scan_data_buah, buat_masker_warna

try:
    import albumentations as A
    HAS_ALBU = True
except: HAS_ALBU = False

def grabcut_foreground(bgr_img):
    """GrabCut presisi untuk berbagai background. Fallback ke Otsu jika gagal."""
    h,w = bgr_img.shape[:2]
    # init rect dari Otsu bbox
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # coba kedua invert
    for cand_th in [cv2.bitwise_not(th), th]:
        # bersihkan
        kernel = np.ones((5,5), np.uint8)
        cand = cv2.morphologyEx(cand_th, cv2.MORPH_OPEN, kernel)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel)
        cnts,_ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 0.01*h*w: continue
        x,y,ww,hh = cv2.boundingRect(c)
        # GrabCut rect diperbesar 2px
        rect = (max(0,x-2), max(0,y-2), min(w-1, ww+4), min(h-1, hh+4))
        mask = np.zeros((h,w), np.uint8)
        bgd = np.zeros((1,65), np.float64)
        fgd = np.zeros((1,65), np.float64)
        try:
            cv2.grabCut(bgr_img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            mask_fg = np.where((mask==1)|(mask==3), 255, 0).astype(np.uint8)
            # bersihkan lagi
            mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel)
            mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel)
            if cv2.countNonZero(mask_fg) < 0.01*h*w: continue
            cnts2,_ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts2: continue
            c2 = max(cnts2, key=cv2.contourArea)
            if cv2.contourArea(c2) < 0.01*h*w: continue
            x2,y2,w2,h2 = cv2.boundingRect(c2)
            return mask_fg, (x2,y2,w2,h2,c2)
        except: continue
    # fallback Otsu generik lama
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower_sv = np.array([0,30,30],dtype=np.uint8)
    upper_sv = np.array([180,255,255],dtype=np.uint8)
    mask_sv = cv2.inRange(hsv, lower_sv, upper_sv)
    for cand in [cv2.bitwise_and(mask_sv,mask_sv,mask=cv2.bitwise_not(th)), cv2.bitwise_and(mask_sv,mask_sv,mask=th)]:
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel)
        if 0.01*h*w < cv2.countNonZero(cand) < 0.9*h*w:
            cnts,_ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                x,y,ww,hh = cv2.boundingRect(c)
                mask_fg = np.zeros_like(cand)
                cv2.drawContours(mask_fg,[c],-1,255,-1)
                return mask_fg, (x,y,ww,hh,c)
    return None, None

def compute_hist_features(bgr_img, mask_fg):
    """Hist HSV H 32b + S 32b + LAB + Hu 7 + circularity/aspect + GLCM."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # hist H 32 bins 0-180, S 32 bins 0-255, normalisasi
    hist_h = cv2.calcHist([hsv],[0],mask_fg,[32],[0,180])
    hist_h = cv2.normalize(hist_h, hist_h).flatten().tolist()
    hist_s = cv2.calcHist([hsv],[1],mask_fg,[32],[0,256])
    hist_s = cv2.normalize(hist_s, hist_s).flatten().tolist()
    # LAB hist untuk tahan lighting
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    hist_l = cv2.calcHist([lab],[0],mask_fg,[16],[0,256])
    hist_l = cv2.normalize(hist_l, hist_l).flatten().tolist()
    # Hu moments
    mom = cv2.moments(mask_fg)
    hu = cv2.HuMoments(mom).flatten()
    # log scale
    hu = (-np.sign(hu)*np.log10(np.abs(hu)+1e-10)).tolist()
    # glcm & shape dari kontur terbesar
    cnts,_ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea) if cnts else None
    circ, ar, kontr = 0,0,0
    bbox=None
    if c is not None:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c,True)
        circ = (4*math.pi*area)/(peri**2) if peri else 0
        x,y,w,h = cv2.boundingRect(c)
        ar = max(w/h, h/w) if h else 0
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        kontr,_ = hitung_tekstur_glcm(gray,x,y,w,h,mask_fg)
        bbox=(x,y,w,h)
    # hsv mean/std untuk hsv_ranges fallback
    hsv_pixels = hsv[mask_fg>0]
    mean_hsv = np.mean(hsv_pixels,axis=0).tolist() if len(hsv_pixels) else [0,0,0]
    return {"hist_h":hist_h,"hist_s":hist_s,"hist_l":hist_l,"hu":hu,"circ":float(circ),"ar":float(ar),"kontr":float(kontr),"mean_hsv":mean_hsv,"bbox":bbox}

def augment_image(bgr_img, n=4):
    if not HAS_ALBU: return []
    h,w = bgr_img.shape[:2]
    # jangan augment terlalu ekstrim untuk buah
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=25, p=0.8, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.7),
        A.Perspective(scale=(0.02,0.05), p=0.3),
    ])
    out=[]
    for _ in range(n):
        try:
            res = transform(image=bgr_img)
            out.append(res["image"])
        except: pass
    return out

def build_hsv_ranges(hsv_vals):
    arr=np.array(hsv_vals,dtype=float)
    mean=np.mean(arr,axis=0)
    if len(arr)>1: std=np.std(arr,axis=0)
    else: std=np.array([7.,30.,30.])
    std[0]=max(min(std[0],15.),5.)
    std[1]=max(min(std[1],30.),15.)
    std[2]=max(min(std[2],30.),15.)
    lo=mean-2*std; hi=mean+2*std
    lo=np.maximum(lo,[0,0,0]); hi=np.minimum(hi,[180,255,255])
    lo=lo.astype(int); hi=hi.astype(int)
    if lo[0]<=5 and hi[0]>=175:
        return [(np.array([0,int(lo[1]),int(lo[2])]), np.array([10,int(hi[1]),int(hi[2])])),
                (np.array([170,int(lo[1]),int(lo[2])]), np.array([180,int(hi[1]),int(hi[2])]))]
    return [(lo,hi)]

def main():
    parser=argparse.ArgumentParser(description="Kalibrasi GrabCut + augmentasi multi-template")
    parser.add_argument("--data",type=str,default="data_buah")
    parser.add_argument("--out-csv",type=str,default="kalibrasi.csv")
    parser.add_argument("--out-json",type=str,default="app/database.json")
    parser.add_argument("--reset",action="store_true")
    parser.add_argument("--aug",type=int,default=2, help="augment per foto (n=2 -> 6 template, default 2 agar cepat)")
    args=parser.parse_args()
    if args.reset:
        for p in [args.out_json,"database.json",args.out_csv]:
            try:
                if os.path.exists(p):
                    os.remove(p); print(f"[RESET] {p} dihapus")
            except: pass
    scan=scan_data_buah(args.data)
    if not scan:
        print(f"[ERROR] Tidak ada data di {args.data}/")
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json,"w") as f: json.dump({},f,indent=2)
        if args.out_json!="database.json":
            with open("database.json","w") as f: json.dump({},f,indent=2)
        return
    rows=[]
    templates=defaultdict(list)  # label -> list template dict
    hsv_raw=defaultdict(list)
    for label,paths in scan.items():
        n=len(paths)
        print(f"\n=== {label} : {n} gambar ===")
        if n<5: print(f"  [INFO] n={n} (<5 saran, tetap diproses + augment x{args.aug})")
        for p in sorted(paths):
            img=cv2.imread(p)
            if img is None: continue
            img=cv2.resize(img,(500,int(img.shape[0]*500/img.shape[1])))
            # original
            for aug_idx, bgr in enumerate([img]+augment_image(img, args.aug if n<5 else 2)):
                mask,info=grabcut_foreground(bgr)
                if mask is None: 
                    if aug_idx==0: print(f"  {os.path.basename(p):25} GAGAL grabcut")
                    continue
                feats=compute_hist_features(bgr,mask)
                # simpan template
                templates[label].append({"hist_h":feats["hist_h"],"hist_s":feats["hist_s"],"hist_l":feats["hist_l"],"hu":feats["hu"],"circ":feats["circ"],"ar":feats["ar"],"kontr":feats["kontr"],"mean_hsv":feats["mean_hsv"],"src":os.path.basename(p)+ (f"_aug{aug_idx}" if aug_idx else "")})
                hsv_raw[label].append(feats["mean_hsv"])
                if aug_idx==0:
                    # untuk CSV hanya original
                    print(f"  {os.path.basename(p):25} circ={feats['circ']:.3f} ar={feats['ar']:.2f} kontr={feats['kontr']:.0f} H={int(feats['mean_hsv'][0]):3} S={int(feats['mean_hsv'][1]):3} aug={len(templates[label])}")
                    rows.append([p,label,label,cv2.countNonZero(mask),feats["circ"],feats["ar"],feats["kontr"],feats["mean_hsv"][0],feats["mean_hsv"][1],feats["mean_hsv"][2],1])
    with open(args.out_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["file","gt","candidate","mask_area","circularity","aspect_ratio","contrast","h_mean","s_mean","v_mean","is_gt"])
        # rows sudah ringkas, tulis ulang dengan format lama
        for r in rows:
            w.writerow(r)
    print(f"\n[INFO] {args.out_csv} {len(rows)} baris original, total templates {sum(len(v) for v in templates.values())}")

    # Build HSV ranges fallback
    hsv_ranges_map={}
    for label,vals in hsv_raw.items():
        hsv_ranges_map[label]=build_hsv_ranges(vals)
        print(f"[HSV] {label}: {len(vals)} samples -> {[(lo.tolist(),hi.tolist()) for lo,hi in hsv_ranges_map[label]]}")

    # stats dari templates
    new_db={}
    for label in scan.keys():
        tpls=templates.get(label,[])
        if not tpls:
            print(f"[WARN] {label}: 0 template -> skip")
            continue
        circs=[t["circ"] for t in tpls]; ars=[t["ar"] for t in tpls]; cons=[t["kontr"] for t in tpls]
        mean_circ=float(np.mean(circs)); std_circ=float(np.std(circs) if len(circs)>1 else 0.15*mean_circ)
        mean_ar=float(np.mean(ars)); std_ar=float(np.std(ars) if len(ars)>1 else 0.15*mean_ar)
        mean_k=float(np.mean(cons)); std_k=float(np.std(cons) if len(cons)>1 else 0.2*mean_k)
        std_circ=max(std_circ,0.03); std_ar=max(std_ar,0.05); std_k=max(std_k,30)
        hsv_arr=np.array([t["mean_hsv"] for t in tpls],dtype=float)
        hsv_mean=np.mean(hsv_arr,axis=0).tolist(); hsv_std=np.std(hsv_arr,axis=0).tolist() if len(hsv_arr)>1 else [7,30,30]
        stats={"mean_circularity":mean_circ,"std_circularity":std_circ,"mean_aspect_ratio":mean_ar,"std_aspect_ratio":std_ar,"mean_contrast":mean_k,"std_contrast":std_k,"min_circularity":float(min(circs)),"max_circularity":float(max(circs)),"min_aspect_ratio":float(min(ars)),"max_aspect_ratio":float(max(ars)),"min_contrast":float(min(cons)),"max_contrast":float(max(cons)),"n":len(tpls),"hsv_mean":hsv_mean,"hsv_std":hsv_std}
        margin=0.25
        def mrange(vs): lo=min(vs); hi=max(vs); m=(hi-lo)*margin if hi!=lo else abs(hi)*margin; return lo-m, hi+m
        c_lo,c_hi=mrange(circs); a_lo,a_hi=mrange(ars); k_lo,k_hi=mrange(cons)
        rules={"min_circularity":round(c_lo,2),"max_circularity":round(min(1.0,c_hi),2),"min_aspect_ratio":round(a_lo,2),"max_aspect_ratio":round(a_hi,2),"min_contrast":int(k_lo),"max_contrast":int(k_hi)}
        new_db[label]={"hsv_ranges":hsv_ranges_map[label],"stats":stats,"rules":rules,"templates":tpls}

    out_json={}
    for label,v in new_db.items():
        lst=[]
        for lo,hi in v["hsv_ranges"]:
            lst.append([lo.tolist() if hasattr(lo,"tolist") else list(lo), hi.tolist() if hasattr(hi,"tolist") else list(hi)])
        out_json[label]={"hsv_ranges":lst,"stats":v["stats"],"rules":v["rules"],"templates":v["templates"]}
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json,"w") as f: json.dump(out_json,f,indent=2)
    print(f"\n[INFO] {args.out_json} {len(out_json)} kelas, {sum(len(v['templates']) for v in new_db.values())} templates")
    if args.out_json!="database.json":
        with open("database.json","w") as f: json.dump(out_json,f,indent=2)

if __name__=="__main__": main()
