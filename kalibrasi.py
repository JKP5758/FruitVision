"""Kalibrasi data-driven: dump fitur mentah dan usulkan rules baru."""
import cv2, numpy as np, math, csv, os
from skimage.feature import graycomatrix, graycoprops
from app import DATABASE_BUAH, LABEL_GROUND_TRUTH, buat_masker_warna

def glcm_unmasked(gray, x,y,w,h):
    roi = gray[y:y+h, x:x+w]
    if roi.size==0:
        return 0,0
    # downscale jika besar
    if max(roi.shape)>150:
        scale=150/max(roi.shape)
        roi=cv2.resize(roi,(int(roi.shape[1]*scale),int(roi.shape[0]*scale)))
    g=graycomatrix(roi, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    return graycoprops(g,'contrast')[0,0], graycoprops(g,'homogeneity')[0,0]

rows=[]
for fname, gt_label in LABEL_GROUND_TRUTH.items():
    if not os.path.exists(fname):
        continue
    img=cv2.imread(fname)
    img=cv2.resize(img,(500,int(img.shape[0]*500/img.shape[1])))
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(f"\n=== {fname} GT={gt_label} ===")
    for nama, data in DATABASE_BUAH.items():
        mask=buat_masker_warna(hsv,data["hsv_ranges"])
        area=cv2.countNonZero(mask)
        if area<500:
            continue
        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c=max(cnts,key=cv2.contourArea)
        area_c=cv2.contourArea(c)
        peri=cv2.arcLength(c,True)
        x,y,w,h=cv2.boundingRect(c)
        circ=(4*math.pi*area_c)/(peri**2) if peri>0 else 0
        ar=max(w/h,h/w) if h else 0
        kontras, homo=glcm_unmasked(gray,x,y,w,h)
        is_gt = (nama==gt_label)
        print(f"  {nama:22} area={area:6} ac={area_c:6.0f} circ={circ:.3f} ar={ar:.2f} kontras={kontras:.0f} homo={homo:.3f} {'<<GT' if is_gt else ''}")
        rows.append([fname,gt_label,nama,area,area_c,circ,ar,kontras,homo,int(is_gt)])

# tulis csv
with open("kalibrasi.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["file","gt","candidate","mask_area","contour_area","circularity","aspect_ratio","contrast","homogeneity","is_gt"])
    w.writerows(rows)
print("\n[INFO] kalibrasi.csv tertulis",len(rows),"baris")

# analisis statistik per GT (hanya baris is_gt)
from collections import defaultdict
import numpy as np
stats=defaultdict(list)
for r in rows:
    if r[9]==1:  # is_gt (index 9)
        stats[r[1]].append(r)

print("\n=== STATISTIK GT (unmasked contrast) ===")
for gt, lst in stats.items():
    circs=[x[5] for x in lst]
    ars=[x[6] for x in lst]
    cons=[x[7] for x in lst]
    print(f"{gt}:")
    print(f"  circ {np.mean(circs):.3f} range [{min(circs):.3f},{max(circs):.3f}]")
    print(f"  ar   {np.mean(ars):.3f} range [{min(ars):.3f},{max(ars):.3f}]")
    print(f"  kontr {np.mean(cons):.0f} range [{min(cons):.0f},{max(cons):.0f}]")

# usulan rules (rentang + margin 15%)
print("\n=== USULAN RULES (margin 20%) ===")
for gt, lst in stats.items():
    circs=[x[5] for x in lst]
    ars=[x[6] for x in lst]
    cons=[x[7] for x in lst]
    # margin 20% atau std
    def margin_range(vals, pct=0.2, decimals=3):
        lo=min(vals); hi=max(vals)
        margin=(hi-lo)*pct if hi!=lo else abs(hi)*pct
        if margin==0:
            margin=0.05
        return max(0,lo-margin), hi+margin
    c_lo,c_hi=margin_range(circs,0.25)
    a_lo,a_hi=margin_range(ars,0.25)
    k_lo,k_hi=margin_range(cons,0.25)
    # bulatkan
    print(f'"{gt}": min_circularity {c_lo:.2f}, max {c_hi:.2f} | ar {a_lo:.2f}-{a_hi:.2f} | kontr {k_lo:.0f}-{k_hi:.0f}')
