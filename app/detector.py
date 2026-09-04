import cv2
import numpy as np
import math
import os
import json

DEFAULT_DATABASE = {}
LABEL_MAP = {}

# === Konstanta global (disepakati 15% threshold) ===
DETECTION_THRESHOLD = 0.15  # prob < 0.15 -> Tidak ada buah terdeteksi (sinkron README)
RESIZE_WIDTH = 500
MIN_AREA_RATIO = 0.005
IOU_NMS_THRESHOLD = 0.5
MERGE_AR_RANGE = (1.8, 3.5)
HIST_WEIGHTS = (0.5, 0.3, 0.2)  # H, S, L
HU_WEIGHT = 0.05
SHAPE_WEIGHT = 0.08

def _dist_to_prob(dist, scale=1.0):
    """Konversi chi2/Hu distance -> probabilitas 0-1 konsisten."""
    return max(0.0, min(1.0, math.exp(-dist * scale)))

def _load_database():
    # Single source of truth: app/database.json (sinkron kalibrasi.py)
    candidates = ["app/database.json", "database.json", "data_buah/database.json"]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                if not data:
                    return {}
                db = {}
                for label, v in data.items():
                    ranges = []
                    for lo, hi in v.get("hsv_ranges", []):
                        ranges.append((np.array(lo, dtype=np.int32), np.array(hi, dtype=np.int32)))
                    entry = {"hsv_ranges": ranges}
                    if "stats" in v:
                        entry["stats"] = v["stats"]
                    if "rules" in v:
                        entry["rules"] = v["rules"]
                    if "templates" in v:
                        entry["templates"] = v["templates"]
                    if "stats" not in entry and "rules" in entry:
                        r=entry["rules"]
                        entry["stats"]={
                            "mean_circularity": (r["min_circularity"]+r["max_circularity"])/2,
                            "std_circularity": max(0.03, (r["max_circularity"]-r["min_circularity"])/4),
                            "mean_aspect_ratio": (r["min_aspect_ratio"]+r["max_aspect_ratio"])/2,
                            "std_aspect_ratio": max(0.05, (r["max_aspect_ratio"]-r["min_aspect_ratio"])/4),
                            "mean_contrast": (r["min_contrast"]+r["max_contrast"])/2,
                            "std_contrast": max(30, (r["max_contrast"]-r["min_contrast"])/4),
                            "min_circularity": r["min_circularity"],
                            "max_circularity": r["max_circularity"],
                            "min_aspect_ratio": r["min_aspect_ratio"],
                            "max_aspect_ratio": r["max_aspect_ratio"],
                            "min_contrast": r["min_contrast"],
                            "max_contrast": r["max_contrast"],
                            "n": r.get("n", 1)
                        }
                    db[label] = entry
                if db:
                    return db
                return {}
            except Exception as e:
                print(f"[WARN] Gagal load {p}: {e}")
    return {}

def get_database():
    return _load_database()

def buat_masker_warna(hsv_img, hsv_ranges):
    mask_gabungan = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for batas_bawah, batas_atas in hsv_ranges:
        mask = cv2.inRange(hsv_img, batas_bawah, batas_atas)
        mask_gabungan = cv2.bitwise_or(mask_gabungan, mask)
    kernel = np.ones((5, 5), np.uint8)
    mask_gabungan = cv2.morphologyEx(mask_gabungan, cv2.MORPH_OPEN, kernel)
    mask_gabungan = cv2.morphologyEx(mask_gabungan, cv2.MORPH_CLOSE, kernel)
    return mask_gabungan

def hitung_tekstur_glcm(gray_img, x, y, w, h, mask=None):
    roi_gray = gray_img[y:y+h, x:x+w]
    if roi_gray.size == 0:
        return 0, 0
    _ = mask
    max_side = 150
    if max(roi_gray.shape) > max_side:
        scale = max_side / max(roi_gray.shape)
        new_w = max(1, int(roi_gray.shape[1] * scale))
        new_h = max(1, int(roi_gray.shape[0] * scale))
        roi_gray = cv2.resize(roi_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    try:
        from skimage.feature import graycomatrix, graycoprops
        glcm = graycomatrix(roi_gray, distances=[5], angles=[0,1,2,3], levels=256, symmetric=True, normed=True)
        kontras = np.mean([graycoprops(glcm, 'contrast')[0, i] for i in range(4)])
        homogenitas = np.mean([graycoprops(glcm, 'homogeneity')[0, i] for i in range(4)])
    except Exception:
        kontras, homogenitas = 0, 0
    return kontras, homogenitas

def ekstrak_fitur_kontur(contour, gray, mask, x, y, w, h):
    area = cv2.contourArea(contour)
    keliling = cv2.arcLength(contour, True)
    if keliling == 0:
        return None
    aspek_rasio = max(float(w)/h, float(h)/w) if h != 0 else 0
    circularity = (4 * math.pi * area) / (keliling ** 2)
    kontras, homogenitas = hitung_tekstur_glcm(gray, x, y, w, h, mask=mask)
    return {"area": area, "keliling": keliling, "aspect_ratio": aspek_rasio, "circularity": circularity, "kontras": kontras, "homogenitas": homogenitas, "x": x, "y": y, "w": w, "h": h}

def _chi2(h1, h2):
    h1=np.array(h1,dtype=float); h2=np.array(h2,dtype=float)
    denom=h1+h2+1e-10
    return 0.5*np.sum((h1-h2)**2/denom)

def _hu_distance(hu1, hu2):
    return np.linalg.norm(np.array(hu1)-np.array(hu2))

def _grabcut_multi(bgr_img):
    """Return list of (mask, bbox, contour) untuk multi-buah via GrabCut + fallback HSV."""
    h,w=bgr_img.shape[:2]
    # Coba GrabCut
    gray=cv2.cvtColor(bgr_img,cv2.COLOR_BGR2GRAY)
    _, th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    masks=[]
    for cand_th in [cv2.bitwise_not(th), th]:
        kernel=np.ones((5,5),np.uint8)
        cand=cv2.morphologyEx(cand_th,cv2.MORPH_OPEN,kernel)
        cand=cv2.morphologyEx(cand,cv2.MORPH_CLOSE,kernel)
        cnts,_=cv2.findContours(cand,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        # ambil beberapa kontur besar, bukan hanya max, untuk multi
        cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
        for c in cnts:
            if cv2.contourArea(c) < MIN_AREA_RATIO*h*w: continue
            x,y,ww,hh=cv2.boundingRect(c)
            rect=(max(0,x-2),max(0,y-2),min(w,ww+4),min(h,hh+4))
            mask=np.zeros((h,w),np.uint8)
            bgd=np.zeros((1,65),np.float64); fgd=np.zeros((1,65),np.float64)
            try:
                cv2.grabCut(bgr_img,mask,rect,bgd,fgd,3,cv2.GC_INIT_WITH_RECT)
                mask_fg=np.where((mask==1)|(mask==3),255,0).astype(np.uint8)
                mask_fg=cv2.morphologyEx(mask_fg,cv2.MORPH_OPEN,kernel)
                mask_fg=cv2.morphologyEx(mask_fg,cv2.MORPH_CLOSE,kernel)
                # pisahkan kontur lagi untuk multi
                cnts2,_=cv2.findContours(mask_fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for c2 in cnts2:
                    if cv2.contourArea(c2) < MIN_AREA_RATIO*h*w: continue
                    x2,y2,w2,h2=cv2.boundingRect(c2)
                    m=np.zeros_like(mask_fg)
                    cv2.drawContours(m,[c2],-1,255,-1)
                    masks.append((m,(x2,y2,w2,h2),c2))
            except: pass
        if masks: break
    if masks:
        # NMS sederhana hapus duplikat IoU tinggi
        masks=sorted(masks,key=lambda x: cv2.contourArea(x[2]),reverse=True)
        filtered=[]
        for m,b,c in masks:
            dup=False
            for _,b2,_ in filtered:
                # IoU
                x1,y1,w1,h1=b; x2,y2,w2,h2=b2
                xi1=max(x1,x2); yi1=max(y1,y2); xi2=min(x1+w1,x2+w2); yi2=min(y1+h1,y2+h2)
                inter=max(0,xi2-xi1)*max(0,yi2-yi1)
                union=w1*h1+w2*h2-inter
                if union and inter/union>IOU_NMS_THRESHOLD:
                    dup=True; break
            if not dup: filtered.append((m,b,c))
        filtered=filtered[:5]
        # Merge warna: coba gabung semua bagian jika hasil gabungan lebih mirip template daripada bagian terpisah
        # Ini memperbaiki terong lanskap yang terpecah tangkai hijau + ungu + highlight (pir/apel) vs hstack multi buah yang tetap terpisah
        if len(filtered) > 1 and len(filtered) <= 4:
            merged=np.zeros_like(filtered[0][0])
            for m,_,_ in filtered:
                merged=cv2.bitwise_or(merged,m)
            kernel7=np.ones((7,7),np.uint8)
            merged=cv2.morphologyEx(merged,cv2.MORPH_CLOSE,kernel7)
            # cek merge vs individual via hist (tanpa database, pakai heuristic: merged lebih kompak dan tidak terlalu besar)
            xs=[b[0] for _,b,_ in filtered]; ys=[b[1] for _,b,_ in filtered]
            xe=[b[0]+b[2] for _,b,_ in filtered]; ye=[b[1]+b[3] for _,b,_ in filtered]
            ex,ey,ew,eh = min(xs), min(ys), max(xe)-min(xs), max(ye)-min(ys)
            enclose_area = ew*eh
            sum_areas = sum(cv2.contourArea(c) for _,_,c in filtered)
            total_area = h*w
            # Untuk terong lanskap, merged prob biasanya > individu (karena hist rata-rata match template)
            # Untuk hstack 2 buah terpisah jauh, merged akan sangat besar dan mencakup background di tengah -> hist tidak match
            # Heuristik: jika enclose tidak terlalu besar (<0.7 total) dan sum/enclose >0.35 (cukup padat) -> coba merge
            if enclose_area/total_area < 0.75 and sum_areas/enclose_area > 0.15:
                cnts,_=cv2.findContours(merged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c=max(cnts,key=cv2.contourArea)
                    if cv2.contourArea(c) > 0.01*h*w:
                        # bandingkan: hitung chi2 merged vs template Terong (jika ada) vs max individu
                        # Sederhana: jika merged area mendekati sum areas (tidak banyak background), merge
                        x,y,w2,h2=cv2.boundingRect(c)
                        m2=np.zeros_like(merged)
                        cv2.drawContours(m2,[c],-1,255,-1)
                        # Jika merged aspect ratio lebih masuk akal untuk terong lanskap (~2.5) daripada individu terpecah
                        ar_merge = max(w2/h2, h2/w2) if h2 else 0
                        # terong lanskap ar 2.5, pir 1.0, apel 1.1 -> merge ar dalam MERGE_AR_RANGE lebih Terong
                        if MERGE_AR_RANGE[0] < ar_merge < MERGE_AR_RANGE[1]:
                            return [(m2,(x,y,w2,h2),c)]
        return filtered
    # fallback HSV union semua kelas
    return []

def _compute_query_features(bgr_img, mask_fg):
    hsv=cv2.cvtColor(bgr_img,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(bgr_img,cv2.COLOR_BGR2LAB)
    hist_h=cv2.calcHist([hsv],[0],mask_fg,[32],[0,180]); hist_h=cv2.normalize(hist_h,hist_h).flatten()
    hist_s=cv2.calcHist([hsv],[1],mask_fg,[32],[0,256]); hist_s=cv2.normalize(hist_s,hist_s).flatten()
    hist_l=cv2.calcHist([lab],[0],mask_fg,[16],[0,256]); hist_l=cv2.normalize(hist_l,hist_l).flatten()
    mom=cv2.moments(mask_fg)
    hu=cv2.HuMoments(mom).flatten()
    hu=(-np.sign(hu)*np.log10(np.abs(hu)+1e-10))
    cnts,_=cv2.findContours(mask_fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c=max(cnts,key=cv2.contourArea) if cnts else None
    circ,ar,kontr=0,0,0
    bbox=None
    if c is not None:
        area=cv2.contourArea(c); peri=cv2.arcLength(c,True)
        circ=(4*math.pi*area)/(peri**2) if peri else 0
        x,y,w,h=cv2.boundingRect(c); ar=max(w/h,h/w) if h else 0
        gray=cv2.cvtColor(bgr_img,cv2.COLOR_BGR2GRAY)
        kontr,_=hitung_tekstur_glcm(gray,x,y,w,h,mask_fg)
        bbox=(x,y,w,h)
    return {"hist_h":hist_h,"hist_s":hist_s,"hist_l":hist_l,"hu":hu,"circ":circ,"ar":ar,"kontr":kontr,"bbox":bbox,"mask":mask_fg}

def _hitung_probabilitas(fitur, stats):
    def gauss_sim(x, mean, std):
        if std <= 1e-6: std = max(0.03, abs(mean)*0.15)
        z = abs(x - mean) / std
        s = math.exp(-0.5 * z * z)
        return s, z
    s_circ, zc = gauss_sim(fitur["circularity"], stats["mean_circularity"], stats["std_circularity"])
    s_ar, za = gauss_sim(fitur["aspect_ratio"], stats["mean_aspect_ratio"], stats["std_aspect_ratio"])
    s_kon, zk = gauss_sim(fitur["kontras"], stats["mean_contrast"], stats["std_contrast"])
    prob = (s_circ + s_ar + s_kon) / 3.0
    detail = {"s_circ": s_circ, "s_ar": s_ar, "s_kon": s_kon, "z_circ": zc, "z_ar": za, "z_kon": zk}
    return prob, detail

def _identifikasi_core(img_bgr, database=None, verbose=True):
    if database is None:
        database = get_database()
    if not database:
        h0,w0=img_bgr.shape[:2]
        rasio=500/float(w0) if w0 else 1
        img=cv2.resize(img_bgr,(500,int(h0*rasio))) if w0 else img_bgr
        return {"img":img,"hsv":cv2.cvtColor(img,cv2.COLOR_BGR2HSV),"gray":cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),"hasil":"Belum ada database - lakukan kalibrasi (isi data_buah/ lalu Kalibrasi Ulang)","fitur":{},"objek_terdeteksi":False,"semua_deteksi":[],"mask_terbaik":None,"bbox":None,"scale_ratio":rasio,"bboxes":[]}

    lebar_target=RESIZE_WIDTH
    h0,w0=img_bgr.shape[:2]
    rasio=lebar_target/float(w0)
    tinggi_target=int(h0*rasio)
    img=cv2.resize(img_bgr,(lebar_target,tinggi_target))
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    total_piksel=img.shape[0]*img.shape[1]
    min_area_threshold=MIN_AREA_RATIO*total_piksel

    # Multi-mask via GrabCut
    multi_masks=_grabcut_multi(img)
    # Fallback jika tidak ada mask: pakai HSV per kelas (single)
    semua_deteksi=[]
    bboxes=[]
    objek_terdeteksi=False

    if multi_masks:
        for mask_fg,bbox,cnt in multi_masks:
            if cv2.countNonZero(mask_fg) < min_area_threshold: continue
            qfeat=_compute_query_features(img, mask_fg)
            # bandingkan ke semua templates per kelas
            per_class=[]
            for nama_buah,data in database.items():
                templates=data.get("templates")
                best_d=float('inf'); best_detail=None
                if templates:
                    for tpl in templates:
                        wH, wS, wL = HIST_WEIGHTS
                        d_h=_chi2(qfeat["hist_h"], tpl["hist_h"])*wH + _chi2(qfeat["hist_s"], tpl["hist_s"])*wS + _chi2(qfeat["hist_l"], tpl["hist_l"])*wL
                        d_hu=_hu_distance(qfeat["hu"], tpl["hu"])*HU_WEIGHT
                        # shape gauss
                        d_circ=abs(qfeat["circ"]-tpl.get("circ",0.5))/0.25
                        d_ar=abs(qfeat["ar"]-tpl.get("ar",1.5))/0.6
                        d=d_h + d_hu + (d_circ+d_ar)*SHAPE_WEIGHT
                        if d < best_d:
                            best_d=d
                            best_detail={"d_h":d_h,"d_hu":d_hu}
                else:
                    # fallback stats gauss
                    stats=data.get("stats")
                    if stats:
                        fitur={"circularity":qfeat["circ"],"aspect_ratio":qfeat["ar"],"kontras":qfeat["kontr"]}
                        prob,_=_hitung_probabilitas(fitur,stats)
                        best_d= -math.log(max(prob,1e-6))
                        best_detail={}
                    else: continue
                # convert distance ke prob 0-1 via helper konsisten
                prob = _dist_to_prob(best_d)
                per_class.append((nama_buah, prob, best_d, best_detail, qfeat["circ"], qfeat["ar"], qfeat["kontr"], bbox, mask_fg))
            if not per_class: continue
            per_class.sort(key=lambda x: x[1], reverse=True)
            top_name, top_prob, _, _, circ, ar, kontr, bbox, mask_fg = per_class[0]
            objek_terdeteksi=True
            # simpan semua kelas untuk objek ini
            for nama,prob,d,det,cc,aa,kk,_,_ in per_class:
                semua_deteksi.append({"nama_buah":nama,"probabilitas":float(prob),"circularity":float(cc),"aspect_ratio":float(aa),"kontras":float(kk),"bbox":bbox,"mask":mask_fg,"_obj":len(bboxes)})
            # bboxes untuk web multi
            bboxes.append({"x":int(bbox[0]),"y":int(bbox[1]),"w":int(bbox[2]),"h":int(bbox[3]),"label":top_name,"prob":float(top_prob),"circ":float(circ),"ar":float(ar),"kontr":float(kontr)})
        # sort semua_deteksi global
        semua_deteksi.sort(key=lambda d: d["probabilitas"], reverse=True)
        if bboxes:
            # hasil utama = objek dengan prob tertinggi
            best_box=max(bboxes, key=lambda b: b["prob"])
            if best_box["prob"] < DETECTION_THRESHOLD:
                hasil_identifikasi=f"Tidak ada buah terdeteksi (tertinggi {best_box['label']} {best_box['prob']*100:.1f}%)"
                mask_terbaik=None; bbox_terbaik=None; data_fitur_terbaik={}
            else:
                hasil_identifikasi=f"{best_box['label']} ({best_box['prob']*100:.1f}%)"
                # cari mask terbaik
                mask_terbaik=None
                for m,b,c in multi_masks:
                    if b== (best_box["x"],best_box["y"],best_box["w"],best_box["h"]):
                        mask_terbaik=m; break
                bbox_terbaik=(best_box["x"],best_box["y"],best_box["w"],best_box["h"])
                data_fitur_terbaik={"Area (px)":float(best_box["w"]*best_box["h"]) ,"Circularity":float(best_box["circ"]),"Aspect Ratio":float(best_box["ar"]),"Kontras":float(best_box["kontr"]),"Probabilitas":float(best_box["prob"])}
            return {"img":img,"hsv":hsv,"gray":gray,"hasil":hasil_identifikasi,"fitur":data_fitur_terbaik,"objek_terdeteksi":objek_terdeteksi,"semua_deteksi":semua_deteksi,"mask_terbaik":mask_terbaik,"bbox":bbox_terbaik,"scale_ratio":rasio,"bboxes":bboxes}
    # Fallback single HSV per kelas (jika GrabCut gagal)
    mask_terbaik=None; data_fitur_terbaik={}; bbox_terbaik=None; semua_deteksi=[]
    objek_terdeteksi=False
    for nama_buah,data in database.items():
        mask=buat_masker_warna(hsv,data["hsv_ranges"])
        area_piksel=cv2.countNonZero(mask)
        if area_piksel > min_area_threshold:
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                kontur_terbesar=max(contours,key=cv2.contourArea)
                area=cv2.contourArea(kontur_terbesar); keliling=cv2.arcLength(kontur_terbesar,True)
                if keliling>0:
                    objek_terdeteksi=True
                    x,y,w,h=cv2.boundingRect(kontur_terbesar)
                    fitur=ekstrak_fitur_kontur(kontur_terbesar,gray,mask,x,y,w,h)
                    if fitur is None: continue
                    # jika ada templates, pakai hist matching juga
                    templates=data.get("templates")
                    if templates:
                        # hitung query hist untuk mask ini
                        qfeat=_compute_query_features(img,mask)
                        best_d=float('inf')
                        wH, wS, wL = HIST_WEIGHTS
                        for tpl in templates:
                            d=_chi2(qfeat["hist_h"],tpl["hist_h"])*wH+_chi2(qfeat["hist_s"],tpl["hist_s"])*wS+_chi2(qfeat["hist_l"],tpl["hist_l"])*wL
                            if d<best_d: best_d=d
                        prob=_dist_to_prob(best_d)
                    else:
                        stats=data.get("stats")
                        if stats: prob,detail=_hitung_probabilitas(fitur,stats)
                        else: prob=1.0; detail={}
                        best_d= -math.log(max(prob,1e-6))
                    semua_deteksi.append({"nama_buah":nama_buah,"circularity":fitur["circularity"],"aspect_ratio":fitur["aspect_ratio"],"kontras":fitur["kontras"],"homogenitas":fitur["homogenitas"],"area":area,"area_piksel":area_piksel,"bbox":(x,y,w,h),"probabilitas":float(prob),"detail":{}, "mask":mask})
    semua_deteksi.sort(key=lambda d: d["probabilitas"], reverse=True)
    bboxes=[]
    if semua_deteksi:
        # kelompokkan per bbox untuk multi (fallback tetap single)
        for d in semua_deteksi[:3]:
            bboxes.append({"x":int(d["bbox"][0]),"y":int(d["bbox"][1]),"w":int(d["bbox"][2]),"h":int(d["bbox"][3]),"label":d["nama_buah"],"prob":float(d["probabilitas"])})
        top=semua_deteksi[0]; prob=top["probabilitas"]
        if prob<DETECTION_THRESHOLD:
            hasil_identifikasi=f"Tidak ada buah terdeteksi (tertinggi {top['nama_buah']} {prob*100:.1f}%)"
            mask_terbaik=None; bbox_terbaik=None; data_fitur_terbaik={}
        else:
            hasil_identifikasi=f"{top['nama_buah']} ({prob*100:.1f}%)"
            mask_terbaik=top["mask"]; bbox_terbaik=top["bbox"]
            data_fitur_terbaik={"Area (px)":float(top["area"]),"Area Mask (px)":int(top["area_piksel"]),"Circularity":float(top["circularity"]),"Aspect Ratio":float(top["aspect_ratio"]),"Kontras":float(top["kontras"]),"Homogenitas":float(top["homogenitas"]),"Probabilitas":float(prob)}
    else:
        hasil_identifikasi="Tidak ada buah terdeteksi (tidak ada warna cocok)"
    return {"img":img,"hsv":hsv,"gray":gray,"hasil":hasil_identifikasi,"fitur":data_fitur_terbaik,"objek_terdeteksi":objek_terdeteksi,"semua_deteksi":semua_deteksi,"mask_terbaik":mask_terbaik,"bbox":bbox_terbaik,"scale_ratio":rasio,"bboxes":bboxes}

def identifikasi_frame(frame_bgr, verbose=False, database=None):
    return _identifikasi_core(frame_bgr, database=database, verbose=verbose)

def identifikasi_buah(image_path, verbose=True, headless=False, save_dir="output", database=None):
    img=cv2.imread(image_path)
    if img is None:
        return {"image": image_path, "hasil": "Tidak ada buah terdeteksi (file tidak ditemukan)", "error": "file not found"}
    result=_identifikasi_core(img, database=database, verbose=False)
    if verbose:
        print("\n"+"="*60)
        print(f"MULAI PROSES SCANNING (multi-template): {image_path}")
        print("="*60)
        for d in result["semua_deteksi"][:6]:
            print(f"\n[?] {d['nama_buah']}: prob={d['probabilitas']*100:.1f}% | circ={d['circularity']:.3f} ar={d['aspect_ratio']:.2f} kontr={d['kontras']:.0f}")
        print("\n"+"="*60)
        print(f"KESIMPULAN AKHIR: {result['hasil']}")
        if result.get("bboxes"):
            print(f"Multi bboxes: {result['bboxes']}")
        print("="*60)
        if result["fitur"]:
            for k,v in result["fitur"].items():
                print(f" - {k}: {v:.4f}" if isinstance(v,float) else f" - {k}: {v}")
        print("="*60+"\n")
    if save_dir:
        os.makedirs(save_dir,exist_ok=True)
        import os.path as op
        base=op.splitext(op.basename(image_path))[0]
        cv2.imwrite(os.path.join(save_dir,f"{base}_resized.png"),result["img"])
        if result["mask_terbaik"] is not None:
            safe_label=result["hasil"].split(" (")[0].replace(" ","_").replace("/","-")
            cv2.imwrite(os.path.join(save_dir,f"{base}_mask_{safe_label}.png"),result["mask_terbaik"])
            img_kotak=result["img"].copy()
            # gambar semua bboxes
            for b in result.get("bboxes",[]):
                x,y,w,h=b["x"],b["y"],b["w"],b["h"]
                cv2.rectangle(img_kotak,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img_kotak,f"{b['label']} {b['prob']*100:.0f}%",(x,max(15,y-10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.imwrite(os.path.join(save_dir,f"{base}_bbox.png"),img_kotak)
    if not headless and verbose:
        try:
            cv2.imshow("1. Gambar Asli",result["img"])
            if result["mask_terbaik"] is not None:
                cv2.imshow(f"2. Masker: {result['hasil']}",result["mask_terbaik"])
                if result["bbox"]:
                    img_kotak=result["img"].copy()
                    x,y,w,h=result["bbox"]
                    cv2.rectangle(img_kotak,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.imshow("3. Bounding Box",img_kotak)
            cv2.waitKey(0); cv2.destroyAllWindows()
        except Exception as e:
            print(f"[WARN] imshow gagal: {e}")
    return {"image": image_path, "hasil": result["hasil"], "fitur": result["fitur"], "objek_terdeteksi": result["objek_terdeteksi"], "semua_deteksi": result["semua_deteksi"], "bbox": result["bbox"], "bboxes": result.get("bboxes",[])}

def scan_data_buah(root="data_buah"):
    out={}
    if not os.path.exists(root): return out
    for entry in os.listdir(root):
        folder=os.path.join(root,entry)
        if not os.path.isdir(folder): continue
        if entry.startswith("."): continue
        label=entry.strip().capitalize()
        files=[]
        for f in os.listdir(folder):
            if f.lower().endswith((".png",".jpg",".jpeg",".bmp")):
                files.append(os.path.join(folder,f))
        if files: out[label]=sorted(files)
    return out
