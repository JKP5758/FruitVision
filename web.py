import os, base64, cv2, numpy as np, json, time, threading
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from app.detector import identifikasi_frame, get_database, scan_data_buah

app = Flask(__name__)
CORS(app)

# --- watchdog auto kalibrasi (aktif via env, efisien inotify 2s) ---
WATCHDOG_ENABLED = os.getenv("WATCHDOG_ENABLED", "true").lower() in ("1", "true", "yes", "on")
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

last_calibrate = 0
calibrate_lock = threading.Lock()
_watchdog_timer = None
_watchdog_timer_lock = threading.Lock()

def run_kalibrasi(reset=True):
    global last_calibrate
    with calibrate_lock:
        now = time.time()
        if now - last_calibrate < 3:  # debounce 3s
            return
        last_calibrate = now
        print(f"[AUTO] Kalibrasi ulang terpicu... (reset={reset})")
        try:
            import subprocess, sys
            # reset sebelum kalibrasi untuk hapus ghost class
            if reset:
                for p in ["app/database.json", "database.json", "kalibrasi.csv"]:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                            print(f"[RESET] {p} dihapus")
                    except: pass
            args = [sys.executable, "kalibrasi.py", "--data", "data_buah", "--aug", "2"]
            if reset:
                args.append("--reset")
            res = subprocess.run(args,
                                 capture_output=True, text=True, timeout=90)
            print(res.stdout[-1000:])
            if res.stderr:
                print("[AUTO ERR]", res.stderr[-500:])
        except Exception as e:
            print(f"[AUTO] Gagal kalibrasi: {e}")

if WATCHDOG_AVAILABLE and WATCHDOG_ENABLED:
    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            # tangani file & folder (hapus folder = is_directory true)
            is_dir = event.is_directory
            path = event.src_path
            if "data_buah" not in path:
                return
            if not is_dir and not path.lower().endswith((".png",".jpg",".jpeg",".bmp")):
                return
            # untuk folder, hanya event deleted/created/moved
            if is_dir and event.event_type not in ("deleted", "created", "moved"):
                return
            # debounce single timer: cancel timer lama, buat baru 2s
            global _watchdog_timer
            with _watchdog_timer_lock:
                if _watchdog_timer is not None:
                    try:
                        _watchdog_timer.cancel()
                    except: pass
                _watchdog_timer = threading.Timer(2.0, lambda: run_kalibrasi(reset=True))
                _watchdog_timer.daemon = True
                _watchdog_timer.start()

    try:
        observer = Observer()
        observer.schedule(Handler(), path="data_buah", recursive=True)
        observer.start()
        print("[WATCHDOG] Auto-kalibrasi aktif memantau data_buah/ (inotify 2s debounce, WATCHDOG_ENABLED=true)")
    except Exception as e:
        print(f"[WARN] Watchdog gagal: {e}")
        WATCHDOG_AVAILABLE = False
else:
    if not WATCHDOG_AVAILABLE:
        print("[WATCHDOG] watchdog lib tidak tersedia — pakai poller 10s")
    elif not WATCHDOG_ENABLED:
        print("[WATCHDOG] Nonaktif via WATCHDOG_ENABLED=false — pakai poller 10s")

# fallback polling tiap 10s jika watchdog tidak tersedia
if not (WATCHDOG_AVAILABLE and WATCHDOG_ENABLED):
    def poller():
        last_mtime = 0
        last_count = -1
        last_dir_count = -1
        while True:
            time.sleep(10)
            try:
                mtime = 0
                count = 0
                for root, _, files in os.walk("data_buah"):
                    for f in files:
                        count += 1
                        p = os.path.join(root, f)
                        try:
                            mtime = max(mtime, os.path.getmtime(p))
                        except: pass
                # hitung folder count juga untuk deteksi hapus folder
                dir_count = len([d for d in os.listdir("data_buah") if os.path.isdir(os.path.join("data_buah", d))]) if os.path.exists("data_buah") else 0
                if mtime > last_mtime or count != last_count or dir_count != last_dir_count:
                    last_mtime = mtime
                    last_count = count
                    last_dir_count = dir_count
                    run_kalibrasi(reset=True)
            except: pass
    threading.Thread(target=poller, daemon=True).start()
    print("[POLLER] Fallback polling 10s aktif")

@app.route("/")
def index():
    db = get_database()
    classes = list(db.keys())
    scan = scan_data_buah("data_buah")
    counts = {k: len(v) for k, v in scan.items()}
    return render_template("index.html", classes=classes, counts=counts)

@app.route("/api/classes")
def api_classes():
    db = get_database()
    scan = scan_data_buah("data_buah")
    # kirim stats jika ada, fallback rules
    db_out={}
    for k,v in db.items():
        db_out[k]= v.get("stats") or v.get("rules")
    return jsonify({
        "classes": list(db.keys()),
        "counts": {k: len(v) for k, v in scan.items()},
        "database": db_out
    })

@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    run_kalibrasi(reset=True)
    db=get_database()
    db_out={k: v.get("stats") or v.get("rules") for k,v in db.items()}
    return jsonify({"status": "ok", "database": db_out})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "no image"}), 400
        b64 = data["image"]
        # hapus prefix data:image/jpeg;base64,
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "decode failed"}), 400

        start = time.time()
        res = identifikasi_frame(frame, verbose=False)
        latency = (time.time() - start) * 1000

        bbox = res.get("bbox")
        bboxes = res.get("bboxes", [])
        # normalisasi bbox ke koordinat asli frame (karena detector resize ke 500)
        # detector sudah resize internal, tapi bbox relatif terhadap resized 500, kita kirim apa adanya
        # frontend akan scale sesuai video size
        return jsonify({
            "hasil": res["hasil"],
            "bbox": {"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]), "h": int(bbox[3])} if bbox else None,
            "bboxes": [{"x": int(b["x"]), "y": int(b["y"]), "w": int(b["w"]), "h": int(b["h"]), "label": b["label"], "prob": round(b["prob"]*100,1)} for b in bboxes],
            "fitur": res.get("fitur", {}),
            "objek_terdeteksi": res.get("objek_terdeteksi", False),
            "latency_ms": round(latency, 1),
            "semua_deteksi": [
                {"nama_buah": d["nama_buah"], "probabilitas": round(d.get("probabilitas",0)*100,1), "circularity": round(d["circularity"],3), "aspect_ratio": round(d["aspect_ratio"],2), "kontras": round(d["kontras"],1)}
                for d in res.get("semua_deteksi", [])
            ]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "decode failed"}), 400
    start = time.time()
    res = identifikasi_frame(frame, verbose=False)
    latency = (time.time() - start) * 1000
    bboxes = res.get("bboxes", [])
    # untuk upload kita juga simpan bbox image sebagai base64 untuk ditampilkan
    # encode bbox image
    bbox_b64 = None
    if bboxes and res.get("img") is not None:
        img_kotak = res["img"].copy()
        for b in bboxes:
            x,y,w,h = b["x"], b["y"], b["w"], b["h"]
            cv2.rectangle(img_kotak, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img_kotak, f"{b['label']} {b['prob']*100:.0f}%", (x, max(15, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        _, buf = cv2.imencode(".jpg", img_kotak)
        bbox_b64 = base64.b64encode(buf).decode()
    elif res.get("bbox") and res.get("img") is not None:
        img_kotak = res["img"].copy()
        x,y,w,h = res["bbox"]
        cv2.rectangle(img_kotak, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img_kotak, res["hasil"], (x, max(15, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        _, buf = cv2.imencode(".jpg", img_kotak)
        bbox_b64 = base64.b64encode(buf).decode()
    return jsonify({
        "hasil": res["hasil"],
        "bbox": {"x": int(res["bbox"][0]), "y": int(res["bbox"][1]), "w": int(res["bbox"][2]), "h": int(res["bbox"][3])} if res.get("bbox") else None,
        "bboxes": [{"x": int(b["x"]), "y": int(b["y"]), "w": int(b["w"]), "h": int(b["h"]), "label": b["label"], "prob": round(b["prob"]*100,1)} for b in bboxes],
        "bbox_image": bbox_b64,
        "fitur": res.get("fitur", {}),
        "probabilitas": res.get("semua_deteksi", [{}])[0].get("probabilitas",0) if res.get("semua_deteksi") else 0,
        "semua_deteksi": [{"nama_buah": d["nama_buah"], "probabilitas": round(d.get("probabilitas",0)*100,1)} for d in res.get("semua_deteksi", [])],
        "latency_ms": round(latency,1),
    })

if __name__ == "__main__":
    # ensure data_buah ada (dinamis, tidak hard-code)
    os.makedirs("data_buah", exist_ok=True)
    # warning token dummy
    token = os.getenv("TUNNEL_TOKEN", "")
    if token.startswith("dummy-") or token == "":
        print("[WARN] TUNNEL_TOKEN dummy/kosong — Cloudflare Tunnel tidak akan connect (isi .env dengan token asli)")
    # jika database sudah ada dan valid (multi-template), jangan reset otomatis (hemat) - prioritaskan app/database.json
    has_templates = False
    for _p in ["app/database.json", "database.json"]:
        if os.path.exists(_p) and os.path.getsize(_p) > 1000:
            try:
                import json as _js
                with open(_p,"r") as f: _d=_js.load(f)
                if any("templates" in v for v in _d.values()):
                    has_templates = True
                    print(f"[STARTUP] Database ditemukan di {_p} ({len(_d)} kelas)")
                    break
            except: pass
    if has_templates:
        print("[STARTUP] Database multi-template sudah ada, skip kalibrasi awal")
    else:
        print("[STARTUP] Reset & kalibrasi awal...")
        run_kalibrasi(reset=True)
    port = int(os.environ.get("PORT", 5000))
    print(f"[WEB] http://localhost:{port} — classes: {list(get_database().keys())} — watchdog={'ON' if (WATCHDOG_AVAILABLE and WATCHDOG_ENABLED) else 'OFF (poller 10s)'}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
