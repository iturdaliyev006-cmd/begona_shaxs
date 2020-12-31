import asyncio
import base64
import json
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ---------------- SOZLAMALAR ----------------
DROIDCAM_URL = "http://192.168.137.97:4747/video"
FACES_DIR = "xodimlar"        # xodimlar/Ali/1.jpg, xodimlar/Ali/2.jpg ...
MODEL = "Facenet512"          # tez va aniq model
DETECTOR = "opencv"           # aniqroq kerak bo'lsa: "retinaface" yoki "ssd"
THRESHOLD = 0.30              # Facenet512 + cosine uchun. Kichik = qattiqroq
CHECK_EVERY = 0.7             # soniya: yuzni qancha tez-tez tekshirish
ALERT_AFTER = 3               # ketma-ket nechta tekshiruvda begona chiqsa signal beriladi
ALERT_COOLDOWN = 30           # soniya: signal qayta-qayta yuborilmasligi uchun
SEND_WIDTH = 1280             # brauzerga yuboriladigan kadr eng katta kengligi (kattalashtirilmaydi)
JPEG_QUALITY = 90             # 1-100: yuqori = tiniqroq, lekin trafik ko'proq
RECONNECT_DELAY = 3           # soniya: kamera uzilsa qayta ulanish oralig'i
ROTATE = cv2.ROTATE_90_CLOCKWISE   # tasvir yotgan bo'lsa buriladi. Kerak bo'lmasa: None
MIRROR = True                      # True = chap-o'ng almashtiriladi (ko'zgudagidek)
PORT = 8000
DEBUG = True                       # terminalda har tekshiruv natijasi chiqadi (sinov uchun)

# --- OVOZ (kompyuter dinamigida, faqat Windows) ---
SOUND_ON_SERVER = True             # begona aniqlanganda kompyuterda sirena chalinadi
SIREN_CYCLES = 5                   # sirena necha marta takrorlanadi

# --- TELEGRAM ---
TELEGRAM_TOKEN = "8558669824:AAEGoWmmV7ZUwb-a9QqtSWM0EYFwin5J1D8"
TELEGRAM_CHAT_IDS: list[str] = []  # bo'sh qoldirsangiz, dastur botga /start yozganlarni o'zi topadi
                                   # qo'lda yozish uchun: ["123456789"]
ALERT_DIR = "begonalar"            # begona rasmlari shu papkaga ham saqlanadi
# --------------------------------------------

os.makedirs(FACES_DIR, exist_ok=True)

# onnxruntime ishlayaptimi, dastur boshida tekshiramiz
try:
    import onnxruntime  # noqa: F401
    print(f"[+] onnxruntime {onnxruntime.__version__} ishlayapti")
except Exception as e:
    print("=" * 60)
    print(f"[XATO] onnxruntime yuklanmadi: {e}")
    print("Yechim:")
    print("  1) Visual C++ Redistributable o'rnating: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("  2) Kompyuterni qayta yoqing")
    print("  3) python -m pip install --force-reinstall onnxruntime==1.18.1")
    print("=" * 60)

from deepface import DeepFace  # noqa: E402

known_faces: list[tuple[str, np.ndarray]] = []   # (ism, embedding)
state = {"force_alert": False}                    # /test-alert shu bayroqni yoqadi


# ======================= YUZ TANISH =======================
def load_known_faces():
    """Xodimlar rasmlaridan embedding'larni bir marta hisoblab oladi."""
    known_faces.clear()
    for root, _, files in os.walk(FACES_DIR):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(root, FACES_DIR)
            name = os.path.splitext(f)[0] if rel == "." else os.path.basename(rel)
            try:
                rep = DeepFace.represent(
                    img_path=path,
                    model_name=MODEL,
                    detector_backend=DETECTOR,
                    enforce_detection=True,
                )
                known_faces.append((name, np.array(rep[0]["embedding"])))
            except Exception as e:
                print(f"[!] {path} rasmida yuz topilmadi: {e}")
    print(f"[+] {len(known_faces)} ta yuz yuklandi: {sorted({n for n, _ in known_faces})}")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def recognize(frame):
    """Kadrdagi barcha yuzlarni topadi. [(ism yoki None, (x, y, w, h)), ...] qaytaradi."""
    try:
        faces = DeepFace.represent(
            img_path=frame,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=True,   # yuz bo'lmasa ValueError beradi
        )
    except ValueError:
        return []                      # kadrda yuz yo'q

    results = []
    for face in faces:
        emb = np.array(face["embedding"])
        area = face["facial_area"]
        box = (area["x"], area["y"], area["w"], area["h"])

        best_name, best_dist = None, 1.0
        for name, known_emb in known_faces:
            d = cosine_distance(emb, known_emb)
            if d < best_dist:
                best_name, best_dist = name, d

        matched = best_name if best_dist <= THRESHOLD else None
        if DEBUG:
            print(f"[i] Yuz: {matched or 'BEGONA'} (eng yaqin: {best_name}, masofa {best_dist:.2f}, chegara {THRESHOLD})")
        results.append((matched, box))
    return results


def _siren():
    """Kompyuter dinamigida sirena chaladi (alohida oqimda, dasturni to'xtatmaydi)."""
    try:
        import winsound
        for _ in range(SIREN_CYCLES):
            winsound.Beep(1500, 300)
            winsound.Beep(900, 300)
    except Exception as e:
        print(f"[!] Ovoz chiqarib bo'lmadi (faqat Windows'da ishlaydi): {e}")


def on_alert():
    """Qorovulga signal berish joyi. Bu yerga o'zingizga kerakli narsani qo'shing."""
    print("🚨 DIQQAT: BEGONA SHAXS ANIQLANDI!")
    if SOUND_ON_SERVER:
        threading.Thread(target=_siren, daemon=True).start()


background_tasks: set = set()   # fon vazifalari to'planib qolmasligi uchun
resolved_chat_ids: list[str] = []   # ishlatiladigan chat ID lar (qo'lda yoki avtomatik)


def init_telegram():
    """Chat ID ni aniqlaydi (kerak bo'lsa botga yozilgan /start dan) va test xabar yuboradi."""
    resolved_chat_ids.clear()
    resolved_chat_ids.extend(str(c) for c in TELEGRAM_CHAT_IDS if c)

    if not TELEGRAM_TOKEN:
        print("[!] TELEGRAM_TOKEN yozilmagan, Telegram o'chiq.")
        return

    if not resolved_chat_ids:
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates", timeout=15
            )
            data = r.json()
            if not data.get("ok"):
                print(f"[!] Telegram token xato yoki ishlamayapti: {data}")
                return
            for upd in data.get("result", []):
                msg = upd.get("message") or upd.get("my_chat_member") or {}
                chat = msg.get("chat") or {}
                if chat.get("id") is not None and str(chat["id"]) not in resolved_chat_ids:
                    resolved_chat_ids.append(str(chat["id"]))
        except Exception as e:
            print(f"[!] Telegramga ulanib bo'lmadi: {e}")
            return

    if not resolved_chat_ids:
        print("[!] Chat ID topilmadi. Telegramda botingizni ochib /start yozing va dasturni qayta ishga tushiring.")
        return

    print(f"[+] Telegram chat ID: {resolved_chat_ids}")
    for chat_id in resolved_chat_ids:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                data={"chat_id": chat_id, "text": "✅ Xavfsizlik kamerasi tizimi ishga tushdi."},
                timeout=15,
            )
            print("[+] Telegram test xabari yuborildi" if r.ok
                  else f"[!] Telegram test xatosi: {r.status_code} {r.text}")
        except Exception as e:
            print(f"[!] Telegram test xabarini yuborib bo'lmadi: {e}")


def send_alert_photo(jpg_bytes: bytes, stamp: str):
    """Begona rasmini papkaga saqlaydi va Telegramga yuboradi (alohida oqimda ishlaydi)."""
    os.makedirs(ALERT_DIR, exist_ok=True)
    path = os.path.join(ALERT_DIR, time.strftime("begona_%Y%m%d_%H%M%S.jpg"))
    with open(path, "wb") as f:
        f.write(jpg_bytes)
    print(f"[+] Begona rasmi saqlandi: {path}")

    chat_ids = list(resolved_chat_ids)
    if not TELEGRAM_TOKEN or not chat_ids:
        print("[!] Telegram tayyor emas (token yoki chat ID yo'q), rasm yuborilmadi.")
        return

    date_part, time_part = stamp.split(" ")
    caption = f"🚨 DIQQAT: BEGONA SHAXS!\n📅 Sana: {date_part}\n🕒 Soat: {time_part}"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    for chat_id in chat_ids:
        try:
            r = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": ("begona.jpg", jpg_bytes, "image/jpeg")},
                timeout=20,
            )
            if r.ok:
                print(f"[+] Telegramga yuborildi (chat {chat_id})")
            else:
                print(f"[!] Telegram xatosi (chat {chat_id}): {r.status_code} {r.text}")
        except Exception as e:
            print(f"[!] Telegramga yuborib bo'lmadi (chat {chat_id}): {e}")


# ======================= WEBSOCKET =======================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, message: dict):
        text = json.dumps(message)
        for ws in list(self.active_connections):
            try:
                await ws.send_text(text)
            except Exception:
                self.disconnect(ws)    # uzilgan ulanishni o'chiramiz


manager = ConnectionManager()


# ======================= KAMERA =======================
def open_camera():
    cam = cv2.VideoCapture(DROIDCAM_URL)
    if cam.isOpened():
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # kechikishni kamaytiradi
        print(f"[+] Kamera ulandi: {DROIDCAM_URL}")
    else:
        print(f"[!] Kamera ochilmadi: {DROIDCAM_URL}")
    return cam


async def send_status_only(status: str, message: str):
    """Kadr bo'lmaganda faqat holat xabarini yuboradi."""
    if manager.active_connections:
        await manager.broadcast({"status": status, "message": message, "image": None})


async def camera_loop():
    """Kamerani bitta fon vazifasi o'qiydi, barcha mijozlarga yuboradi."""
    camera = await asyncio.to_thread(open_camera)

    status, message = "CLEAR", "Kamera faol. Yuz kutilmoqda..."
    results = []              # oxirgi tanish natijalari (ramka chizish uchun)
    unknown_streak = 0
    last_check = 0.0
    last_alert = 0.0
    pending_alert = False     # True bo'lsa, keyingi chizilgan kadr Telegramga yuboriladi
    task = None

    while True:
        # Kamera ochiq emas bo'lsa, qayta ulanishga harakat qilamiz
        if not camera.isOpened():
            await send_status_only(
                "NOCAM", f"Kamera ulanmayapti ({DROIDCAM_URL}). Qayta urinilmoqda..."
            )
            camera.release()
            await asyncio.sleep(RECONNECT_DELAY)
            camera = await asyncio.to_thread(open_camera)
            continue

        ret, frame = await asyncio.to_thread(camera.read)
        if not ret or frame is None:
            await send_status_only("NOCAM", "Kamera signali uzildi. Qayta ulanilmoqda...")
            camera.release()
            await asyncio.sleep(RECONNECT_DELAY)
            camera = await asyncio.to_thread(open_camera)
            continue

        if ROTATE is not None:
            frame = cv2.rotate(frame, ROTATE)

        # 1) Fon vazifasi tugagan bo'lsa, natijani olamiz
        if task is not None and task.done():
            try:
                results = task.result()
            except Exception as e:
                print(f"Tanishda xatolik: {e}")
                results = []
            task = None

            unknown_count = sum(1 for n, _ in results if n is None)
            if unknown_count:
                unknown_streak += 1
            else:
                unknown_streak = 0

            if DEBUG and unknown_count:
                print(f"[i] Begona ketma-ketligi: {unknown_streak}/{ALERT_AFTER}")

            if unknown_streak >= ALERT_AFTER:
                status, message = "ALERT", "DIQQAT: BEGONA SHAXS!"
                if time.time() - last_alert > ALERT_COOLDOWN:
                    last_alert = time.time()
                    on_alert()
                    pending_alert = True
            elif results and unknown_count == 0:
                names = ", ".join(sorted({n.upper() for n, _ in results}))
                status, message = "EMPLOYEE", f"Xodim: {names}"
            elif unknown_count:
                status, message = "CHECKING", "Tekshirilmoqda..."
            else:
                status, message = "CLEAR", "Yuz aniqlanmadi"

        # /test-alert chaqirilsa, begona aniqlanmasa ham signalni sinab ko'ramiz
        if state["force_alert"]:
            state["force_alert"] = False
            print("[i] Test signal: sirena va Telegram rasmi yuborilmoqda...")
            on_alert()
            pending_alert = True

        # 2) Yangi tekshiruvni fon rejimida boshlaymiz (oqimni to'xtatmaydi)
        if (
            task is None
            and known_faces
            and manager.active_connections
            and time.time() - last_check > CHECK_EVERY
        ):
            last_check = time.time()
            task = asyncio.create_task(asyncio.to_thread(recognize, frame.copy()))

        if not known_faces:
            status, message = "CLEAR", "xodimlar papkasi bo'sh yoki yuz topilmadi!"

        # 3) Kadrni ko'zgudek aylantiramiz (tanish asl kadrda bo'lgan) va ramkalar chizamiz
        if MIRROR:
            frame = cv2.flip(frame, 1)
        fw = frame.shape[1]
        for name, (x, y, w, h) in results:
            if MIRROR:
                x = fw - x - w
            color = (0, 200, 0) if name else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name or "BEGONA", (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Begona aniqlansa, ramkali kadrni Telegramga yuboramiz (fon rejimida, oqimni to'xtatmaydi)
        if pending_alert:
            pending_alert = False
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            alert_frame = frame.copy()          # vaqt faqat yuboriladigan rasmga yoziladi
            ah, aw = alert_frame.shape[:2]
            scale = max(0.6, aw / 900)
            (tw, th), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            cv2.rectangle(alert_frame, (0, ah - th - 24), (tw + 24, ah), (0, 0, 0), -1)
            cv2.putText(alert_frame, stamp, (12, ah - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
            ok, alert_buf = cv2.imencode(".jpg", alert_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                t = asyncio.create_task(
                    asyncio.to_thread(send_alert_photo, alert_buf.tobytes(), stamp))
                background_tasks.add(t)
                t.add_done_callback(background_tasks.discard)

        # 4) Brauzerga yuboramiz
        if manager.active_connections:
            h0, w0 = frame.shape[:2]
            if w0 > SEND_WIDTH:   # faqat kichraytiramiz, kattalashtirib xiralashtirmaymiz
                small = cv2.resize(frame, (SEND_WIDTH, int(h0 * SEND_WIDTH / w0)),
                                   interpolation=cv2.INTER_AREA)
            else:
                small = frame
            _, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            await manager.broadcast({
                "status": status,
                "message": message,
                "image": "data:image/jpeg;base64," + base64.b64encode(buffer).decode(),
            })

        await asyncio.sleep(0.03)


# ======================= ILOVA =======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_known_faces)
    await asyncio.to_thread(init_telegram)
    loop_task = asyncio.create_task(camera_loop())
    yield
    loop_task.cancel()


app = FastAPI(lifespan=lifespan)

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="uz">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Xavfsizlik kamerasi</title>
<style>
  body { margin:0; font-family:Arial,sans-serif; background:#111; color:#fff; text-align:center; }
  h1 { margin:16px 0 8px; font-size:22px; }
  #banner { margin:8px auto; padding:14px; width:90%; max-width:640px; border-radius:8px;
            font-size:20px; font-weight:bold; background:#444; transition:background .2s; }
  .CLEAR { background:#444 !important; }
  .EMPLOYEE { background:#1b8f3a !important; }
  .CHECKING { background:#b8860b !important; }
  .NOCAM { background:#6b3fa0 !important; }
  .ALERT { background:#d11a2a !important; animation:blink .6s infinite; }
  @keyframes blink { 50% { opacity:.5; } }
  #video { max-width:90%; max-height:80vh; width:auto; height:auto; object-fit:contain;
           border-radius:8px; background:#000; min-height:240px; }
  button { margin:12px; padding:10px 20px; font-size:16px; border:0; border-radius:6px;
           background:#2d6cdf; color:#fff; cursor:pointer; }
  #conn { font-size:13px; color:#aaa; }
</style>
</head>
<body>
  <h1>Xavfsizlik kamerasi</h1>
  <div id="banner" class="CLEAR">Ulanmoqda...</div>
  <img id="video" alt="Kamera">
  <div><button id="soundBtn">🔔 Ovozli signalni yoqish</button></div>
  <div id="conn"></div>

<script>
  const banner = document.getElementById("banner");
  const video = document.getElementById("video");
  const conn = document.getElementById("conn");
  const soundBtn = document.getElementById("soundBtn");

  let audioCtx = null, soundOn = false, lastBeep = 0;

  soundBtn.onclick = () => {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    soundOn = !soundOn;
    if (soundOn) { audioCtx.resume(); lastBeep = 0; beep(); }   // yoqilganda test ovozi
    soundBtn.textContent = soundOn ? "🔕 Ovozni o'chirish" : "🔔 Ovozli signalni yoqish";
  };

  function beep() {
    if (!soundOn || !audioCtx) return;
    const now = Date.now();
    if (now - lastBeep < 1000) return;
    lastBeep = now;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "square";
    osc.frequency.value = 900;
    gain.gain.value = 0.6;
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.start(); osc.stop(audioCtx.currentTime + 0.4);
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss://" : "ws://";
    const ws = new WebSocket(proto + location.host + "/ws/stream");

    ws.onopen = () => { conn.textContent = "Server bilan ulandi"; };
    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.image) video.src = d.image;
      banner.textContent = d.message;
      banner.className = d.status;
      if (d.status === "ALERT") beep();
    };
    ws.onclose = () => {
      conn.textContent = "Ulanish uzildi, qayta ulanmoqda...";
      setTimeout(connect, 2000);
    };
    ws.onerror = () => ws.close();
  }
  connect();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()   # ulanish uzilganini aniqlash uchun
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/test-alert")
async def test_alert():
    """Brauzerda http://localhost:8000/test-alert ochilsa, sirena va Telegram rasmi sinab ko'riladi."""
    state["force_alert"] = True
    return {"ok": True, "message": "Test signal yuborildi. Terminalni va Telegramni kuzating."}


@app.post("/reload")
async def reload_faces():
    """Yangi xodim rasmi qo'shilgandan keyin serverni o'chirmasdan bazani yangilaydi."""
    await asyncio.to_thread(load_known_faces)
    return {"loaded": len(known_faces)}


# ======================= ISHGA TUSHIRISH =======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)