import requests
import streamlit as st, sqlite3, pickle, os, time, threading
import numpy as np, pandas as pd, cv2, av
import fr_lite as face_recognition
from PIL import Image
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="FaceTrack", page_icon="🎯", layout="wide")

import tempfile
_here = os.path.dirname(os.path.abspath(__file__))
if _here.startswith("/mount"):          # Streamlit Cloud: yahan likhna mana hai
    DB = os.path.join(tempfile.gettempdir(), "attendance.db")
else:                                   # local computer
    DB = os.path.join(_here, "attendance.db")
THRESHOLD = 0.5      # kam = zyada strict
MIN_GAP = 60         # IN ke itne second baad OUT lagega
SHOW_SEC = 4         # result kitni der dikhe, phir naya session

# ---------------- Style ----------------
st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 20% 0%, #1b1f3b 0%, #0b0d1a 55%);}
#MainMenu, footer {visibility: hidden;}
.hero {padding: 26px 30px; border-radius: 22px; margin-bottom: 16px;
  background: linear-gradient(120deg, #6a11cb 0%, #2575fc 60%, #00d2ff 100%);
  box-shadow: 0 10px 40px rgba(37,117,252,.35);}
.hero h1 {color:#fff; margin:0; font-size:2.3rem; font-weight:800;}
.hero p {color:#e8f0ff; margin:6px 0 0 0;}
.stat {background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12);
  border-radius:18px; padding:16px; text-align:center;}
.stat .num {font-size:2rem; font-weight:800; color:#fff;}
.stat .lbl {color:#9fb3ff; font-size:.85rem; text-transform:uppercase; letter-spacing:1px;}
.res {border-radius:18px; padding:20px 24px; margin:10px 0; color:#fff; font-size:1.1rem;}
.res b {font-size:1.4rem;}
.in  {background: linear-gradient(120deg,#11998e,#38ef7d); color:#04281f;}
.out {background: linear-gradient(120deg,#f7971e,#ffd200); color:#3a2600;}
.warn{background: linear-gradient(120deg,#4e54c8,#8f94fb);}
.bad {background: linear-gradient(120deg,#cb2d3e,#ef473a);}
.stTabs [data-baseweb="tab"] {background:rgba(255,255,255,.06); border-radius:12px; padding:10px 22px; color:#cfd8ff;}
.stTabs [aria-selected="true"] {background: linear-gradient(120deg,#6a11cb,#2575fc); color:#fff;}
.stButton>button, .stFormSubmitButton>button {border:0; border-radius:14px; font-weight:700;
  color:#fff; background: linear-gradient(120deg,#6a11cb,#2575fc);}
</style>
""", unsafe_allow_html=True)

# ---------------- Database ----------------
@st.cache_data(ttl=3600)
def get_ice_servers():
    stun = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        app = st.secrets["METERED_APP"]
        key = st.secrets["METERED_KEY"]
        r = requests.get(f"https://{app}.metered.live/api/v1/turn/credentials",
                         params={"apiKey": key}, timeout=8)
        r.raise_for_status()
        return stun + r.json()
    except Exception:
        return stun 
@st.cache_resource
def get_conn():
    c = sqlite3.connect(DB, check_same_thread=False, timeout=10)
    c.execute("CREATE TABLE IF NOT EXISTS students (name TEXT, encoding BLOB, photo BLOB)")
    c.execute("CREATE TABLE IF NOT EXISTS attendance (name TEXT, date TEXT, in_time TEXT, out_time TEXT)")
    cols = [r[1] for r in c.execute("PRAGMA table_info(students)")]
    if "photo" not in cols:
        c.execute("ALTER TABLE students ADD COLUMN photo BLOB")
    c.commit()
    return c
conn = get_conn()

def result_card(kind, html):
    st.markdown(f'<div class="res {kind}">{html}</div>', unsafe_allow_html=True)

def get_face(file):
    """camera_input / upload se (encoding, jpeg_bytes, error)"""
    img = Image.open(file).convert("RGB")
    if img.width > 640:
        img = img.resize((640, int(img.height * 640 / img.width)))
    arr = np.array(img)
    boxes = face_recognition.face_locations(arr)
    if len(boxes) == 0:
        return None, None, "Face nazar nahi aya. Roshni mein seedha dekhein."
    if len(boxes) > 1:
        return None, None, "Ek se zyada face hain. Frame mein sirf ek banda rahe."
    enc = face_recognition.face_encodings(arr, boxes)[0]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return enc, buf.tobytes(), None

def save_student(name, enc, photo):
    conn.execute("INSERT INTO students (name, encoding, photo) VALUES (?,?,?)",
                 (name.strip(), pickle.dumps(enc), photo))
    conn.commit()

# ---------------- Live processor (kiosk) ----------------
class Kiosk(VideoProcessorBase):
    def __init__(self):
        self.conn = sqlite3.connect(DB, check_same_thread=False, timeout=10)
        self.lock = threading.Lock()
        self.min_gap = MIN_GAP
        self.i = 0
        self.reset()

    def reset(self):
        """Naya session: sab kuch saaf, agla banda aa sakta hai"""
        self.state = "scanning"      # scanning / result / unknown
        self.kind, self.msg = None, ""
        self.until = 0
        self.loaded = False
        self.names, self.encs = [], np.empty((0, 128))
        self.boxes = []
        self.streak_name, self.streak, self.unk = None, 0, 0
        self.unk_enc, self.unk_photo = None, None

    def load(self):
        with self.lock:
            rows = self.conn.execute("SELECT name, encoding FROM students").fetchall()
        self.names = [r[0] for r in rows]
        self.encs = np.array([pickle.loads(r[1]) for r in rows]) if rows else np.empty((0, 128))
        self.loaded = True

    def mark(self, name):
        now = datetime.now()
        today, tnow = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
        with self.lock:
            row = self.conn.execute("SELECT rowid, in_time FROM attendance WHERE name=? AND date=?",
                                    (name, today)).fetchone()
            if row is None:
                self.conn.execute("INSERT INTO attendance VALUES (?,?,?,NULL)", (name, today, tnow))
                self.conn.commit()
                return "in", f"{name} - IN {tnow}"
            gap = (now - datetime.strptime(f"{today} {row[1]}", "%Y-%m-%d %H:%M:%S")).total_seconds()
            if gap < self.min_gap:
                return "warn", f"{name} - IN already at {row[1]}"
            self.conn.execute("UPDATE attendance SET out_time=? WHERE rowid=?", (tnow, row[0]))
            self.conn.commit()
            return "out", f"{name} - OUT {tnow}"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        now = time.time()

        if self.state == "result" and now >= self.until:
            self.reset()
        if self.state == "unknown" and now >= self.until:     # pop-up band kar diya gaya
            self.reset()

        if self.state == "scanning":
            if not self.loaded:
                self.load()
            self.i += 1
            if self.i % 4 == 0:
                small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb)
                self.boxes = []
                if boxes:
                    boxes.sort(key=lambda b: (b[2]-b[0]) * (b[1]-b[3]), reverse=True)
                    box = boxes[0]                              # sab se bara face
                    enc = face_recognition.face_encodings(rgb, [box])[0]
                    name, conf = "Unknown", 0
                    if len(self.names):
                        d = face_recognition.face_distance(self.encs, enc)
                        best = int(np.argmin(d))
                        if d[best] < THRESHOLD:
                            name, conf = self.names[best], int((1 - d[best]) * 100)
                    t, r, b, l = [v * 2 for v in box]
                    self.boxes = [((t, r, b, l), name, conf)]

                    if name != "Unknown":
                        self.unk = 0
                        if name == self.streak_name:
                            self.streak += 1
                        else:
                            self.streak_name, self.streak = name, 1
                        if self.streak >= 3:                    # 3 baar lagatar match = pakka
                            self.kind, self.msg = self.mark(name)
                            self.until = now + SHOW_SEC
                            self.state = "result"
                    else:
                        self.streak, self.streak_name = 0, None
                        self.unk += 1
                        if self.unk >= 5:                       # unknown pakka
                            pad = 40
                            crop = img[max(0, t-pad):min(h, b+pad), max(0, l-pad):min(w, r+pad)]
                            ok, buf = cv2.imencode(".jpg", crop)
                            self.unk_enc, self.unk_photo = enc, buf.tobytes()
                            self.until = now + 40
                            self.state = "unknown"
                else:
                    self.streak, self.streak_name, self.unk = 0, None, 0

        # ---- drawing ----
        for (t, r, b, l), name, conf in self.boxes:
            col = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (l, t), (r, b), col, 2)
            label = f"{name} {conf}%" if name != "Unknown" else "Unknown"
            cv2.rectangle(img, (l, b - 28), (r, b), col, -1)
            cv2.putText(img, label, (l + 6, b - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.state == "result":
            col = {"in": (60, 180, 40), "out": (0, 150, 255), "warn": (200, 110, 60)}[self.kind]
            cv2.rectangle(img, (0, 0), (w, 70), col, -1)
            cv2.putText(img, self.msg, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        elif self.state == "unknown":
            cv2.rectangle(img, (0, 0), (w, 70), (0, 0, 220), -1)
            cv2.putText(img, "NOT ENROLLED", (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- Pop-up ----------------
@st.dialog("❌ Not enrolled")
def enroll_popup(vp):
    enc, photo = vp.unk_enc, vp.unk_photo
    if photo:
        st.image(photo, width=180)
    st.write("Ye face database mein nahi hai. Enroll karna chahte hain?")
    with st.form("popup_enroll"):
        nm = st.text_input("Naam")
        c1, c2 = st.columns(2)
        ok = c1.form_submit_button("➕ Enroll")
        skip = c2.form_submit_button("Skip")
    if ok:
        if not nm.strip():
            st.warning("Naam likhein.")
            return
        save_student(nm, enc, photo)
        vp.reset()
        st.toast(f"{nm.strip()} enroll ho gaya. Ab dobara camera ke saamne aayein.")
        st.rerun()
    if skip:
        vp.reset()
        st.rerun()

# ---------------- Header ----------------
st.markdown("""<div class="hero"><h1>🎯 FaceTrack</h1>
<p>Camera ke saamne aayein, attendance khud lag jati hai. Naya banda? Enroll ka pop-up aa jata hai.</p></div>""",
            unsafe_allow_html=True)

today = datetime.now().strftime("%Y-%m-%d")
n_people = conn.execute("SELECT COUNT(DISTINCT name) FROM students").fetchone()[0]
n_present = conn.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,)).fetchone()[0]
n_out = conn.execute("SELECT COUNT(*) FROM attendance WHERE date=? AND out_time IS NOT NULL", (today,)).fetchone()[0]
c1, c2, c3 = st.columns(3)
for col, num, lbl in [(c1, n_people, "Enrolled people"), (c2, n_present, "Present today"), (c3, n_out, "Checked out")]:
    col.markdown(f'<div class="stat"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
                 unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3 = st.tabs(["🎥 Live Attendance", "➕ Enroll", "📊 Records"])

# ---------------- Tab 1: Live ----------------
with tab1:
    gap_min = st.slider("IN ke kitne minute baad OUT lage", 1, 60, 1)
    ctx = webrtc_streamer(
        key="kiosk",
        video_processor_factory=Kiosk,
        rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    vp = ctx.video_processor
    if vp:
        vp.min_gap = gap_min * 60
        if vp.state == "unknown":
            enroll_popup(vp)

    @st.fragment(run_every=1)
    def status():
        today = datetime.now().strftime("%Y-%m-%d") 
        v = ctx.video_processor
        if not v:
            st.session_state.unk_seen = False
            st.info("START dabayein, phir camera ke saamne aayein.")
            return
        if v.state == "unknown":
            if not st.session_state.get("unk_seen"):
                st.session_state.unk_seen = True
                st.rerun()                       # pop-up kholne ke liye
            result_card("bad", "❌ <b>Not enrolled</b><br>Pop-up mein enroll karein.")
        else:
            st.session_state.unk_seen = False
            if v.state == "result":
                result_card(v.kind, f"<b>{v.msg}</b>")
            else:
                result_card("warn", "👀 Camera ke saamne aayein...")
        rows = conn.execute("SELECT name, in_time, out_time FROM attendance WHERE date=? "
                            "ORDER BY rowid DESC LIMIT 6", (today,)).fetchall()
        if rows:
            st.caption("Aaj ki latest entries")
            st.dataframe(pd.DataFrame(rows, columns=["Name", "IN", "OUT"]),
                         use_container_width=True, hide_index=True)
    status()

# ---------------- Tab 2: Enroll ----------------
with tab2:
    left, right = st.columns([1, 1])
    with left:
        name = st.text_input("Naam", placeholder="e.g. Muhammad Farhan")
        eshot = st.camera_input("Enroll ke liye photo", key="enr_cam")
        if st.button("➕ Enroll karo"):
            if not name.strip():
                result_card("warn", "Naam likhein.")
            elif eshot is None:
                result_card("warn", "Pehle photo lein.")
            else:
                enc, photo, err = get_face(eshot)
                if err:
                    result_card("bad", f"⚠️ {err}")
                else:
                    save_student(name, enc, photo)
                    result_card("in", f"🎉 <b>{name.strip()}</b> enroll ho gaya")
                    st.caption("Behtar accuracy ke liye isi naam se 3-5 photos (alag angle) enroll karein.")
    with right:
        st.markdown("### Enrolled log")
        rows = conn.execute("""SELECT s.name, s.photo, c.cnt FROM students s
            JOIN (SELECT name, MIN(rowid) r, COUNT(*) cnt FROM students GROUP BY name) c
            ON s.rowid = c.r ORDER BY s.name""").fetchall()
        if rows:
            cols = st.columns(3)
            for k, (n, p, cnt) in enumerate(rows):
                with cols[k % 3]:
                    if p:
                        st.image(p, use_container_width=True)
                    st.caption(f"{n} ({cnt} photos)")
                    if st.button("🗑️ Delete", key=f"del_{k}_{n}"):
                        conn.execute("DELETE FROM students WHERE name=?", (n,))
                        conn.commit()
                        st.toast(f"{n} delete ho gaya")
                        st.rerun()

            with st.expander("⚠️ Saare enrolled log delete karo"):
                sure2 = st.checkbox("Haan, sab enrolled log hatao", key="sure_enrolled")
                if st.button("Sab enrolled delete", disabled=not sure2):
                    conn.execute("DELETE FROM students")
                    conn.commit()
                    st.rerun()
        else:
            st.info("Abhi koi enrolled nahi.")
        if os.path.exists("encodings.pkl"):
            if st.button("📥 Kaggle dataset ki encodings import karo"):
                d = pickle.load(open("encodings.pkl", "rb"))
                for n, e in zip(d["names"], d["encodings"]):
                    conn.execute("INSERT INTO students (name, encoding) VALUES (?,?)", (n, pickle.dumps(e)))
                conn.commit()
                st.rerun()
# ---------------- Tab 3: Records ----------------
with tab3:
    df = pd.read_sql_query("SELECT rowid AS ID, name AS Name, date AS Date, in_time AS 'IN', out_time AS 'OUT' "
                           "FROM attendance ORDER BY date DESC, in_time DESC", conn)
    if df.empty:
        st.info("Abhi koi attendance record nahi.")
    else:
        st.caption("Jo records delete karne hain, unki row ke left par tick lagayein.")
        event = st.dataframe(df, use_container_width=True, hide_index=True,
                             on_select="rerun", selection_mode="multi-row", key="rec_table")
        picked = event.selection.rows
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"🗑️ Selected delete karo ({len(picked)})", disabled=not picked):
                ids = [int(df.iloc[i]["ID"]) for i in picked]
                conn.executemany("DELETE FROM attendance WHERE rowid=?", [(i,) for i in ids])
                conn.commit()
                st.toast(f"{len(ids)} record delete ho gaye")
                st.rerun()
        with c2:
            st.download_button("⬇️ CSV download", df.drop(columns="ID").to_csv(index=False),
                               "attendance.csv", "text/csv")

        with st.expander("⚠️ Saare records delete karo"):
            sure = st.checkbox("Haan, mujhe yaqeen hai")
            if st.button("Sab attendance records delete", disabled=not sure):
                conn.execute("DELETE FROM attendance")
                conn.commit()
                st.rerun()
