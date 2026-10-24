# 🎯 FaceTrack: Face Recognition Attendance System

A webcam-based attendance system. A person stands in front of the camera and attendance is marked automatically (no button press). Unknown faces get a **"Not enrolled"** pop-up where they can enroll on the spot.

## Features

- **Live auto attendance**: the camera recognises an enrolled face and marks **IN** instantly.
- **Auto OUT**: the same face seen again after the configured gap (default 1 minute) marks **OUT**.
- **Kiosk sessions**: after each result the session resets so the next person can step in.
- **Enroll pop-up**: an unknown face triggers a pop-up with the captured photo and a name field.
- **Enroll tab**: enroll manually from the camera, view enrolled people with photos, delete anyone.
- **Records tab**: view all attendance, delete selected or all records, export CSV.
- Stable matching: a face must be matched 3 times in a row before attendance is written.

## Tech stack

| Part | Tool |
|---|---|
| UI | Streamlit, streamlit-webrtc |
| Face detection and encoding | `face_recognition` (dlib) |
| Video | OpenCV, PyAV |
| Database | SQLite (photos + 128-d face encodings + attendance) |

## How it works

1. **Enroll**: a photo is converted to a 128-number face encoding and saved with the name and photo in the database.
2. **Live scan**: frames from the webcam are checked for faces; the largest face is compared against stored encodings (distance threshold `0.5`).
3. **Marking**: 3 consecutive matches = confirmed. No record today = `IN`. Record exists and gap >= `MIN_GAP` = `OUT`.
4. **Unknown face**: 5 consecutive non-matches opens the enroll pop-up.

## Project structure

```
app.py             # the whole application
requirements.txt   # Python dependencies
packages.txt       # system packages (for Streamlit Cloud)
README.md
.gitignore
attendance.db      # created automatically on first run (not committed)
```

## Run locally (Windows)

Python **3.10 or 3.11** is recommended.

```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process Bypass
venv\Scripts\activate

pip install dlib-bin
pip install face_recognition --no-deps
pip install face_recognition_models click numpy pillow opencv-python streamlit streamlit-webrtc pandas "setuptools<81"

streamlit run app.py
```

Open `http://localhost:8501` and allow camera access.

## Usage

1. Open the **Live Attendance** tab and press **START**.
2. A new person appears: the **Not enrolled** pop-up opens. Enter a name and press **Enroll** (enroll 2-3 times from different angles for better accuracy).
3. Stand in front of the camera again: **IN** is marked.
4. After the OUT gap passes, show the face again: **OUT** is marked.
5. Check or export everything in the **Records** tab.

## Configuration (top of `app.py`)

| Variable | Default | Meaning |
|---|---|---|
| `THRESHOLD` | `0.5` | Match strictness (lower = stricter) |
| `MIN_GAP` | `60` | Seconds after IN before OUT can be marked |
| `SHOW_SEC` | `4` | How long a result is shown before the next session |

## Deploy on Streamlit Community Cloud

1. Push this project to a **private** GitHub repository (see `.gitignore`; never push `attendance.db`).
2. Go to https://share.streamlit.io, click **New app**, select the repo, branch `main`, main file `app.py`.
3. Open **Advanced settings** and choose **Python 3.11**.
4. Click **Deploy**. The first build compiles dlib and can take 10-15 minutes.

### Known limitations of the free cloud deployment

- **Data is not permanent.** The SQLite file lives on the app's temporary disk, so enrolled people and attendance can be lost when the app restarts or redeploys. For permanent storage, move the database to a hosted one (Supabase/Neon PostgreSQL).
- **Live camera on the cloud** uses WebRTC. On some networks it needs a TURN server in addition to the STUN server already configured in `app.py`.
- Free apps go to sleep when idle; the first load after that is slow.

## Privacy

This app stores **face photos and biometric encodings**. Only enroll people who have given consent, keep the repository and database private, and delete data when it is no longer needed (the delete buttons in the Enroll tab remove a person's photos and face data).

## Author

Muhammad Farhan
