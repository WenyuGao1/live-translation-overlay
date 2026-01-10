# Live Translation Overlay (EN -> Target Language)

A two-window desktop overlay for real-time English speech recognition + translation, with persistent session history and one-click Word export (auto-split).

**Important:** First run downloads **Whisper + NLLB** models from Hugging Face. Internet + disk cache required.

---

## Features
- Real-time microphone capture (sounddevice)
- English ASR via faster-whisper (WhisperModel)
- Audio overlap + text de-duplication to reduce boundary drops/repeats
- Sentence splitting + idle finalize
- Translation via NLLB-200 distilled 1.3B (Transformers)
- Target language selectable at runtime (**dropdown shows English names**, not NLLB codes)
- Two-window UI: semi-transparent background + crisp chroma-key overlay subtitles
- Persistent history stored in SQLite (async, best-effort; may drop records under overload)
- Export full history of the **CURRENT SESSION** to Word (.docx), auto-splitting into multiple files (by record count)

---

## Screenshot 

<img width="2106" height="1357" alt="screenshot" src="https://github.com/user-attachments/assets/3e06e9ec-fbb4-4a17-a126-f64b6ebaf59f" />

## Pipline

flowchart TD
  A[App Start<br/>(Local / Offline)] --> B[Load NLLB-200 + Language Map<br/>(startup)]
  B --> C[(SQLite History<br/>translation_history.sqlite3)]
  C --> D[Start Threads<br/>(ASR + Logger)]

  D --> Q[(AudioQ<br/>audio chunks)]
  Q --> E[ASR Worker<br/>Whisper large-v3<br/>(lazy-loaded)]
  E --> F[Overlap + De-dup<br/>reduce boundary drops]
  F --> G[Translate<br/>NLLB-200 EN → Target]
  G --> H[UI Render<br/>Tkinter 2-window overlay]

  G --> L[(LogQ<br/>async buffer)]
  L --> M[Logger Thread<br/>batch inserts]
  M --> C

  C --> X[Export Current Session<br/>Word (.docx)]
  X --> Y[Auto-split<br/>(e.g., 2000 records/file)]

  RT[Runtime target language switching] -.-> G
  TW[Two-window design<br/>(controller + overlay)] -.-> H
  NB[Non-blocking UI<br/>(async DB writes)] -.-> M


---

## Requirements
- Python 3.9+ (recommended 3.10/3.11)
- Windows recommended
  - Chroma-key transparency via Tk `-transparentcolor` is generally Windows-only; on macOS/Linux the key color may be visible.
- Microphone input device
- Optional: NVIDIA GPU (CUDA) for faster inference

---

## Quick Start (Windows)
~~~bash
python -m venv .venv
.venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt

python main.py
~~~

## Quick Start (macOS/Linux)
~~~bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

python main.py
~~~

---

## How to Use
- The app creates **two windows**:
  1) Background window (semi-transparent, movable/resizable)
  2) Overlay window (borderless subtitles)
- Right-click (overlay or background) menu:
  - Choose EN Color / Choose OUT Color / Quit
- Click **⚙** on the overlay to open Settings:
  - Show/Hide English subtitles
  - Target language dropdown (English names)
  - Audio input device selection
  - Export **CURRENT SESSION** history to Word (.docx)

---

## Export to Word (.docx)
- In Settings, tick: **Export THIS SESSION history**
- Choose a save path
- Export reads from the SQLite DB and writes one or more `.docx` files
- Auto-split every `DOCX_MAX_RECORDS_PER_FILE` records (default 2000)

Note: splitting is by **record count**, not by “lines” or character count.

---

## Optional Configuration (CLI)
Override Hugging Face cache dir:
~~~bash
python main.py --hf-home D:\hf_cache
~~~

Override history DB path:
~~~bash
python main.py --db-path D:\overlay\translation_history.sqlite3
~~~

Disable hf_transfer (if downloads fail):
~~~bash
python main.py --disable-hf-transfer
~~~

---

## Environment Variables (Alternative)
~~~text
HF_HOME=/path/to/cache
HISTORY_DB_PATH=/path/to/translation_history.sqlite3
WHISPER_CT2_REPO=...   (optional: force a specific faster-whisper CT2 repo)
~~~

---

## Privacy
- Audio is processed locally for ASR.
- Translation runs locally with NLLB.
- Internet is only required for first-run model download (unless already cached).

---

## Troubleshooting
### Hugging Face download timeout / failures
Try disabling hf_transfer:
~~~bash
python main.py --disable-hf-transfer
~~~
Also ensure enough disk space for the cache.

### Microphone / sounddevice issues (Windows)
- Confirm your microphone works in Windows Settings
- Reinstall:
~~~bash
pip install sounddevice
~~~
If device selection is wrong, choose the correct input device in Settings.

### Overlay transparency not working (non-Windows)
Expected: Tk `-transparentcolor` is generally Windows-only; on macOS/Linux the key color may remain visible.

---

## Acknowledgements
- faster-whisper (Whisper ASR)
- Meta NLLB-200 (facebook/nllb-200-distilled-1.3B) via Hugging Face Transformers
