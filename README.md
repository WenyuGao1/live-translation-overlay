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





graph TD
    %% ================= STYLE DEFINITIONS =================
    classDef process fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#000;
    classDef storage fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px,color:#000;
    classDef decision fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#000;
    classDef terminator fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#000;
    classDef threadBoundary fill:none,stroke:#90A4AE,stroke-width:2px,stroke-dasharray: 5 5;

    %% ================= MAIN FLOW =================
    Start([Initialize Application]) --> LoadModels[Load Models & Config\nWhisper ASR & NLLB MT\nDeferred Loading]
    LoadModels --> InitDB[(Initialize SQLite DB\nPersistent History)]
    InitDB --> StartThreads[Start Worker Threads]

    %% ================= AUDIO INPUT =================
    subgraph Audio Capture
        Mic(Microphone Input\nsounddevice) --> AudioCB[Audio Callback]
        AudioCB --> AudioQ[(Audio Chunk Queue\nThread-safe)]
    end

    %% ================= WORKER THREAD: ASR & TRANSLATION =================
    subgraph ASR Worker Thread [ASR & Translation Thread]
        direction TB
        StartThreads -.-> ASR_Loop
        ASR_Loop(Main Loop) --> DequeueAudio{Read Audio Queue\n& Normalize}
        AudioQ -.-> DequeueAudio
        DequeueAudio --> OverlapHandle[Handle Audio Overlap\nReduce boundary drops]
        OverlapHandle --> WhisperASR[ASR Transcription English\nfaster-whisper large-v3]
        WhisperASR --> TextProcessing[Text Processing\nDe-duplication & Overlap Merging]
        TextProcessing --> SentenceSplit[Sentence Splitting]
        
        SentenceSplit --> IsComplete{Complete Sentence\nOR\nIdle Timeout Finalize?}
        IsComplete -- No --> BufferText[Buffer Pending English Text]
        BufferText --> ASR_Loop
        
        IsComplete -- Yes --> Translate[Machine Translation MT\nNLLB-200 EN to Target Lang\nSelectable Runtime Target]
        Translate --> Commit[Commit Segment\nEN + Translated Text]
    end

    %% ================= DATA DISTRIBUTION =================
    Commit -- Add to UI Buffer --> RingBuffer[(UI Ring Buffer\nMax segments kept small)]
    Commit -- Enqueue for DB --> LogQ[(Logger Queue\nAsync buffer)]

    %% ================= WORKER THREAD: DATABASE LOGGER =================
    subgraph Logger Thread [Async SQLite Logger Thread]
        StartThreads -.-> LoggerLoop
        LogQ -.-> LoggerLoop(Read Logger Queue)
        LoggerLoop --> BatchInsert[Batch Insert Records\nAvoids UI stalls]
        BatchInsert --> SQLite[(Persistent SQLite DB\ntranslation_history.sqlite3)]
    end

    %% ================= MAIN THREAD: UI & EXPORT =================
    subgraph Main UI Thread [Main Thread: Tkinter UI]
        StartThreads -.-> UIPoll
        UIPoll(UI Polling Loop\nInterval: 0.25s) --> ReadRingBuffer{Read Ring Buffer}
        RingBuffer -.-> ReadRingBuffer
        
        ReadRingBuffer --> RenderUI[Render Two-Window UI]
        
        subgraph UI Components
            RenderUI -- Behind --> BackgroundWin[Background Window\nSemi-transparent Controller]
            RenderUI -- Top --> OverlayWin[Chroma-Key Overlay Window\nCrisp Subtitles, Transparent BG]
            BackgroundWin <--> Settings[Settings Menu\nTarget Lang, Colors, Audio Device]
        end
        
        Settings -- Change Lang --> UpdateMTCfg[Update MT Config\nRuntime change]
        UpdateMTCfg -.-> Translate
        
        BackgroundWin -- Trigger Export --> ExportTask[Start Export Task\nCurrent Session]
    end

    %% ================= EXPORT PROCESS =================
    subgraph Export Process [Temp Thread]
        ExportTask --> ReadDB[Fetch Session Data\nfrom SQLite]
        SQLite -.-> ReadDB
        ReadDB --> GenerateDocx[Generate Word Document .docx\nFull current session history]
        GenerateDocx --> AutoSplit{Record count > limit?\nMax 2000/file}
        AutoSplit -- Yes --> SplitFiles[Auto-Split into multiple files\n_part00N.docx]
        AutoSplit -- No --> SingleFile[Save single .docx file]
        SplitFiles --> SaveToDisk[Save to Disk]
        SingleFile --> SaveToDisk
    end

    %% ================= STYLING APPLICATION =================
    class Start,LoadModels,InitDB,StartThreads,AudioCB,OverlapHandle,WhisperASR,TextProcessing,SentenceSplit,BufferText,Translate,Commit,LoggerLoop,BatchInsert,UIPoll,RenderUI,BackgroundWin,OverlayWin,Settings,UpdateMTCfg,ExportTask,ReadDB,GenerateDocx,SplitFiles,SingleFile,SaveToDisk process;
    class AudioQ,RingBuffer,LogQ,SQLite,Mic storage;
    class DequeueAudio,IsComplete,ReadRingBuffer,AutoSplit decision;
    class ASR_Worker_Thread,Logger_Thread,Main_UI_Thread threadBoundary;
