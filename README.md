\# Worker Safety AI



Computer-vision safety monitoring for factories and construction sites.

Detects missing helmets and vests and restricted-zone intrusions in video,

logs every alert with a snapshot, and shows them on a web dashboard.



\## Pipeline



Video -> YOLO11 detection + ByteTrack tracking -> rule engine

(PPE, restricted zone) -> event log (SQLite + snapshots) -> FastAPI -> dashboard



\## Results (validation set, 340 images)



| Class | mAP50 |

|---|---|

| Person | 0.988 |

| Vest | 0.985 |

| Helmet | 0.977 |

| Non-Helmet | 0.970 |

| All 7 classes | 0.968 (precision 0.956, recall 0.933) |



YOLO11s fine-tuned for 30 epochs on a Colab T4. Runs at 50+ FPS on an RTX 3050 laptop GPU.



\## Known limitations



\- Hi-vis jackets are sometimes missed, because the dataset mostly contains vests.

\- Two people standing shoulder to shoulder can confuse the tracker: IDs change and

&#x20; can cause repeated alerts. A fixed CCTV camera with some distance between workers works better.

\- Validation images come from the same source as the training images, so real-world

&#x20; accuracy will be lower than the numbers above.

\- Zones are drawn for one fixed camera view.



\## Setup



&#x20;   python -m venv .venv

&#x20;   .venv\\Scripts\\activate

&#x20;   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

&#x20;   pip install -r requirements.txt



\## Run



&#x20;   python -m scripts.draw\_zone videos\\your\_video.mp4 0

&#x20;   python -m scripts.run\_safety videos\\your\_video.mp4

&#x20;   uvicorn api.main:app --port 8000



Then open http://127.0.0.1:8000



\## Credits



\- Dataset: "Site Construction Safety" on Roboflow Universe, CC BY 4.0

\- Models: Ultralytics YOLO11 (AGPL-3.0; commercial use needs an Ultralytics enterprise license)

