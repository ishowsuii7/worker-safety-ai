\# Worker Safety AI



Computer-vision safety monitoring for factories and construction sites.

Detects missing helmets and vests, restricted-zone intrusions, falls and

persons lying on the ground in video. Every alert is logged with a snapshot

and shown on a web dashboard.



\## Pipeline



Video -> YOLO11 PPE detection + ByteTrack tracking -> PPE and zone rules

Video -> YOLO11 pose estimation -> fall / person-down rules

Both -> event log (SQLite + snapshots) -> FastAPI -> dashboard



\## Dashboard



!\[Dashboard](docs/dashboard.png)



\## PPE results (validation set, 340 images)



| Class | mAP50 |

|---|---|

| Person | 0.988 |

| Vest | 0.985 |

| Helmet | 0.977 |

| Non-Helmet | 0.970 |

| All 7 classes | 0.968 (precision 0.956, recall 0.933) |



YOLO11s fine-tuned for 30 epochs on a Colab T4. The PPE model alone runs at

50+ FPS on an RTX 3050 laptop GPU; with the pose model added, 25-35 FPS.



!\[Training curves](docs/ppe\_v1/results.png)

!\[Confusion matrix](docs/ppe\_v1/confusion\_matrix\_normalized.png)



\## Fall detection



The pose model finds shoulders and hips. The angle of the shoulders-to-hips line

from vertical gives the posture: upright (under 35 degrees), lying (over 55

degrees with a wide box), or unknown. If the shoulders or hips are not visible,

the system does not guess.



\- FALL: upright, then lying for 2 seconds.

\- PERSON DOWN: lying for 5 seconds, even if the fall itself was not seen.



Tested so far: one stock video of a worker lying on the ground (PERSON DOWN was

raised at 5.0 s, standing workers in the same clip raised nothing) and simulated

body points. The FALL alert (standing to lying) has not yet been tested on real

footage, and the thresholds are untuned.



\## Known limitations



\- Hi-vis jackets are sometimes missed, because the dataset mostly contains vests.

\- Two people standing shoulder to shoulder can confuse the tracker: IDs change and

&#x20; can cause repeated alerts. A fixed CCTV camera with some distance between workers works better.

\- Validation images come from the same source as the training images, so real-world

&#x20; accuracy will be lower than the numbers above.

\- Zones are drawn for one fixed camera view.

\- The PPE model may not detect a person who is lying down; their alert then gets

&#x20; a worker number of 1000 or more.

\- The dashboard counts tracker IDs, not people.



\## Setup



&#x20;   python -m venv .venv

&#x20;   .venv\\Scripts\\activate

&#x20;   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

&#x20;   pip install -r requirements.txt



Models (not stored in the repository):



\- Download ppe\_best.pt from the Releases page and save it as models/ppe\_best.pt

\- Pose model:



&#x20;     python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"

&#x20;     move yolo11s-pose.pt models\\



\## Run



&#x20;   python -m scripts.draw\_zone videos\\your\_video.mp4 0

&#x20;   python -m scripts.run\_safety videos\\your\_video.mp4

&#x20;   uvicorn api.main:app --port 8000



Then open http://127.0.0.1:8000



\## Credits



\- Dataset: "Site Construction Safety" on Roboflow Universe, CC BY 4.0

\- Models: Ultralytics YOLO11 (AGPL-3.0; commercial use needs an Ultralytics enterprise license)

