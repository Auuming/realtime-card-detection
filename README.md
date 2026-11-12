# TableLens — Magic: The Gathering tabletop scanner

Video/webcam → live card borders → click one card → identify it → show its image, name, mana cost, type, and rules text. Video and borders continue updating while details load.

## What is ready

- CollectorVision pretrained corner model and embedding model are installed. Its MTG/Scryfall catalog contains **114,829 reference faces** at setup time. Recognition is no longer limited to the six starter cards.
- Click-only recognition, upright/upside-down search, ambiguity rejection, Scryfall detail/image downloads, local caching, and the side panel are implemented and checked.
- A contour detector supplies borders now. It works best with separated, fully visible cards and clear edges.
- Trained YOLO weights can be connected with `--model`. **YOLO has not been trained.** Training/data-generation scripts are included for later use.
- Independent real gameplay evaluation is still needed. Catalog coverage does not guarantee identification of every printing, language, artwork, new release, foil, tiny card, or occluded card.

## Run the app

```powershell
.\setup.ps1  # First run from the cloned repository folder
.\run.ps1
```

Setup creates the local `.venv` and downloads the pretrained models and MTG catalog. The launcher uses that `.venv` for subsequent runs. Enter `0` and click **Camera / stream** for the default webcam; try `1` for another camera. Use **Open video** for MP4/AVI/MOV/MKV. You can enter direct RTSP/HTTP video URLs; YouTube webpage URLs are not direct feeds. Codec and remote-stream compatibility depend on OpenCV and the source.

Purple borders mark detected regions. Click inside one to load its details. Details remain visible until another click or source change. **Pause / resume** freezes playback for inspection.

The selected video crop appears immediately. Local recognition displays the name without waiting for Scryfall; metadata and the reference image load in a separate background worker. The panel labels the video crop and replaces it with the reference image when ready. Cached images are reused, failed requests have a 60-second retry cooldown, and **Retry image / details** explicitly retries sooner. Interactive requests use 3-second connect and 6-second read timeouts; those are per-request/socket timeouts, not a total loading deadline. New clicks and seeks discard the previous selection's display requests.

Recognition runs locally after setup. The first lookup of an uncached card needs internet for its rules and reference image. Later lookups reuse `data/details/`. Download failure preserves the recognized name with a missing-details message. Cached rules do not automatically refresh; remove the relevant cached JSON/image files for an explicit refresh.

## Reinstall / another computer

Use 64-bit Python with Tk support. This installation was checked with Python 3.13.9, Windows, and CPU ONNX Runtime.

```powershell
.\setup.ps1
.\run.ps1
```

Manual equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-collectorvision.txt
.\.venv\Scripts\python.exe recognition.py --prepare
.\.venv\Scripts\python.exe app.py
```

CollectorVision is pinned to commit `8b295ab80a997766afc00c640a616c1f2716df21`. The archive URL avoids requiring Git during setup. Preparation installs a compatible MTG/Scryfall snapshot and resolves its required pretrained embedding model. Run `recognition.py --prepare` again to explicitly check for catalog updates. Normal app startup uses the locally installed snapshot with `offline=True`.

Models, catalog, Hugging Face cache, and details default to this project's `data/`. Keep it for offline reuse. Recreate `.venv` on another machine. The upstream source archive and example photographs are retained in `data/vendor/` for provenance/testing.

CPU ONNX inference is used in this environment. The detected GPU is a GTX 1650 with 4096 MiB VRAM. Its CUDA training setup has not been validated. Existing Anaconda PyTorch reported conflicting OpenMP libraries; use the project environment for additional setup.

## Architecture

```text
Video / webcam / direct stream
    → contour detector, or trained YOLO with --model
    → live borders
    → click → capture selected region
    → CollectorVision neural corners + native 448×448 perspective warp
    → pretrained embeddings for upright and upside-down orientations
    → MTG catalog search → accept/reject
    → Scryfall ID and matched face → cached/on-demand details → side panel
```

Detection is submitted at most 20 times per second, bounded by actual compute time. It never runs catalog recognition. Selected-card recognition has its own worker. A newer click supersedes the old request; changing source clears pending details. The selection uses the exact frame that produced its border. Borders may still lag on fast movement because no temporal tracker is implemented. Video FPS depends on decoding and UI overhead. A blocked OpenCV network-stream read can still delay the UI.

The corner model refines the clicked region. When it returns no valid card, the selected polygon supplies the fallback warp. Search checks both 0° and 180° orientations. Same-Oracle reprints are grouped by card identity. The `0.78` cosine similarity cutoff and `0.025` minimum score gap are temporarily commented out. The app shows the highest-scoring candidate when the search returns one, along with its cosine similarity as a percentage and a raw score.

The ORB baseline remains available:

```powershell
.\run.ps1 --recognizer orb
```

Its six-image gallery is managed by `catalog.py` and is separate from CollectorVision's embedding catalog. You do not need to download every full-resolution card image to use CollectorVision.

## Connect YOLO after training

Install the optional dependencies and a CUDA-enabled PyTorch build matching your driver using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Verify CUDA instead of assuming the default wheel supports the GPU.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
.\run.ps1 --model runs/card_detector/weights/best.pt
```

Use one YOLO class: `card`, ID `0`. The detector localizes cards; CollectorVision identifies them. There is no separate detector class for each MTG identity. Supplying weights only loads them; it does not start training.

Later, train your labeled real dataset:

```powershell
.\.venv\Scripts\python.exe train.py train --data data/real/dataset.yaml --device 0 --epochs 60 --batch 4
```

Reduce batch size to 2 or 1 if memory is insufficient. Training starts from `yolo11n.pt` at image size 960. **This command was not run.** YOLO boxes are axis-aligned; selected-card recognition handles corner refinement. Occlusion still needs evaluation.

Suggested dataset structure:

```text
data/real/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
  dataset.yaml
```

```yaml
path: E:/Chula/2110433 computer vision/project/data/real
train: images/train
val: images/val
test: images/test
names:
  0: card
```

Optional synthetic warm-start dataset:

```powershell
.\.venv\Scripts\python.exe catalog.py --download-bulk --limit 1000
.\.venv\Scripts\python.exe train.py generate --count 2000 --backgrounds path/to/tabletop-photos
```

Bulk imports accept JSON arrays/JSONL, optionally gzip, and reuse downloaded images. Synthetic generation varies perspective, rotation, backgrounds, brightness, blur/noise, includes negative frames, and splits Oracle identities between training and validation. Real labeled tabletop footage is still needed. Separate recording sessions/playmats between train/validation/test instead of randomly splitting neighboring frames.

## Checks and results

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe verify_collector.py
.\.venv\Scripts\python.exe verify_app.py
.\.venv\Scripts\python.exe verify_offline.py
.\.venv\Scripts\python.exe verify_image_loading.py
```

- Unit checks: multi-card localization, tapped/portrait warps, blank rejection, double-face details, and network failure handling.
- CollectorVision smoke checks: six catalog cards in both orientations, three upstream photographs (ordinary, hidden name, skewed), and blank/noise rejection. All 15 identity cases and 2 rejection cases passed. Machine-readable results, timings, model identity, and catalog version are in `data/collector-verification.json`.
- GUI check: no identification before a click, one clicked-card request, detection continues during deliberately delayed recognition, and image/rules appear.
- Offline check: HTTP calls are blocked while cached recognition, image, and rules are loaded.

Reference renders and upstream photographs are smoke fixtures, not an independent gameplay test set. Webcam hardware, remote streams, and trained YOLO inference have not been exercised. No full-catalog accuracy/FPS claim is made. In one smoke run, warm recognition including both orientations took about 0.55–0.61 seconds; first-time detail/image downloads took several seconds. Those observations are not latency guarantees.

For actual evaluation, measure detector precision/recall and mAP50–95, border latency/FPS, clicked-card Oracle accuracy, unknown false acceptance, and click-to-detail latency. Include sleeves, glare, hands, overlap, tapped cards, different cameras, and card pixel heights. Select thresholds on validation data and leave test sessions untouched. Exact-printing accuracy is a separate task.

## Sources and licenses

- [CollectorVision](https://github.com/HanClinto/CollectorVision): pretrained Cornelius/Milo models and MTG catalog integration, AGPL-3.0-or-later. Installed as a dependency; models were downloaded, not trained here. Review its license for distribution.
- [Catalog v2 documentation](https://github.com/HanClinto/CollectorVision/blob/main/docs/catalog-v2-migration.md): source IDs, faces, Oracle grouping, and compatible offline snapshots.
- [Scryfall](https://scryfall.com/docs/api) and [access guidance](https://scryfall.com/docs/faqs/i-m-having-trouble-accessing-the-scryfall-api-or-i-m-blocked-17): metadata/images and cached, restrained API use. Artwork belongs to its respective rights holders.
- [Ultralytics training](https://docs.ultralytics.com/modes/train/): optional detector training/inference; review licensing when distributing.
- Related work: [Scryglass](https://github.com/KJBurnett/scryglass), [visions-datasets](https://github.com/LauriHursti/visions-datasets). Neither is required at runtime.

Main files: `app.py` (UI/workers), `recognition.py` (CollectorVision/details), `vision.py` (geometry/ORB), `catalog.py` (local image gallery), `train.py` (detector training/data), `setup.ps1` and `run.ps1` (setup/launch), `verify_*.py` and `test_*.py` (checks). `data/` is ignored runtime data; `.venv` is the local Python environment.
