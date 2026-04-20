# Multi-source fusion for predictive traffic

Hybrid pipeline that combines **CCTV-style computer vision** (vehicle density, simple stall heuristics), **Google Maps Distance Matrix** traffic-aware ETAs (or deterministic mocks without a key), a **fusion layer** to damp false-positive congestion, and an **LSTM** that estimates **standstill jam** probability for **15 / 30 / 45 minutes** ahead.

## Full run (recommended)

From the project folder, one command does venv setup, `pip install`, LSTM training if weights are missing, copies `.env` from `.env.example` if needed, picks a free port, and starts the API + dashboard:

```bash
cd "/path/to/Multi-Source Fusion for Predictive Traffic"
chmod +x scripts/run_full.sh
./scripts/run_full.sh
```

Override the port: `PORT=9000 ./scripts/run_full.sh`.

Open the printed URL (e.g. `http://127.0.0.1:8000`). Click **Refresh all zones** to run YOLO, Maps (or mocks), fusion, and LSTM forecasts.

## Manual quick start

```bash
cd "/path/to/Multi-Source Fusion for Predictive Traffic"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_lstm.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

Copy `.env.example` to `.env` and set `GOOGLE_MAPS_API_KEY` for live traffic. Without a key, route traffic is **mocked** so the stack still runs offline.

## API highlights

- `POST /api/refresh_all` — refresh every zone
- `POST /api/zones/{zone_id}/refresh` — single zone (synthetic frame if no upload)
- `POST /api/zones/{zone_id}/upload_frame` — multipart image for real CCTV frames
- `GET /api/zones/{zone_id}/thumbnail` — latest JPEG thumbnail from vision
- `GET /api/zones/{zone_id}/state` — last cached payload

## Notes

- **YOLOv8** (`ultralytics`) downloads weights on first use (`yolov8n.pt` by default).
- **LSTM** weights are produced by `scripts/train_lstm.py` (synthetic supervision). Replace with your own labels when you have historical fused features and jam outcomes.
