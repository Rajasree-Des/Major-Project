# ThermaScope — Local Web Application

**Thermal Satellite Intelligence & Super-Resolution Studio**

## Prerequisites

- Python 3.10+ (`py -m pip install -r requirements.txt`)
- Node.js 18+ for the frontend
- `dataset/processed_qa/` and `pretrained/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth`

## Launch (Windows — two terminals)

**Terminal 1 — API**

```powershell
cd "c:\Users\HP\Downloads\ISRO project\Major-Project"
py -m backend
```

**Terminal 2 — UI**

```powershell
cd "c:\Users\HP\Downloads\ISRO project\Major-Project\frontend"
npm install
npm run dev
```

Open http://127.0.0.1:5173/ — Vite proxies `/api` to port 8000.

## Demo path (live viva)

1. **Command Center** — confirm green API indicator in top bar.
2. **Processing Studio** — Kerala validation scene, patch `000096` (if available), **Run pipeline**.
3. Inspect stage tabs (LR, bicubic, SwinIR, refined, compare slider).
4. **Colorization Studio** — methods A/B/C; download Kelvin `.npy` link.
5. **Results Lab** — locked test Deccan/Assam charts from frozen `results.json`.
6. **Research & About** — limitations.

## API endpoints

| Method | Path |
|--------|------|
| GET | `/api/health` |
| GET | `/api/overview` |
| GET | `/api/scenes`, `/api/scenes/{id}/patches`, patch metadata, previews |
| GET | `/api/results/frozen?experiment=multiscene\|physics&role=validation\|test_locked` |
| POST | `/api/jobs` |
| GET | `/api/jobs/{id}`, `/api/jobs/{id}/logs`, `/api/jobs/{id}/artifacts` |
| GET | `/api/history`, `/api/runs/{id}` |
| GET | `/api/artifacts/{root_key}/{path}` |

## Validation

```powershell
py -m pytest tests/test_api_backend.py -q -m "not slow"
py -m pytest tests/test_api_backend.py::test_single_patch_job_cpu -q
cd frontend; npm run build
```

## Known limitations

- One inference job at a time (HTTP 409 if busy).
- No job cancel — wait for completion or restart API.
- Validation-sample mode skips full stage PNG grid.
- Scene Explorer previews are **reference** data, not SR output.
- UI run metrics ≠ locked multiscene evaluation JSON.

Outputs: `outputs/ui_runs/<run_id>/` (manifest, stages/, colorization PNGs).

See `outputs/ui_integration_report_v1.txt` for last test run.
