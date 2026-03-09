# Flux Image Service

Local image generation API powered by **Flux.1 Schnell 12B** running on an H100 GPU with FP8 quantization.

## Features

- **Single image generation** — async (returns job IDs) or sync (waits for result), up to 4 images per request
- **Batch/dataset generation** — submit up to 50K prompts via API or CSV/JSON file upload
- **Priority queue** — real-time requests always jump ahead of batch jobs
- **SSE progress streaming** — monitor batch progress in real-time
- **Crash recovery** — pending jobs auto-resume on server restart (persisted in SQLite)
- **Multiple formats** — PNG (with metadata), JPEG, WebP
- **Custom resolutions** — any multiple of 8, up to 2048px

## Prerequisites

- **Python** 3.10+
- **NVIDIA GPU** with CUDA 12.x (tested on H100 20GB)
- **NVIDIA drivers** 535+ with CUDA toolkit
- **System RAM** 32 GB recommended (16 GB minimum)
- **Disk** ~12 GB for model weights + storage for generated images

## Setup

```bash
# 1. Clone / navigate to the project
cd flux-image-service

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (includes torch, diffusers, fastapi, etc.)
pip install -e ".[dev]"

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env if needed (model ID, storage path, etc.)
```

## Running the Server

```bash
# Activate venv (if not already)
source .venv/bin/activate

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development (not recommended for GPU — reloads unload the model)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The first startup will:
1. Download the Flux.1 Schnell model from HuggingFace (~12 GB)
2. Compile CUDA kernels via `torch.compile` (~30-60s one-time cost per restart)
3. Run a warm-up inference

Once you see `Flux Image Service ready`, the server is accepting requests.

## Running Tests

```bash
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_api_generate.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Docker

```bash
# Build
docker build -t flux-image-service .

# Run (requires nvidia-container-toolkit)
docker run --gpus all -p 8000:8000 -v ./output:/app/output flux-image-service

# Run with custom env
docker run --gpus all -p 8000:8000 \
  -e MODEL_ID=black-forest-labs/FLUX.1-schnell \
  -e STORAGE_DIR=/app/output \
  -v ./output:/app/output \
  flux-image-service
```

## Linting

```bash
# Check
ruff check app/ tests/

# Auto-fix
ruff check app/ tests/ --fix

# Format
ruff format app/ tests/
```

---

## API Reference

### `POST /generate` — Async Image Generation

Submit a generation request (1–4 images). Returns immediately with job IDs.

**Request:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a majestic mountain landscape at sunset",
    "width": 1024,
    "height": 1024,
    "num_images": 2,
    "seed": 42
  }'
```

**Response** `200 OK`:
```json
{
  "job_ids": ["f47ac10b-58cc-4372-a567-0e02b2c3d479", "a1b2c3d4-5678-9abc-def0-1234567890ab"],
  "status": "pending",
  "image_urls": []
}
```

**Validation error** `422`:
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "width"],
      "msg": "Value error, width must be a multiple of 8"
    }
  ]
}
```

All request parameters for `ImageRequest`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | *required* | Text prompt (1–2000 chars) |
| `negative_prompt` | string \| null | null | Negative prompt |
| `width` | int | 1024 | Image width (64–2048, multiple of 8) |
| `height` | int | 1024 | Image height (64–2048, multiple of 8) |
| `num_steps` | int | 4 | Inference steps (1–50) |
| `guidance_scale` | float | 0.0 | CFG scale (Schnell doesn't use CFG) |
| `seed` | int \| null | null | Random seed (0–4294967295) |
| `num_images` | int | 1 | Number of images (1–4) |
| `format` | string | "png" | Output format: `png`, `jpeg`, `webp` |

---

### `GET /generate/stream` — SSE Progress Stream for Single Generate

Stream status updates for one or more job IDs returned by `/generate` until all are terminal.

**Request:**
```bash
curl -N "http://localhost:8000/generate/stream?job_id=f47ac10b-58cc-4372-a567-0e02b2c3d479&job_id=a1b2c3d4-5678-9abc-def0-1234567890ab"
```

**Events:**
- `progress` — emitted periodically with per-job status and aggregate counters
- `done` — emitted once when all jobs are in `completed`, `failed`, or `cancelled`

**Example event payload:**
```json
{
  "total": 2,
  "completed": 1,
  "failed": 0,
  "cancelled": 0,
  "pending": 1,
  "jobs": [
    {
      "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "status": "completed",
      "image_url": "/jobs/f47ac10b-58cc-4372-a567-0e02b2c3d479/image"
    },
    {
      "id": "a1b2c3d4-5678-9abc-def0-1234567890ab",
      "status": "processing",
      "image_url": null
    }
  ]
}
```

---

### `POST /generate/sync` — Synchronous Generation

Same request body as `/generate`. Blocks until all images are generated (up to 60s timeout).

**Request:**
```bash
curl -X POST http://localhost:8000/generate/sync \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cute cat wearing a hat", "num_images": 1}' \
  --max-time 60
```

**Response** `200 OK`:
```json
{
  "job_ids": ["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
  "status": "completed",
  "image_urls": ["/jobs/f47ac10b-58cc-4372-a567-0e02b2c3d479/image"]
}
```

**Timeout** `408`:
```json
{"detail": "Generation timed out"}
```

---

### `GET /jobs/{job_id}/image` — Serve Generated Image

Returns the image file for a completed job.

**Request:**
```bash
curl http://localhost:8000/jobs/f47ac10b-58cc-4372-a567-0e02b2c3d479/image --output image.png
```

**Response:** Binary image file with appropriate `Content-Type` (`image/png`, `image/jpeg`, or `image/webp`).

**Not found** `404`:
```json
{"detail": "Image not available"}
```

---

### `GET /jobs/{job_id}` — Get Job Details

**Request:**
```bash
curl http://localhost:8000/jobs/f47ac10b-58cc-4372-a567-0e02b2c3d479
```

**Response** `200 OK`:
```json
{
  "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "prompt": "a majestic mountain landscape at sunset",
  "negative_prompt": null,
  "width": 1024,
  "height": 1024,
  "num_steps": 4,
  "guidance_scale": 0.0,
  "seed": 42,
  "status": "completed",
  "priority": 1,
  "format": "png",
  "file_path": "output/images/2026-03-08/f47ac10b-58cc-4372-a567-0e02b2c3d479.png",
  "image_url": "/jobs/f47ac10b-58cc-4372-a567-0e02b2c3d479/image",
  "error_message": null,
  "batch_id": null,
  "created_at": "2026-03-08T12:00:00+00:00",
  "started_at": "2026-03-08T12:00:01+00:00",
  "completed_at": "2026-03-08T12:00:03+00:00"
}
```

---

### `GET /jobs` — List Jobs

**Request:**
```bash
# List all jobs (paginated)
curl "http://localhost:8000/jobs?page=1&page_size=20"

# Filter by status
curl "http://localhost:8000/jobs?status=completed"

# Filter by batch
curl "http://localhost:8000/jobs?batch_id=batch-uuid-here"
```

**Response** `200 OK`:
```json
{
  "jobs": [
    {
      "id": "f47ac10b-...",
      "prompt": "a mountain",
      "status": "completed",
      "width": 1024,
      "height": 1024,
      "image_url": "/jobs/f47ac10b-.../image",
      "...": "..."
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 20
}
```

---

### `DELETE /jobs/{job_id}` — Cancel a Job

Cancels a pending/processing job. Already completed/failed jobs are returned as-is.

**Request:**
```bash
curl -X DELETE http://localhost:8000/jobs/f47ac10b-58cc-4372-a567-0e02b2c3d479
```

**Response** `200 OK`:
```json
{
  "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "cancelled",
  "...": "..."
}
```

---

### `POST /batch` — Create Batch Job

Submit 1–50,000 prompts as a batch. All jobs get BATCH priority (lower than real-time).

**Request:**
```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "name": "training-dataset-v1",
    "prompts": [
      {"prompt": "a red sports car", "width": 512, "height": 512},
      {"prompt": "a blue ocean wave", "width": 1024, "height": 768},
      {"prompt": "a forest path in autumn", "seed": 123}
    ]
  }'
```

**Response** `200 OK`:
```json
{
  "batch_id": "b1c2d3e4-5678-9abc-def0-1234567890ab",
  "name": "training-dataset-v1",
  "total": 3,
  "completed": 0,
  "failed": 0,
  "cancelled": 0,
  "pending": 3,
  "status": "pending",
  "estimated_remaining_seconds": null,
  "created_at": "2026-03-08T12:00:00+00:00",
  "completed_at": null
}
```

---

### `POST /batch/from-file` — Create Batch from File Upload

Upload a CSV or JSON file of prompts.

**CSV upload:**
```bash
curl -X POST http://localhost:8000/batch/from-file \
  -F "file=@prompts.csv" \
  -F "name=my-dataset"
```

CSV format (only `prompt` column is required):
```csv
prompt,width,height,num_steps,seed,format
a red car,512,512,4,42,png
a blue house,1024,1024,,,
a green tree,,,,,
```

**JSON upload:**
```bash
curl -X POST http://localhost:8000/batch/from-file \
  -F "file=@prompts.json" \
  -F "name=my-dataset"
```

JSON format:
```json
[
  {"prompt": "a red car", "width": 512, "height": 512, "seed": 42},
  {"prompt": "a blue house"},
  {"prompt": "a green tree", "format": "jpeg"}
]
```

**Response:** Same as `POST /batch`.

---

### `GET /batch/{batch_id}` — Get Batch Progress

**Request:**
```bash
curl http://localhost:8000/batch/b1c2d3e4-5678-9abc-def0-1234567890ab
```

**Response** `200 OK`:
```json
{
  "batch_id": "b1c2d3e4-5678-9abc-def0-1234567890ab",
  "name": "training-dataset-v1",
  "total": 1000,
  "completed": 342,
  "failed": 2,
  "cancelled": 0,
  "pending": 656,
  "status": "processing",
  "estimated_remaining_seconds": 1968.0,
  "created_at": "2026-03-08T12:00:00+00:00",
  "completed_at": null
}
```

---

### `GET /batch/{batch_id}/stream` — SSE Progress Stream

Real-time Server-Sent Events stream of batch progress. Updates every 2 seconds.

**Request:**
```bash
curl -N http://localhost:8000/batch/b1c2d3e4-5678-9abc-def0-1234567890ab/stream
```

**Response** (event stream):
```
event: progress
data: {"batch_id":"b1c2d3e4-...","total":1000,"completed":342,"failed":2,"pending":656,"status":"processing","estimated_remaining_seconds":1968.0}

event: progress
data: {"batch_id":"b1c2d3e4-...","total":1000,"completed":343,"failed":2,"pending":655,"status":"processing","estimated_remaining_seconds":1965.0}

event: done
data: {"batch_id":"b1c2d3e4-...","total":1000,"completed":998,"failed":2,"pending":0,"status":"failed"}
```

---

### `DELETE /batch/{batch_id}` — Cancel Batch

Cancels all remaining pending jobs in the batch.

**Request:**
```bash
curl -X DELETE http://localhost:8000/batch/b1c2d3e4-5678-9abc-def0-1234567890ab
```

**Response** `200 OK`:
```json
{
  "batch_id": "b1c2d3e4-5678-9abc-def0-1234567890ab",
  "total": 1000,
  "completed": 342,
  "failed": 2,
  "cancelled": 656,
  "pending": 0,
  "status": "cancelled",
  "...": "..."
}
```

---

### `POST /batch/{batch_id}/retry-failed` — Retry Failed Jobs

Re-enqueue only the failed jobs in a batch.

**Request:**
```bash
curl -X POST http://localhost:8000/batch/b1c2d3e4-5678-9abc-def0-1234567890ab/retry-failed
```

**Response** `200 OK`:
```json
{
  "retried": 2,
  "job_ids": ["job-uuid-1", "job-uuid-2"]
}
```

---

### `GET /health` — Health Check

**Request:**
```bash
curl http://localhost:8000/health
```

**Response** `200 OK`:
```json
{
  "status": "ok",
  "model_loaded": true,
  "vram": {
    "available": true,
    "allocated_gb": 12.34,
    "reserved_gb": 14.50,
    "max_allocated_gb": 16.78,
    "total_gb": 20.00,
    "device_name": "NVIDIA H100"
  },
  "worker": {
    "queue_depth": 5,
    "jobs_processed": 142,
    "total_inference_time_s": 398.50,
    "avg_inference_time_s": 2.81
  },
  "uptime_seconds": 3600.0
}
```

---

## Architecture

```
Client → FastAPI → asyncio.PriorityQueue → GPU Worker → Flux Engine
                       ↓                        ↓
                    SQLite                   Local FS
                  (job state)              (images)
```

- **Single GPU worker** — processes one image at a time (VRAM constraint)
- **Priority queue** — REALTIME(1) always before BATCH(5)
- **OOM recovery** — catches GPU OOM, clears cache, continues processing
- **Crash recovery** — on startup, all PENDING/PROCESSING jobs in SQLite are automatically re-enqueued

### Queue Persistence

The in-memory `asyncio.PriorityQueue` is **lost on server restart**. However, **no work is lost** because:

1. All jobs are persisted to SQLite the moment they're created (before being enqueued)
2. On startup, `resume_pending_jobs()` scans for any jobs in `PENDING` or `PROCESSING` state and re-enqueues them
3. Completed jobs and their images are already on disk

This means a restart simply rebuilds the queue from the database. The only effect is a brief pause while the model reloads.

## Configuration

All settings can be overridden via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `black-forest-labs/FLUX.1-schnell` | HuggingFace model ID |
| `DEVICE` | `cuda` | Torch device |
| `DTYPE` | `float8_e4m3fn` | Model precision |
| `ENABLE_TORCH_COMPILE` | `true` | Enable torch.compile optimization |
| `DEFAULT_NUM_STEPS` | `4` | Default inference steps |
| `DEFAULT_GUIDANCE_SCALE` | `0.0` | Default CFG scale |
| `MAX_RESOLUTION` | `2048` | Maximum allowed width/height |
| `MAX_IMMEDIATE_IMAGES` | `4` | Max images per /generate request |
| `STORAGE_DIR` | `./output` | Where images and DB are stored |
| `MAX_QUEUE_SIZE` | `10000` | Maximum queue capacity |
| `SYNC_TIMEOUT` | `60.0` | Timeout for /generate/sync (seconds) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

## Performance (H100 20GB, FP8)

| Resolution | Time per image | 10K dataset |
|-----------|---------------|-------------|
| 512×512 | ~0.8–1.2s | ~2–3 hours |
| 1024×1024 | ~2–4s | ~6–10 hours |
| 1536×640 | ~2–3s | ~5–8 hours |
