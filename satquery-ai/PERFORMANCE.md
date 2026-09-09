# Performance

SatQuery AI loads specialist models once per process through the registry and uses a configurable inference queue. The production default is one Uvicorn worker and `MAX_CONCURRENT_INFERENCE=1` to avoid duplicating large models or exhausting GPU memory.

Measure startup/model load, upload validation, inference, report generation, resident memory, and GPU allocation in the target deployment. Structured logs include request and model timing fields. Compare CPU and GPU behavior with representative GeoTIFF and PNG inputs before increasing concurrency.

Do not enable extra workers for GPU inference without measuring total model memory per worker. Persisting `MODEL_CACHE_DIR` avoids repeated downloads after restarts. Upload/report TTL cleanup limits disk growth. Model weights and datasets are external deployment inputs and are not included in the image.

A clean benchmark should report p50/p95 upload and inference latency, model load time, peak RAM/GPU memory, queue wait time, and failure rate for each task. Full measurements require target hardware and model weights and are not inferred from unit tests.
