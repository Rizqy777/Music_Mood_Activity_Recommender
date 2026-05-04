$env:MAX_ROWS_PER_DATASET="200"
$env:PRODUCER_BATCH_SIZE="25"
$env:PRODUCER_BATCH_DELAY_SECONDS="0.5"

.\.venv\Scripts\python.exe scripts\run_pipeline.py --monitor-terminals