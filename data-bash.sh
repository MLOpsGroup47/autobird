# run_job.sh
#!/usr/bin/env bash

gcloud ai custom-jobs create \
  --region=europe-west1 \
  --display-name=test-run \
  --config=config_cpu.yaml \
  --command="python src/call_of_birds_autobird/data.py" \
  --args="--raw-dir=gs://my-bucket/raw" \
  --args="--processed-dir=gs://my-bucket/processed"

    