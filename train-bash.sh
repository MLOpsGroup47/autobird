# run_job.sh
#!/usr/bin/env bash

gcloud ai custom-jobs create \
  --region=europe-west1 \
  --display-name=test-run \
  --config=config_cpu.yaml \
  --command="python src/call_of_birds_autobird/train.py" \
  --args=--epochs=10 \
  --args=--batch-size=128 \
  --args=--learning-rate=0.001 \
  --args=--data-path=gs://autobird-data-bucket/dataset/
    
