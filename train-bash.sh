# run_job.sh
#!/usr/bin/env bash

gcloud ai custom-jobs create \
  --region=europe-west1 \
  --display-name=test-run-parallel \
  --config=config_gpu.yaml \
  --args="paths.processed_dir=gs://birdcage-bucket/data/processed" \
  --args="paths.x_train=gs://birdcage-bucket/data/processed/train_x.pt" \
  --args="paths.x_val=gs://birdcage-bucket/data/processed/val_x.pt" \
  --args="paths.y_train=gs://birdcage-bucket/data/processed/train_y.pt" \
  --args="paths.y_val=gs://birdcage-bucket/data/processed/val_y.pt" \
  --args="train.hp.epochs=100" \
    
