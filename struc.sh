#!/usr/bin/env bash
set -e

dirs=(
  ".github/workflows"
  "configs"
  "data/raw" "data/processed"
  "dockerfiles"
  "docs/source"
  "models"
  "notebooks"
  "reports/figures"
  "src/call_of_birds_autobird"
  "src/call_of_func/data" "src/call_of_func/dataclasses" "src/call_of_func/train" "src/call_of_func/utils"
  "tests"
)

for d in "${dirs[@]}"; do mkdir -p "$d"; done

