# call_of_birds_autobird

mlops project - automatic bird call classifier

## Project Description

**Overall goal of the project** 

This project aims to build a production ready machine learning model for classifying bird species from 5 seconds of their call using a transformer based model. Bird vocalizations give critical insights for biodiversity assessment and ecological monitoring, but manual classification is very time consuming and requires expert knowledge. By combining state of the art machine learning techniques with MLOps best practices our goal is to make a robust, scalable, maintainable, and reproducible solution for classifying bird species from their calls that can operate reliably in production settings. 

**Data**

- Source: 2,161 bird vocalizations (calls, songs, and other sounds) recorded before 2023. Original dataset is hosted on Kaggle: https://www.kaggle.com/code/dima806/bird-species-by-sound-detection/input.
- Classes: 114 species are included, with 3–30 recordings per species. Durations range from a few seconds up to ~30 minutes.
- Layout: Audio and any metadata placed under [data/raw](data/raw). Preprocessed outputs (e.g., clipped segments or features) are written to [data/processed](data/processed).
- Formats: Common audio formats MP3 are provided and kept in [data/raw](data/raw).

Preprocessing is orchestrated via the dataset utilities in [src/call_of_birds_autobird/data.py](src/call_of_birds_autobird/data.py). Run preprocessing to generate model-ready artifacts in [data/processed](data/processed):

	Using `invoke` (recommended):

	```bash
	uv run invoke preprocess_data
	```

	Or call the module directly:

	```bash
	uv run src/call_of_birds_autobird/data.py data/raw data/processed
	```

The preprocessing step is designed to be extended to: segment long recordings into 5-second clips (the model’s target window), normalize audio, and optionally compute spectrogram or log-mel features. Exact feature choices can be configured or implemented within [src/call_of_birds_autobird/data.py](src/call_of_birds_autobird/data.py).



**Models** 

The model is based on a transformer based architecture for audio classification and uses a pretrained Wav2Vec2 backbone. It takes raw audio recordings as input, which are resampled to 16 kHz and split into fixed length segments. A classification layer is trained on top of the pretrained model to predict bird species from their calls. The pretrained model has learned general patterns in sound from large collections of audio recordings, which helps the system perform well even with limited labeled bird data. This makes the approach effective, scalable, and suitable for real world use.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
| 	├──
|   ├── .gitkeep
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
|   ├── figures/
|   ├── README.md
│   └── report.py
├── src/                      # Source code
│   ├── call_of_birds_autobird/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
│	└──	call_of_func/
│		├── data/
│		├── dataclasses/
│		├── train/
│		└──utils/
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
