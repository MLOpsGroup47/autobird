# call_of_birds_autobird

mlops project - automatic bird call classifier

## Project Description

**Overall goal of the project** 

This project aims to build a production ready machine learning model for classifying bird species from 5 seconds of their call using a transformer based model. Bird vocalizations give critical insights for biodiversity assessment and ecological monitoring, but manual classification is very time consuming and requires expert knowledge. By combining state of the art machine learning techniques with MLOps best practices our goal is to make a robust, scalable, maintainable, and reproducible solution for classifying bird species from their calls that can operate reliably in production settings. 

**Data** 

The dataset consists of 2161 bird "voices" recorded recorded before 2023, which includes both calls, songs and other sounds made by birds. The dataset can be found from kaggle at https://www.kaggle.com/code/dima806/bird-species-by-sound-detection/input.

114 different bird sepcies are present, with each species having 3-30 audio recordings assigned. The audio recordings range from a couple of seconds in length to almost half an hour. 


**Models** 

Transformer-based architecture. 


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
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
