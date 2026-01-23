# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [x] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [x] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [x] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [x] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [x] Write some documentation for your application (M32)
* [x] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer: 
Group 47

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s224473, s224022, s224031, s214776

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We have predominantly implemented the recommended tools shown in the course exercises to solve the MLops part of the assignment (i.e to check off most of the boxes in the section above). As far as packages/libraries not introduced in the course, we have used the torchaudio library in out data processing pipeline. This was used, among other things, to transform raw audio signals into log-mel spectrograms and apply audio data augmentation during training.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used uv together with pyproject.toml to manage dependencies in our project. All project dependencies are declared in pyproject.toml, while exact package versions are locked in uv.lock to ensure full reproducibility. Whenever a new dependency was to added, the command: "uv add <package-name>" was used, which updates the pyproject.py and uv.lock files. Assuming a new member is working on a branch with an up-to-date pyproject.toml and uv.lock file, the command: "uv sync", downloads all necessary dependcies ensures they are working with an exact copy of the environment, and ensures reproducibility. 


### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We used cookiecutter as a template for our repository, specifically we initiallized the project with the following command and template:

uvx cookiecutter \
 https://github.com/SkafteNicki/mlops_template 

We have filed out the following: 
- src: this includes 2 sub folders, "call_of_func" for modular/helper functions for both train, dataclasses, data and utils. the other subfolder "call_of_birds_autobird" for main functions eg. train.py, evaluate.py, api.py, model.py, visualize.py etc
- configs: hydra config files with parameters etc. Subfolders for    data            hyperparams     optimizer       pathing         prof            scheduler       wandb
- reports: subfolders include  eval_metrics    figures         report.py       torch_prof
- notesbooks: data visualizations (raw and processed), model tesing
- models: subfolder "checkpoints" stores and metrics and model for last.pt and best.pt
- docker: dockerfiles

We have added the following folders: 
- wandb: experiment logging
- outputs: displays how are config files are stuctured for dataprocessing and training for each run
- data: data storage, two subfolders voice_of_birds (raw) and processed (processed data)
- api: to configure and develop APIs

We have removed the following folders: 
- docs
- .devcontainer


### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We used ruff for linting and formatting, configured with a maximum line length of 120 characters and enabled the I (import sorting) and D (docstring) rule sets. For docstrings, we followed the Google style convention, while explicitly ignoring selected documentation rules (D100–D107) to avoid enforcing docstrings everywhere and instead focus on meaningful, higher-level documentation. We chose these settings because they ensure uniformity in the code but aren't too restrictive. 

Additionally, we implemented mypy for type checking to catch type-related errors early and to make function interfaces and data structures more explicit.

Both of these concepts are important for coding projects (in particular group projects) as they ensure uniformity/reproducibility in the code which makes the code more readable both for co-developers and for non-developers overlooking the code.  




## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:
8 individual unit tests targeting API, data, model and training. We are primarily testing function level, meaning we are testing the functions to be used in larger procedures. For the API we test reading the root and using our classify endpoint which performs inference from a POST event. Data tests include confirming config class and its attributes along with a couple of tests of helper functions for the data processing. We test the model by forwading dummy data and thus confirming the model from the output received from the forward pass. Lastly we test the training by simulating 1 iteration of our training loop alongside testing a relevant helper function. 

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- question 8 fill here ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We implemented both branches and pull requests throughout our project work. We had our main branch and a branch for each group member. When developing new src or implementing changes in repository, all members worked primarily on their own branch to avoid conflicts. When the changes were working on the given branch X (including passing all tests), and making sure that all of the newest commits in main were merged into X, we then open PRs from the branch X and merged the changes into main. We used a mix of git commands directly from the terminal (git add, git commit -m, git fetch, git merge etc) and using the Github desktop app. 



### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

Yes, we implemented DVC to manage our data, to version and track datasets stored outside the Git repository. DVC was configured with a remote gcloud storage bucket, which allowed us to keep large data files outside our of Git, while maintaining a link between code, configuration, and data version used in experiments. 

It helped us improve the reproducibility of our project. Each of our git commmands is associated with a specific version of the data in the DVC, making it easy to determine which data was used for train, validation and testing. ADDD MORE

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

--- question 11 fill here ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used hydra and typer, but hydra was mainly used, typer is optional. In the configs/ folder contain subfolder with configuration for training hyperparameters, distributed training, profiling, paths configurations, and preprocessing configuratiosns. 
#### To run data preprocessing, use the following commands: 
```bash
uvr data processing.sr=16000
uvr data data.train_split=0.8
uvr data pathing.paths.raw_dir=data/voice_of_birds
```
Optional commands and their defaults can be seen below: 
```yaml
preprocessing:
  sr: 16000
  clip_sec: 5.0
  n_fft: 1024
  hop_length: 512
  n_mels: 64
  fq_min: 20
  fq_max: 8000
  min_rms: 0.005
  min_mel_std: 0.10
  min_samples: 50

data:
  train_split: 0.8
  test_split: 0.1
  seed: 4
  clip_sec: 5.0
  stride_sec: 2.5
  pad_last: true
```
#### To run training, use the following commands:
```bash
uvr train train.hp.lr=0.0003
uvr train train.optim._target_=torch.optim.Adam
uvr train train.slr._target_=torch.optim.lr_scheduler.StepLR
uvr train pathing.paths.ckpt_dir=models/checkpoints
```

```yaml
# @package train.hp

epochs: 3
lr: 0.0003
batch_size: 32
d_model: 64
n_heads: 2
n_layers: 1
pin_memory: false
shuffle_train: true
shuffle_val: false
num_workers: 2
amp: true
grad_clip: 1.0
use_wandb: true
use_ws: false

# @package train.optim

_target_: torch.optim.Adam
lr: 0.0003
weight_decay: 0.0001

# @package train.slr

_target_: torch.optim.lr_scheduler.StepLR
step_size: 5
gamma: 0.5
```


profiler options
```yaml
# @package train.prof
enabled: false
wait: 1
warmup: 1
active: 3
repeat: 1
record_shapes: true
profile_memory: true
with_stack: false
```

Pathing options
```yaml
paths:
  root: .
  raw_dir: data/voice_of_birds
  processed_dir: data/processed
  reports_dir: reports
  eval_dir: reports/eval_metrics
  ckpt_dir: models/checkpoints
  profile_dir: reports/torch_prof
  out_dir: ${paths.processed_dir}/shards
  x_train: ${paths.processed_dir}/train_x.pt
  y_train: ${paths.processed_dir}/train_y.pt
  x_val: ${paths.processed_dir}/val_x.pt
  y_val: ${paths.processed_dir}/val_y.pt
```

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We ensured reproducibility by using hydra configuration files to manage all experiemtns parameters. Each experiment is run with fully specified configs, including preprocessing settings, model architecture, training hyperparameters, and runtime options. When a run is launched, Hydra automatically creates a dedicated log directory under outputs/<date>/<time>/, which contains the exact configuration files used for that experiment. This guarantees that no parameter choices are lost, even when experimenting with multiple hyperparameter variations.

During training, all active parameters are printed to the terminal and stored in the Hydra output directory. In addition, when train.hp.use_wandb=True, we log training and validation metrics, learning rates, and model performance statistics to a Weights & Biases project, providing a persistent and centralized experiment history.

To reproduce a previous experiment, one can simply locate the Hydra output folder, inspect the saved configuration files, and rerun the training using those exact parameters. This setup ensures that preprocessing steps, training behavior, and results can be reproduced by others or at a later time.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

We have setup W&B for experiment logging of our training and additionally setup a hyperparameter sweep.yaml file in our configs folder. We pass our Hydra/Omegaconf configuration to wandb when we initialize it which ensures that all hyperparams are store correclty for each run. The runs are logged to track model dimension, number of layers and attentionheads, bacth size, and learning rate (eg. dm64_L1_H2_bs16_lr0.003). We track the train/validations loss and accuracy metrics and learning rate, so that we can track/detect overfitting, convergence etc. 

[This figure](figures/wandb_experiement_logs.png) shows the logging of all of our previous runs on the left hand side. On the right we see the train/validation loss and accuracy graphs and a learning rate over each epochs, and this allows us to track the performance of each run, and take note of which hyperparamter configurations yield the best results.

[This figure](figures/wandb_hyperparameter_sweep.png) shows a hyper parameter sweep which follows the configuration seen in configs/wandb/sweep.yaml. Here we run 10 experiments and vary the learning rate, epochs and batch sizes for each run, and this is done in order to find an optimal configuration of hyperparamters to minimize the validation and training loss of the model. The left sidepanel shows the 10 runs for the sweep, and the sweep graphs (top row of images) shows which runs yields the lowest validation loss, what parameters are most influential with respect to validation loss, and the rightmost graph shows a "pathway graph" which shows how the different configurations of batch_size + epochs + learning rate impact the validation loss. The remaining graphs are train/val loss and accuracy for the 10 runs.



### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging methods varied from groups members. Mostly print() was enough to debug the script. Profiling (torch.profiler) is included in train_engine.py, our code is already fucking prime. We used the profiler to update train_engine, from always using spectogram augmentation to it being optional, through the profiler it was seen that it was very computations heavy and took long time to due. Plus our training accuracy increased.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

Our bucket is called bird-cage bucket and contains 3 folders, data/ models/ voice_of_birds/

data/ includes a subfolder processed/ which contains all processed data
models/ contains .pt and .pth files of models
voice_of_birds/ contains folders for each species of bird, each containing 

[bucket image including processed data](figures/bucket.png)
[bucket image including bird mp3 files](figures/bucket_2.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[artifact registry](figures/artifact_registry.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

[cloud build history](figures/cloud_build_history.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 24 fill here ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Yes, we did manage to implement monitoring to check drift in data. We did this locally in a script src/call_of_func/data/data_drift.py, and this script essentially compares current data (our processed validation split) with reference data (the processed train split) across relevant sound metrics (mel_mean, mel_std, energy_mean, energy_std) and generates an Evidently html report that summarizes if there is data drifting or not. We also managed to implement this feature API on cloud run, ***FJERN DETTE HVIS IKKE VI NÅR FÆRDIG, OG TILFØJ DEL OM DEPLOYED MONITORING OGSÅ***




## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

Student s214776 was in charge of developing of setting up the initial cookie cutter project, model development, data preprocessing, training, distributed training, profiler implementation, amp/quantization, and configurations of yaml configs and dataclass configs.

Student s224473

Student s224022 

Student s224031 








