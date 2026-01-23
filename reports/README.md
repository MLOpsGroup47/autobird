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
* [x] Deploy to the cloud a drift detection API (M27)
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

The total code coverage of the code i 33%, which includes all of our source code. 100% code coverage does not necessarily ensure an errorless coding-pipeline. The coverage is only representative of the tests built by the developers and these may themselves be full of errors. But even if the errors don't have errors, and the cover 100% of the code, there is no guarantee that the code will behave flawlessly on unexpected input data. So code coverage can be a helpful indicator, but cannot be solely relied on. 

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

For our continuous intefration we have split it into 2 primary workflow files: one for doing tests and one for running linting, formatting and typecheck. 

Arguably the most important is our "Run tests" workflow located [here](../.github/workflows/tests.yaml). It runs all our pytest tests, but crucially it does so on 4 different platforms. Our group consists of users of windows, mac with M2/3 and mac with intel. Thus we set up our continuous integration to run pytests on all those 3 platforms in addition to ubuntu to ensure it would also run on a linux platform. Additionally we tested all platforms on python version 3.11 and 3.12 to ensure multiple working python versions for our project. An example of a triggered workflow can be seen here <https://github.com/MLOpsGroup47/autobird/actions/runs/21289542698>.  

Secondly we made a combined workflow called "Run linting" located [here](../.github/workflows/tests.yaml). Despite the name the workflow covers both linting, formatting and type checking. We use ruff for linting and formatting to ensure our python code lives up to the PEP8 standard, and to overall keep the code consistant and readable. For type checking we use mypy to further enhance readablity. An example of a triggered workflow can be found here <https://github.com/MLOpsGroup47/autobird/actions/runs/21289542694>.

We also tried to add a workflow for doing load testing of our API, but due to time constraints the workflow was not finished. 

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

In this project we developed multiple images: training, data preprocessing, inference, data drifting, and gcp test api.
For example can one use the training docker by writing this in the terminal:
```bash
docker run train:latest lr=0.0003 batch_size=32
```
See <https://console.cloud.google.com/artifacts/docker/autobird-484409/europe-west1/autobird-container/train-image?authuser=1&project=autobird-484409>
This training docker can take the same arguments as ```bash uvr train ```
See readme.md in src/


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

Compute Engine: Was used to create and run Virtual Machines "VMs". 

Vertex AI: Was used to automatically start, set up and end VMs for each experiment. 

Cloud Storage: We created a bucket to store our raw data, processed data, and trained models.

Artifact Registry: Repository to hold our container images. 

CLoud run: Was used to deploy our model and allow interaction through FastAPI interface. 

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

We used the compute enginge through Vertex AI to run our experiments. Our experiments were carried out on machines with following hardware:
| Component       | Specification                |
|:----------------|:-----------------------------|
| **VM Family** | N1 Series (`n1-standard-8`) |
| **vCPUs** | 8 Virtual CPUs               |
| **Memory** | 30GB RAM                     |
| **Accelerator** | NVIDIA Tesla T4 (16GB VRAM) |


The CPUs and 30GB RAM ensured sufficient compute for data loading, while the GPU was critical for accelerating the training of our model.
We launched these VMs through Vertex AI Custom Jobs using a custom Docker container stored in the Artifact Registry. This approach allowed us to ensure Compute Engine instances could execute our code consistently

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

We used Vertex AI to train our model as this eased the process of launcing, setting up and terminating VMs. Using the compute engine allowed us to train our model longer than training on our own computers ensuring better model performance. The experiments were started using our container images stored in the artifact registry by running following train-bash script in the terminal:
```bash 
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
```
With config file:
```bash
workerPoolSpecs:
    machineSpec:
        machineType: n1-standard-8
        acceleratorType: NVIDIA_TESLA_T4
        acceleratorCount: 1
    replicaCount: 1
    containerSpec:
        imageUri: europe-west1-docker.pkg.dev/autobird-484409/autobird-container/trainer:v1-gpu
        env:
        - name: WANDB_MODE
          value: "online"
        - name: WANDB_API_KEY
          value: "--INSERT_WANDB_API_KEY--" 
        - name: WANDB_PROJECT
          value: "autobird"
```
Which submits a job to Vertex AI which launches a VM with the specifications listed before and terminates it after completion. The benefit of using the compute engine was to run multiple experiments simultanously making hyperparameter tuning easier and faster as they are just changed in the train-bash script. 

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

We managed to write a working API for doing inference using our model. We made our API script working both locally and in the cloud by adjusting path strings in order to use the native project paths when working locally, and a path to our Bucket when used in the cloud. Apart from the root we made 2 endpoint called "files" and "classify". The "files" endpoint uses a GET event to list the files located in our model folder, which depends on the platform. "classify" receives an audio file through a POST event and returns the predicted class as response. We used FastAPI the make the API. We added collection of input and output as a light form of monitoring of the API. 

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

We managed to deploy our API both locally and in the cloud. We first working on serving the model locall using uvicorn to make sure the python script worked. Afterwords we made a docker image containing the API and hosted it locally. Finally, we deployed in the cloud using Cloud Run, at first by pushing and deploying the locally built image, but in the end though continuous deployment, which triggered on all pull requests to our main branch. 

To invoke our the service a user would call:

´´´bash
curl -X 'POST' \
  'https://inference-api-1047655691608.europe-west1.run.app/classify/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'audio=@audiofile.mp3;type=audio/mpeg'
´´´

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

We peformed both unit testing and load testing. As explained in question 7, we performed unit tests of our API using pytest as part of our continuous integration. For load testing we used locust where we load tested with up to 1000 users at peak without it crashing, but with 99% percentile response time on around 6500-7000 for all endpoints. Both root and "files" endpoints had 0 failures but the "classify" endpoint has a failure rate of 100% due to non resovled issue with the locust script. Thus the load testing showed that we can serve 1000 concurrent users, but with subpar response times (~6.5s). This is without accounting for inference time, caused by the faulty script.

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

Yes, to an extent. We did manage to implement monitoring to check drift in data. We did this locally in a script src/call_of_func/data/data_drift.py, and this script essentially compares current data (our processed validation split) with reference data (the processed train split) across relevant sound metrics (mel_mean, mel_std, energy_mean, energy_std) and generates an Evidently html report that summarizes if there is data drifting or not. This is important because it allows us to detect when new incoming data starts to differ from the data the model was trained on, which can lead to degraded model performance if not addressed in time. (In an industry setting we would substitute current data with new incoming data instead of just the validation data). We also managed to implement this feature in an API on cloud run (autobird-drift), which does precisely the same, but uses data directly from the bucket instead. The endpoints <url>/drift/report displays the report and <url>/drift/run returns a JSON object that determines whether drift is detected in the dataset, number of samples in current/ref and status for the run. 

In the inference API we also managed to implement monitoring of input-output. Here we collect the time and name of the uploaded file to the and we output a prediction of the class. This is important because it enables basic traceability and observability of the inference pipeline, making it easier to debug incorrect predictions and understand how the model is being used in practice. For future steps, we could have logged prediction confidence, model version, response time etc, which would allow us to monitor performance and trigger retraining when necessary.





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

We ended up spending in total 10.3$ during the development of our project. The most costly service we used was by far Vertex AI/Compute Engine which we have spend 7.26$ on and the secod most costly service was Cloud storage on which we have used 2.55$. The Vertex AI/Compute Engine is charged by usage and is the service we have used the most since we have trained multiple models for multiple hours, including using GPU which increases the cost. It has been super helpful to use the cloud, especially for training as our model training takes very long even when using GPU thus training on our own computers would have been painful. Although parts of the cloud lead to many painful hours we could not have trained our models without.

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
- Starting point is our local setup where we develop code on our seperate machines. Here we built the source code (including, model, training, evaluation, data-preprocessing pipeline etc.), config files (hydra/dataclass), dockerfiles (train/data/api), DVC, tests, logging (wandb) and profiling (TorchProfiling), cloudbuild.yaml, all in the uv framework.  
- A bucket is filled manually in GCP, which stores our data. 
- When we commit and push local changes and PRs, we have setup CI-pipeline via Github actions that tests code functionality and ensures proper code quaility. Additionally our cloudbuild.yaml is triggered via PRs which automatically builds docker images for data, train and api. We also make sure to push to DVC to track data versioning. 
- The API image (which includes all of our src) is automatically deployed in Cloud Run, which acts as a serverless service which can perform inference using our trained model. 
- Alternatively, we build a docker image, which we the push to GCP, where we train the model (VM or Vertex AI), and the best model is deployed for inference.


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

One of the biggest challenges in our project was handling packages and dependencies across different machines. Ensuring that all environments could run the same libraries consistently turned out to be difficult, especially due to hardware differences. In particular, an older Mac with an Intel chip could not support newer versions of PyTorch, which required us to adapt our setup accordingly. We also encountered persistent issues with the Librosa library, which failed to run reliably across machines and took considerable time to troubleshoot. Eventually, this issue was resolved by switching from Librosa to TorchAudio, which provided better compatibility and allowed all team members to run the same pipeline. Overall, managing dependencies and achieving a stable, reproducible environment was one of the most time-consuming aspects of the project.


We ended up spending many hours to get our training docker to run in gcloud. When building the images on the cloud and running it, the data from the bucket could not load, or it could not load modules. Thus we ended up building images locally and push them to the Artifact Registry which worked without any changes in the code and still used data from the bucket.  


Struggles of student s214776 - The preprocessing pipeline was difficult to implement correctly, and integrating Distributed Data Parallel (DDP) into `train_engine.py`. Ensuring no data leakage was a pain, as audio files had to be chunked while keeping all chunks from the same file within a single dataset split. This was necessary to prevent chunks from the same audio appearing in the training, validation, and test sets. Keeping track of which files were used during training also required careful handling.

Using DDP introduced additional pain. Understanding how to use multiple cores and processes for parallel training was a pain, and resolving duplicated log outputs, e.g. repeated `print("training started")` statements, was frustrating.


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

--- question 31 fill here ---
Student s224473 was in charge of developing a parts of the source code, (including the evaluation script), ensuring that the codebase was fully compatible with TorchAudio after transitioning away from Librosa setting up wandb configuration to track all training runs and implemented a hyperparameter sweep, and setting up data drift monitoring (Evidently), both locally and as a deployed API in the cloud.

Student s214776 was in charge of setting up the initial cookie cutter project, model development, data preprocessing, training, distributed training, profiler implementation, amp/quantization, and configs for for all of the respectively, in configs/.

Student s224022 was in charge of setting up the cloud and train our models in the cloud, including building the docker images, developing a bash script to ease the job submission. As mentioned the I struggled with building the images in the cloud and ended up building them locally and push them afterwards. Also I ensured our code was runable both locally and in the cloud using the data stored in out bucket.

Student s224031 


### Declaration on the Use of Artificial Intelligence Tools

In this project, artificial intelligence (AI) tools were used as a supporting aid in the development process, including chatGPT, Gemini, Vertex AI and Grok. The use of AI was limited to assistance with programming-related tasks, including code structuring, debugging, optimization, and clarification of technical concepts.

The AI tools were used solely as a support for the authors’ own work. All design choices, implementations, experiments, analyses, and evaluations were carried out by the authors. Any AI-generated suggestions were critically assessed, tested, and, where necessary, modified before inclusion.

AI tools were not used to generate experimental data, conduct analyses automatically, or produce results, interpretations, or conclusions. The authors take full responsibility for the academic content, correctness, originality, and integrity of the project.



