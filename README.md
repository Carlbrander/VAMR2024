### VAMR 2024 ###

### VO Pipeline for VAMR 2024 Mini Project. ###



### Setup ###

#### _Option 1. The conda way:_
- Use conda and the provided `mini_project.yml` file to initiate the conda environment.
It is based on the minimal set of possible libraries needed to get everything working and should be kept as lean as possible.

#### _Option 2. The python (pyenv) way:_
- Using Python 3.9.19, install requirements with `pip install -r requirements.txt`

#### Donwload data:
To use the VO pipeline, download all three datasets from the website and include them in the `data` folder inside this repository "VAMR2024" folder, i.e. unpack the .zip archives into the `VAMR2024/data/...`




### Run VO Pipeline ###

The Pipeline can be run with 

```
python main.py
```

And using "--ds 1" or "--ds 2" switches between the datasets. (ds 0 is the standard)
