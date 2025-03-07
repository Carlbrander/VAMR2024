### VAMR 2024 ###

### VO Pipeline for VAMR 2024 Mini Project. ###



### Setup ###

#### _The conda way:_
- Use conda and the provided `mini_project.yml` file to initiate the conda environment.
It is based on the minimal set of possible libraries needed to get everything working and should be kept as lean as possible.

#### Donwload data:
To use the VO pipeline, download all three datasets from the website and include them in the `data` folder inside this repository "VAMR2024" folder, i.e. unpack the .zip archives into the `VAMR2024/data/...`




### Run VO Pipeline ###

The Pipeline can be run with: `python main.py`
- Kitti (default): `python main.py --ds 0`
- Malaga: `python main.py --ds 1`
- Parking: `python main.py --ds 2`

You can pass the following arguments to speed up:
- avoid interactive plotting: `--visualize_dashboard False`
- only visualize every n'th frame: `--visualize_every_nth_frame 200`

```
python main.py
```


### Hardware Used ###

- OS: Ubuntu 22.04LTS
- Used Threads /CPU Cores: 1
- CPU: i7-1280P
- RAM: 16GB
- Laptop: Lenovo Yoga i9 14IAP7

More details about pipeline speed (fps) can be find in the report.


