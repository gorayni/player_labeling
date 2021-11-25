# Player Description Creation

This repository contains the code for automatically labeling player's team and his optical flow vector direction.

The repository includes:
* Semantically segment frames from SoccerNet-V2 using PointRend.
* Creating a training subset of the people inside the soccer field with the following classes:
	- Referee
	- Team A
	- Team B
	- Goalkeeper 1
	- Goalkeeper 2
* Training a MobileNet-V3 from this subset and labeling each match from SoccerNet-V2.
* Extracting the optical flow vector direction from all players.
* Creating the description for each match in SoccerNet-V2.

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below).

## Citation

If you use this code, please cite the following paper:

Alejandro Cartas, Adrià Arbués-Sangüesa, Gloria Haro, and Coloma Ballester. "Towards Video Summarization: A Temporal Multimodal Method for Action Spotting in Sports Videos" EgoVIP Workshop at IROS (2021).

```
@Misc{cartas2021activitySpotting,
  author       = {Alejandro Cartas, Adrià Arbués-Sangüesa, Gloria Haro, and Coloma Ballester},
  title        = {Towards Video Summarization: A Temporal Multimodal Method for Action Spotting in Sports Videos},
  howpublished = {EgoVIP Workshop at International Conference on Intelligent Robots and Systems (IROS 2021)},
  month        = Octuber,
  year         = {2021},
}
```

## Requirements
Python 3.9, PyTorch 1.10.0, and other common packages listed in `environment.yml`. (A Conda environment can be created from it following the instructions below.)

# Setup

1. Clone this repository:

```shell
git clone https://github.com/gorayni/player_labeling
```

2. Create and initialize a conda environment:

```shell
cd player_labeling
conda env create --name player_labeling --file environment.yml
conda activate player_labeling
```

3. Change the SoccerNetV2 directory path in `scripts/soccernet_conf.sh` file

# Semantic Segmentation

For the player labeling and vector direction it is necessary to semantically segment the people in each frame from the dataset. For batch processing first set the number of semantic segmentation processes running in parallel in `scripts/soccernet_conf.sh`, the default value is `NUM_PROCESSES=2`. 

The next step is to split a list videos to segment for each process:

```shell
./scripts/split_remaining_videos.sh
```

This will create a number of `NUM_PROCESSES` files containing the videos to process with names starting with the prefix `tmp_video_file_`

For each temporal file we extract the semantic segmentation by executing a command like this:

```shell
python segmentation.py -v tmp_video_file_aa
```

# Player Labeling


# Optical Flow Vector


# Single video


