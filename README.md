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

4. Calculate the number of optical flow frames previously obtained by running the command:

```shell
./scripts/calculate_optical_flow_num_frames.sh
```

5. Calculate the sampling aspect ratio of the videos by running the command:

```shell
./scripts/calculate_sampling_aspect_ratios.sh
```

# Single Soccer Match

In this example we assume that the SoccerNetV2 directory path is `/datasets/soccernet` and that our desired soccer match to process is `/datasets/soccernet/europe_uefa-champions-league/2014-2015/2014-11-04 - 22-45 Arsenal 3 - 3 Anderlecht/`. 


```shell
soccernet_path="/datasets/soccernet"
match_path="/datasets/soccernet/europe_uefa-champions-league/2014-2015/2014-11-04 - 22-45 Arsenal 3 - 3 Anderlecht"
```

### Semantic Segmentation

```shell
python segmentation.py -s "$match_path"/1_HQ.mkv
python segmentation.py -s "$match_path"/2_HQ.mkv
```

### Player velocity from optical flow

```shell
python velocity.py -s "$match_path" -g 0 -d "$soccernet_path"
```

### Player team classification

```shell
python prepare_training_subset.py -s "$match_path"
python train.py -s "$match_path" -g 0
python team_prediction.py config/training.json --GPU 0 --num_loading_processes 10 -s "$match_path"
```

### Joining the results

```shell
python join_results.py -s "$match_path"
```

# Multiple Soccer Matches


## Semantic Segmentation

For the player labeling and vector direction it is necessary to semantically segment the people in each frame from the dataset. For batch processing first set the number of semantic segmentation processes running in parallel in `scripts/soccernet_conf.sh`, the default value is `NUM_SEGMENTATION_PROCESSES=2`. 

The next step is to split a list videos to segment for each process:

```shell
./scripts/split_remaining_videos_to_segment.sh
```

This will create a number of `NUM_SEGMENTATION_PROCESSES` files containing the videos to process with names starting with the prefix `tmp_video_file_`

For each temporal file we extract the semantic segmentation by executing a command like this:

```shell
python segmentation.py -v tmp_videos_file_aa
```
## Player velocity from optical flow

## Player team classification

```shell
./scripts/split_remaining_matches_to_label.sh
```

```shell
python prepare_training_subset.py -m tmp_matches_file_aa
```
