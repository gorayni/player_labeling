import logging
import subprocess
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from pixellib.torchbackend.instance import instanceSegmentation as PointRend
from tqdm import tqdm

from segmentation import segment_directory
from util import load_log_configuration

import hydra
from IO import Match

@hydra.main(version_base=None, config_path="conf", config_name="segment_")
def main(cfg: DictConfig):

    # Validation for the mutually exclusive group
    if (cfg.single_match is None) == (cfg.matches is None):
        raise ValueError("You must provide either 'single_match' OR 'matches', but not both/neither.")
    
    load_log_configuration(cfg.logs.config, cfg.logs.dir)
    
    point_rend = PointRend()
    point_rend.load_model(cfg.segmentation.weights)
    point_rend.predictor.model.cuda()

    if cfg.videos:
        with cfg.videos.open() as f:
            videos = [Path(line) for line in f.read().splitlines()]
    else:
        videos = [cfg.single_video]

    for video in tqdm(videos, desc="Overall Progress", leave=True, position=0):

        half = int(video.stem[0]) - 1

        match = Match(video.parent)
        if cfg.all_frames:
            segmentations_path = match.paths.all_segmentations[half]
            frames = match.paths.all_frames[half]
        else:
            segmentations_path = match.paths.segmentations[half]
            frames = match.paths.frames[half]

        if segmentations_path.exists():
            print(video)
            continue

        if not frames.path.exists():
            frames.path.mkdir(parents=True, exist_ok=True)

            logging.info(f"Extracting frames into {frames.path}")

            if cfg.all_frames:
                subprocess.check_call(
                    f'./scripts/extract_frames.sh -a "{str(video)}" "{frames.path}"',
                    shell=True,
                )
            else:
                subprocess.check_call(
                    f'./scripts/extract_frames.sh "{str(video)}" "{frames.path}"',
                    shell=True,
                )

        start = time.time()
        semantic_seg = segment_directory(point_rend, frames, cfg.segmentation.batch_size)
        np.save(match.paths.segmentations[half], semantic_seg)
        logging.info(f"Video processing time is {time.time() - start} seconds")
    
if __name__ == "__main__":
    main()