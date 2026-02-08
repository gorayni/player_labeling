import json
import logging
import logging.config
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from addict import Dict
from skimage import io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from IO import load_bboxes
from kitman.regions import get_masked_patch


def get_segmented_people(semantic_seg):
    return [(bb, mask_cnts) for bb, mask_cnts, class_id in zip(semantic_seg['boxes'],
                                                               semantic_seg['masks'],
                                                               semantic_seg['class_ids'])
            if class_id == 0]  # Person class ID is 0

class MaskedPatch():
    def __init__(self, path, masked_patch):
        self.path = path
        self.masked_patch = masked_patch


def _load_masked_patches(args):
    half, frame_idx, frame_path, segmented_people = args
    frame = io.imread(frame_path)

    masked_patches = []
    for idx, (bb, mask_cnts) in enumerate(segmented_people):
        masked_patch = get_masked_patch(frame, bb, mask_cnts)
        masked_patches.append(MaskedPatch((half, frame_idx, idx), masked_patch))
    return masked_patches


class MaskedPatchesDataset(Dataset):

    def __init__(self, match_path: Path, transform=None, num_processes=None):
        self.match_path = match_path
        self.transform = transform
        self.data = []

        num_patches = MaskedPatchesDataset.get_num_patches(match_path)
        with tqdm(total=num_patches, desc='Loading patches progress', leave=True, position=0) as pbar:
            for half in range(2):
                num_rgb_frames = len(load_bboxes(match_path, half))

                segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
                semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

                frames_path = match_path.joinpath(f'{half + 1}_HQ', 'frames')
                
                halves, frame_indices, frame_paths, segmented_people  = [], [], [], []
                for frame_idx in range(num_rgb_frames):
                    segmented_people_in_frame = get_segmented_people(semantic_seg[frame_idx ])
                    if len(segmented_people_in_frame) > 0:
                        halves.append(half)
                        frame_indices.append(frame_idx)
                        frame_paths.append(frames_path.joinpath(f'{frame_idx + 1:05}.jpg'))
                        segmented_people.append(segmented_people_in_frame)

                with Pool(processes=num_processes) as pool:

                    for masked_patches in pool.imap_unordered(_load_masked_patches, zip(halves, frame_indices, frame_paths, segmented_people)):
                        self.data.extend(masked_patches)
                        pbar.update(len(masked_patches))

    @staticmethod
    def get_num_patches(match_path: Path):
        num_patches = 0
        for half in range(2):
            num_rgb_frames = len(load_bboxes(match_path, half))

            segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
            semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

            for idx in range(num_rgb_frames):
                num_patches += len(get_segmented_people(semantic_seg[idx]))
        return num_patches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        masked_patch = self.data[idx].masked_patch
        if self.transform:
            masked_patch = self.transform(masked_patch)
        return self.data[idx].path, masked_patch


def __init_results():
    results = {}
    for half in range(2):
        num_rgb_frames = len(load_bboxes(match_path, half))

        segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
        semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

        results[half] = {}
        for frame_idx in range(num_rgb_frames):
            segmented_people = get_segmented_people(semantic_seg[frame_idx])
            results[half][frame_idx] = [None] * len(segmented_people)
    return results


def main(model_args, pred_args, match_path, num_loading_processes):
    logging.info("Parameters:")
    logging.info(model_args)
    logging.info(pred_args)

    if model_args.name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, model_args.num_classes)
    elif model_args.name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=False)
        model.classifier._modules['3'] = nn.Linear(1280, model_args.num_classes)
    elif model_args.name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier._modules['3'] = nn.Linear(1024, model_args.num_classes)

    checkpoint = torch.load(model_args.weights_path)
    model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model.cuda()

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    match_dataset = MaskedPatchesDataset(match_path, data_transform, num_loading_processes)
    loader = DataLoader(match_dataset,
                        batch_size=pred_args.batch_size,
                        shuffle=False,
                        num_workers=pred_args.max_num_workers,
                        sampler=None)
    model.eval()

    results = __init_results()
    with tqdm(enumerate(loader), total=len(loader)) as t:
        for i, (paths, inputs) in t:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            predictions = outputs.cpu().detach().numpy()

            halves, frame_indices, indices = map(lambda x: x.cpu().detach().numpy(), paths)

            for half, frame_idx, idx, prediction in zip(halves, frame_indices, indices, predictions):
                results[half][frame_idx][idx] = prediction
    return results


def parse_args():
    parser = ArgumentParser(description='Players labeling training')
    parser.add_argument('conf', help='JSON model configuration filepath',
                        type=lambda p: Path(p))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--single_match',
                       help='Directory path for the match to process (default: None)',
                       default=None, type=lambda p: Path(p))
    group.add_argument('-m', '--matches',
                       help='Path for a file containing a matches list to process',
                       default=None, type=lambda p: Path(p))
    parser.add_argument('--num_loading_processes', required=False,
                        help='Number of processes to load masked patches (default: 1)',
                        default=1, type=int)
    parser.add_argument('--max_num_workers', required=False,
                        help='Number of workers for dataloader (default: 2)',
                        default=2, type=int)
    parser.add_argument('--weights', required=False,
                        help='Weights to load (default: None)',
                        default=None, type=str)
    parser.add_argument('--GPU', required=False,
                        help='ID of the GPU to use (default: -1)',
                        default=-1, type=int)
    parser.add_argument('--log_config', required=False,
                        help='Logging configuration file (default: config/log_config.yml)',
                        default="config/log_config.yml", type=lambda p: Path(p))

    args = parser.parse_args()

    if args.matches:
        with args.matches.open() as f:
            match_paths = [Path(m) for m in f.read().splitlines()]
    else:
        match_paths = [args.single_match]

    with open(args.conf) as json_file:
        conf = json.load(json_file)
        conf = Dict(conf)

    prediction = Dict({'batch_size': conf.optimization.initial.batch_size,
                       'max_num_workers': args.max_num_workers,
                       'weights': args.weights})

    with open(args.log_config, 'rt') as f:
        log_config = yaml.safe_load(f.read())
    logs = Dict({'log_config': log_config})

    return Dict({'match_paths': match_paths,
                 'model': conf.model,
                 'prediction': prediction,
                 'logs': logs,
                 'GPU': args.GPU,
                 'num_loading_processes': args.num_loading_processes})


if __name__ == '__main__':
    args = parse_args()

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    for match_path in tqdm(args.match_paths, desc='Overall Progress', leave=True, position=0):

        team_classification_results_fpath = match_path.joinpath(f'team_classification_results.npy')
        if team_classification_results_fpath.exists():
            continue

        logging.info(str(match_path))

        if not args.prediction.weigths:
            weights_dir = match_path.joinpath('player_labeling', 'weights')
            args.model.weights_dir = weights_dir.joinpath(args.model.name)
            args.model.weights_path = args.model.weights_dir.joinpath("initial_model.pth.tar")
        else:
            args.model.weights_path = args.prediction.weigths

        log_fname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_prediction.log')
        log_fpath = args.model.weights_dir.joinpath('logs', log_fname)
        args.logs.log_config['handlers']['file_handler']['filename'] = str(log_fpath)

        log_fpath.parent.mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(args.logs.log_config)

        start = time.time()
        logging.info('Starting main function')
        try:
            results = main(args.model, args.prediction, match_path, args.num_loading_processes)
            np.save(team_classification_results_fpath, results)
        except Exception as e:
            logging.info(f'An exception occurred: {e}')
        logging.info(f'Total Execution Time is {time.time() - start} seconds')
