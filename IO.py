import csv
import json
import uuid
from enum import Enum, unique
from pathlib import Path
from typing import Dict, List, Union
from functools import reduce
from operator import truediv

from kitman.data import DirPathsBuilder, NonZeroBasedIndex
from kitman.snv2 import load_groundtruth_bboxes

import numpy as np
from skimage import io

from kitman.regions import get_patch


class MatchPaths(DirPathsBuilder):
    def __init__(self, match_path):
        match_path = Path(match_path)
        super().__init__(
            match_path,
            {
                "calibrations": "{}_field_calib_ccbv.json",
                "clustered_segmentations": "clustering_{}_{}.npy",
                "clustered_means": "cluster_means_{}.npy",
                "clustered_segmentations_bkg": "clustering_bkg_{}_{}.npy",
                "clustered_bkg_means": "cluster_bkg_means_{}.npy",
                "frames": [["{}_HQ", "frames"], "{:05d}.jpg"],
                "groundtruth": "groundtruth.npy",  # NEEDED, seems it was created somehow
                "sampling_aspect_ratio": "sampling_aspect_ratio.txt",  # NEEDED, seems it was created somehow
                ################
                "all_segmentations": "all_segmentation_results_{}_HQ.npy",  # PointRend segmentation results for all frames
                "fixed_indices_results": "fixed_indices_results_{}.npz",  # Created for Matches to correspond RGB frames to optical flow frames
                "maskrcnn_bboxes": "{}_player_boundingbox_maskrcnn.json",  # Original Mask R-CNN bounding boxes
                "segmentations": "segmentation_results_{}_HQ.npy",  # PointRend segmentation results
                "velocity_results": "velocity_results.npy",
                "u": [["{}_HQ", "u"], "{:06d}.jpg"],  # Optical flow U component frames
                "v": [["{}_HQ", "v"], "{:06d}.jpg"],  # Optical flow V component frames
            },
            NonZeroBasedIndex(),
        )
        self.match = match_path
        self.frames_extension = "jpg"


class Match:
    def __init__(self, match_path):
        self.paths = MatchPaths(match_path)
        self.num_rgb_frames = [None, None]
        self.num_optical_flow_frames = [None, None]
        self.groundtruth_bboxes = [None, None]

    def optical_flow(self, half: int, idx: int):
        u = io.imread(self.paths.u[half, idx]).astype(np.float32)
        v = io.imread(self.paths.v[half, idx]).astype(np.float32)
        return (np.stack((u, v), axis=2) - 128) / 127.999

    def bboxes(self, half: int, idx: int):
        if self.groundtruth_bboxes[half] is None:
            self.load_groundtruth_bboxes(half)
        return self.groundtruth_bboxes[half][idx]

    def load_groundtruth_bboxes(self, half: int):
        bboxes_path = self.paths.maskrcnn_bboxes[half]
        self.groundtruth_bboxes[half] = load_groundtruth_bboxes(bboxes_path)

    def load_metadata(self):
        for half in range(2):
            if self.groundtruth_bboxes[half] is None:
                self.load_groundtruth_bboxes(half)
            self.num_rgb_frames[half] = len(self.groundtruth_bboxes[half])


@unique
class Players(Enum):
    REFEREE = 0
    TEAM_A = 1
    TEAM_B = 2
    GOALKEEPER_1 = 3
    GOALKEEPER_2 = 4


def load_jsons(full_match_path: Path, half: int, predictions_fname: str):
    json_path = full_match_path / predictions_fname.format(half + 1)
    with json_path.open() as json_file:
        data = json.load(json_file)
        return data['predictions']


def load_bboxes(full_match_path: Path, half: int):
    return load_jsons(full_match_path, half, '{}_player_boundingbox_maskrcnn.json')


def load_calibration(full_match_path, half):
    return load_jsons(full_match_path, half, '{}_field_calib_ccbv.json')


def load_num_optical_flow_frames(dataset_path: Union[Path | str]) -> Dict[Path, List[int]]:
    dataset_path = Path(dataset_path)
    num_optical_flow_frames_filepath = dataset_path / "num_optical_flow_frames.csv"
    with num_optical_flow_frames_filepath.open(mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        num_optical_flow_frames = {
            dataset_path
            / r["match_path"]: [
                int(r["num_frames_first_half"]),
                int(r["num_frames_second_half"]),
            ]
            for r in csv_reader
        }
    return num_optical_flow_frames


def load_sampling_aspect_ratios(dataset_path: Path):
    sars_filepath = dataset_path.joinpath('sampling_aspect_ratio.csv')
    with sars_filepath.open(mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        sars = {dataset_path.joinpath(r["match_path"]): reduce(truediv, map(float, r["SAR"].split(':'))) for r in csv_reader}
    return sars


def load_flow(half_match_path, idx):
    u_frame_path = half_match_path.joinpath('u', f'{idx:06}.jpg')
    u = io.imread(u_frame_path).astype(np.float32)

    v_frame_path = half_match_path.joinpath('v', f'{idx:06}.jpg')
    v = io.imread(v_frame_path).astype(np.float32)

    return (np.stack((u, v), axis=2) - 128) / 127.999


def export_vott(match_path, half, player_bboxes, model, preprocess=None, dataset_name='SoccerNet'):
    tags = ['Referee', 'Team A', 'Team B', 'Goalkeeper 1', 'Goalkeeper 2']
    frames_dir = match_path.joinpath(f'{half + 1}_HQ', 'frames')
    annotations_dir = match_path.joinpath(f'{half + 1}_HQ', 'annotations')

    database = {'name': dataset_name,
                'sourceConnection': {
                    'name': match_path.stem,
                    'providerType': 'localFileSystemProxy',
                    'providerOptions': {
                        'folderPath': f'{frames_dir}',
                        'relativePath': True
                    },
                    'id': str(uuid.uuid4())
                },
                'targetConnection': {
                    'name': f'{match_path.stem} Annotations',
                    'providerType': 'localFileSystemProxy',
                    'providerOptions': {
                        'folderPath': f'{annotations_dir}',
                        'relativePath': True
                    },
                    'id': str(uuid.uuid4())
                },
                'videoSettings': {
                    'frameExtractionRate': 2
                },
                'tags': [
                    {
                        'name': "Referee",
                        'color': "#000000"
                    },
                    {
                        'name': "Team A",
                        'color': "#015cda"
                    },
                    {
                        'name': "Team B",
                        'color': "#ff123b"
                    },
                    {
                        'name': "Goalkeeper 1",
                        'color': "#6917aa"
                    },
                    {
                        'name': "Goalkeeper 2",
                        'color': "#70c400"
                    }
                ],
                "useSecurityToken": False,
                "id": str(uuid.uuid4()),
                "activeLearningSettings": {
                    "autoDetect": False,
                    "predictTag": True,
                    "modelPathType": "coco"
                },
                "exportFormat": {
                    "providerType": "vottJson",
                    "providerOptions": {
                        "assetState": "visited",
                        "includeImages": True
                    }
                },
                "version": "2.2.0",
                "assets": {}
                }

    model.eval()
    annotations = []
    for idx in sorted(player_bboxes[half].keys()):
        bboxes = player_bboxes[half][idx]

        frame_path = frames_dir.joinpath(f'{idx + 1:05}.jpg')
        f = io.imread(frame_path)

        asset = {'format': 'jpg',
                 'id': f'{idx + 1:05}',
                 'name': f'{idx + 1:05}.jpg',
                 'path': f'file:{str(frame_path).replace(" ", "%20")}',
                 'size': {'width': f.shape[1],
                          'height': f.shape[0]},
                 'state': 1,
                 'type': 1}

        database['assets'][f'{idx + 1:05}'] = asset

        annotation = {'asset': asset.copy(),
                      'regions': []}

        for i, bb in enumerate(bboxes):
            patch = get_patch(f, bb, False)
            if preprocess:
                patch = preprocess(patch)
            patch = patch.unsqueeze(0).cuda()
            c = np.argmax(model(patch).cpu().detach().numpy())

            width, height = bb[2] - bb[0], bb[3] - bb[1]
            region = {'id': f'{i + 1:05}',
                      'type': 'RECTANGLE',
                      'tags': [tags[c]],
                      'boundingBox': {'left': bb[0],
                                      'top': bb[1],
                                      'height': height,
                                      'width': width},
                      'points': [{'x': bb[0],
                                  'y': bb[1]},
                                 {'x': bb[0] + width,
                                  'y': bb[1]},
                                 {'x': bb[0] + width,
                                  'y': bb[1] + height},
                                 {'x': bb[0],
                                  'y': bb[1] + height}]}
            annotation['regions'].append(region)
        annotations.append(annotation)

    annotations_dir.mkdir(parents=True, exist_ok=True)

    database_fpath = annotations_dir.joinpath(f'{dataset_name}.vott')
    with database_fpath.open('w') as f:
        json.dump(database, f)

    for a in annotations:
        annotation_fpath = annotations_dir.joinpath(f'{a["asset"]["id"]}-asset.json')
        with annotation_fpath.open('w') as f:
            json.dump(a, f)


def _to_coords(bbox):
    x0 = bbox['left']
    y0 = bbox['top']
    x1 = bbox['width'] + x0
    y1 = bbox['height'] + y0
    return [x0, y0, x1, y1]


def _to_label(region):
    categories = {'Referee': 0,
                  'Team A': 1,
                  'Team B': 2,
                  'Goalkeeper 1': 3,
                  'Goalkeeper 2': 4}

    bbox = _to_coords(region['boundingBox'])
    category = categories[region['tags'][0]]
    return bbox, category


def import_vott_dir(annotations_dir):
    labels = {}
    for annotated_frame_fpath in sorted(annotations_dir.glob('*.json')):
        with annotated_frame_fpath.open('r') as f:
            annotated_frame = json.load(f)
        if not annotated_frame['regions']:
            continue
        id_ = annotated_frame['asset']['name'][:-4]
        labels[id_] = [_to_label(r) for r in annotated_frame['regions']]
    return labels


def import_vott_file(vott_export_fpath):
    with vott_export_fpath.open('r') as f:
        annotated_dataset = json.load(f)

    labels = {}
    for annotated_frame in annotated_dataset['assets'].values():
        if not annotated_frame['regions']:
            continue
        id_ = annotated_frame['asset']['name'][:-4]
        labels[id_] = [_to_label(r) for r in annotated_frame['regions']]
    return labels
