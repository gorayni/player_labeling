import json
import uuid
from enum import Enum, unique

import numpy as np
from skimage import io

from regions import get_patch


@unique
class Players(Enum):
    REFEREE = 0
    TEAM_A = 1
    TEAM_B = 2
    GOALKEEPER_1 = 3
    GOALKEEPER_2 = 4


def load_jsons(full_match_path, half, predictions):
    json_fname = predictions.format(half + 1)
    json_path = full_match_path.joinpath(json_fname)

    with json_path.open() as json_file:
        data = json.load(json_file)
        return data['predictions']


def load_bboxes(full_match_path, half):
    return load_jsons(full_match_path, half, '{}_player_boundingbox_maskrcnn.json')


def load_calibration(full_match_path, half):
    return load_jsons(full_match_path, half, '{}_field_calib_ccbv.json')


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
