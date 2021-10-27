import json

from enum import Enum, unique


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
