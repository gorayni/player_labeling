import numpy as np
import torch
from imantics import Polygons
from util import images_size
from tqdm import tqdm
from skimage.io import imread


def preprocess_batch(point_rend, frames, height, width):
    inputs = []
    for i in range(len(frames)):
        image = point_rend.predictor.aug.get_transform(frames[i]).apply_image(frames[i])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs.append({"image": image, "height": height, "width": width})
    return inputs


def cut_masks(bboxes, masks, copy=False):
    masks_ = []
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        if copy:
            masks_.append(np.copy(masks[i][y1:y2, x1:x2]))
        else:
            masks_.append(masks[i][y1:y2, x1:x2])
    return masks_


def to_polygons(masks):
    if isinstance(masks, list):
        return [Polygons.from_mask(m).points for m in masks]
    return Polygons.from_mask(masks).points


def segment(point_rend, inputs, polygonal_masks=True):
    predictions = point_rend.predictor.model(inputs)

    outputs = []
    for p in predictions:
        masks = p["instances"].pred_masks
        scores = p["instances"].scores
        class_ids = p["instances"].pred_classes

        boxes = p["instances"].pred_boxes.tensor
        boxes = torch.as_tensor(boxes, dtype=torch.int64)
        boxes = boxes.cpu().numpy().astype(np.int16)

        if torch.cuda.is_available():
            class_ids = class_ids.cpu().numpy().astype(np.int16)
            masks = masks.cpu().numpy()
            scores = scores.cpu().numpy()
        else:
            class_ids = class_ids.numpy().astype(np.int16)
            masks = masks.numpy()
            scores = scores.numpy()

        if polygonal_masks:
            masks = to_polygons(cut_masks(boxes, masks))
        else:
            masks = cut_masks(boxes, masks, True)

        outputs.append({"boxes": boxes,
                        "class_ids": class_ids,
                        "scores": scores,
                        "masks": masks})
    return outputs


def segment_directory(point_rend, frame_paths, batch_size, num_frames=None):
    if num_frames is None:
        frames_dir = frame_paths.path
        num_frames = len(list(frames_dir.glob('*'+ frame_paths.suffix)))

    height, width = images_size(frame_paths[0])

    results = []
    for i in tqdm(range(0, num_frames, batch_size)):
        frames = [imread(frame_paths[j]) for j in range(i, min(i + batch_size, num_frames))]
        with torch.no_grad():
            inputs = preprocess_batch(point_rend, frames, height, width)
            predictions = segment(point_rend, inputs)
            torch.cuda.empty_cache()

            results.extend(predictions)
            del inputs
    return results
