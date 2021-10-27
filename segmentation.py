import torch
from PIL import Image
from skimage.io import imread
import numpy as np
from tqdm.notebook import tqdm


def images_size(frames_dir):
    im = Image.open(frames_dir.joinpath(f'{1:05}.jpg'))
    width, height = im.size
    return height, width


def preprocess_batch(point_rend, frames, height, width):
    inputs = []
    for i in range(len(frames)):
        image = point_rend.predictor.aug.get_transform(frames[i]).apply_image(frames[i])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs.append({"image": image, "height": height, "width": width})
    return inputs


def cut_masks(boxes, masks):
    masks_ = list()
    num_detections = len(boxes)
    for i in range(num_detections):
        x1, y1, x2, y2 = boxes[i]
        masks_.append(np.copy(masks[i][y1:y2, x1:x2]))
    return masks_


def segment(point_rend, inputs):
    predictions = point_rend.predictor.model(inputs)

    outputs = list()
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

        masks = cut_masks(boxes, masks)
        outputs.append({"boxes": boxes,
                        "class_ids": class_ids,
                        "scores": scores,
                        "masks": masks})
    return outputs


def segment_video(point_rend, frames_dir, num_frames, batch_size):
    height, width = images_size(frames_dir)
    results = []
    for i in tqdm(range(0, num_frames, batch_size)):
        frames = [imread(frames_dir.joinpath(f'{j + 1:05}.jpg')) for j in range(i, i + batch_size)]
        with torch.no_grad():
            inputs = preprocess_batch(point_rend, frames, height, width)
            predictions = segment(point_rend, inputs)
            torch.cuda.empty_cache()

            results.extend(predictions)
            del inputs
    return results
