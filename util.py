import logging.config
from datetime import datetime
from pathlib import Path

import faiss
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from typing import NamedTuple


class Shape(NamedTuple):
    height: int
    width: int


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_dir_loader(data_dir, batch_size, num_workers=4, train=True):
    if train:
        data_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAutocontrast(p=0.2),
                transforms.RandomGrayscale(p=0.15),
                transforms.ToTensor(),
            ]
        )
    else:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers
    )


def load_log_configuration(log_config: Path, logs_dir: Path):
    log_fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_fpath = logs_dir.joinpath(log_fname)
    with log_config.open(mode="rt") as f:
        log_config = yaml.safe_load(f.read())
        log_config["handlers"]["file_handler"]["filename"] = str(log_fpath)

    log_fpath.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(log_config)


def images_size(frame_path: Path) -> Shape:
    im = Image.open(frame_path)
    width, height = im.size
    return Shape(height=height, width=width)


class FaissKMeans:
    def __init__(self, num_clusters, max_iter, gpu_devices=None):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.gpu_devices = gpu_devices
        self.kmeans = None
        self.centroids = None
        self.index = None

    def fit(self, x):
        if self.gpu_devices:
            self.kmeans = faiss.Clustering(x.shape[1], self.num_clusters)
            self.kmeans.niter = self.max_iter

            res = {g: faiss.StandardGpuResources() for g in self.gpu_devices}
            flat_config = {}
            for g in self.gpu_devices:
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = False
                cfg.device = g
                flat_config[g] = cfg

            if len(self.gpu_devices) > 1:
                indexes = [
                    faiss.GpuIndexFlatL2(res[g], x.shape[1], flat_config[g])
                    for g in self.gpu_devices
                ]
                self.index = faiss.IndexReplicas()
                for sub_index in indexes:
                    self.index.addIndex(sub_index)
            else:
                self.index = faiss.GpuIndexFlatL2(
                    res[self.gpu_devices[0]],
                    x.shape[1],
                    flat_config[self.gpu_devices[0]],
                )

            # perform the training
            self.kmeans.train(x, self.index)
            self.centroids = faiss.vector_float_to_array(self.kmeans.centroids).reshape(
                self.num_clusters, x.shape[1]
            )
        else:
            kmeans = faiss.Kmeans(
                d=x.shape[1], k=self.num_clusters, niter=self.max_iter
            )
            kmeans.train(x)
            self.centroids = kmeans.centroids
            self.index = kmeans.index

    def predict(self, x):
        return self.index.search(x, 1)[1]
