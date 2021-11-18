import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets
from torchvision import transforms
from datetime import datetime


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
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=train,
                      num_workers=num_workers)


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, dir_path: Path):
        dir_path.mkdir(parents=True, exist_ok=True)
        log_fname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.pkl')
        self.path = dir_path.joinpath(log_fname)
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with self.path.open(mode='wb') as fp:
            pickle.dump(self.data, fp, -1)
