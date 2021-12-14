import logging.config
from datetime import datetime
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


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
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomGrayscale(p=0.15),
            transforms.ToTensor()
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


def load_log_configuration(log_config: Path, logs_dir: Path):
    log_fname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
    log_fpath = logs_dir.joinpath(log_fname)
    with log_config.open(mode='rt') as f:
        log_config = yaml.safe_load(f.read())
        log_config['handlers']['file_handler']['filename'] = str(log_fpath)

    log_fpath.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(log_config)
