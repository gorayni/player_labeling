import json
import logging
import logging.config
import os
import time
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from addict import Dict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from util import AverageMeter
from util import get_dir_loader


def train(loaders, model, optimizer, scheduler, criterion, writer, best_weights_path: Path, max_epochs=1000,
          evaluation_frequency=20):
    logging.info("Start initial training")

    best_loss = np.Inf
    running_train_loss, running_valid_loss = 0., 0.

    for epoch in range(max_epochs):

        # train for one epoch
        training_loss = process_epoch(loaders.train, model, criterion, optimizer, epoch + 1)
        running_train_loss += training_loss

        # evaluate on validation set
        validation_loss = test(loaders.valid, model, criterion, optimizer, epoch + 1)
        running_valid_loss += validation_loss

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }

        if validation_loss < best_loss:
            logging.info(f'Validation loss: {validation_loss:.2f} Training loss: {training_loss:.2f}')
            torch.save(state, best_weights_path)

        best_loss = min(validation_loss, best_loss)

        # Test the model on the validation set
        if (epoch + 1) % evaluation_frequency == 0:
            writer.add_scalar('training loss',
                              running_train_loss / evaluation_frequency,
                              (epoch + 1) * len(loaders.train))

            writer.add_scalar('validation loss',
                              running_valid_loss / evaluation_frequency,
                              (epoch + 1) * len(loaders.valid))

            running_train_loss, running_valid_loss = 0.0, 0.0

        # Reduce LR on Plateau after patience reached
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(validation_loss)
        curr_lr = optimizer.param_groups[0]['lr']
        if curr_lr is not prev_lr and scheduler.num_bad_epochs == 0:
            logging.info("Plateau Reached!")

        if prev_lr < 2 * scheduler.eps and scheduler.num_bad_epochs >= scheduler.patience:
            logging.info("Plateau Reached and no more reduction -> Exiting Loop")
            break
    return


def process_epoch(loader, model, criterion, optimizer, epoch, evaluate_=False):
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    description = '{mode} {epoch}: ' \
                  'Time {avg_time:.3f}s (it:{it_time:.3f}s) ' \
                  'Data:{avg_data_time:.3f}s (it:{it_data_time:.3f}s) ' \
                  'Loss {loss:.4e}'

    if evaluate_:
        model.eval()
        mode = 'Evaluate'
    else:
        model.train()
        mode = 'Train'

    start = time.time()
    with tqdm(enumerate(loader), total=len(loader)) as t:
        for i, (inputs, labels) in t:
            # measure data loading time
            data_time.update(time.time() - start)

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), loader.batch_size)

            if not evaluate_:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            t.set_description(description.format(mode=mode,
                                                 epoch=epoch,
                                                 avg_time=batch_time.avg,
                                                 it_time=batch_time.val,
                                                 avg_data_time=data_time.avg,
                                                 it_data_time=data_time.val,
                                                 loss=losses.avg))

    return losses.avg


def test(loader, model, criterion, optimizer, epoch):
    return process_epoch(loader, model, criterion, optimizer, epoch, evaluate_=True)


def initial_training(model_args, opt_args, train_args, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt_args.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 amsgrad=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                                           patience=opt_args.patience)
    writer = SummaryWriter(train_args.log_dir, train_args.initial_comment)

    Loaders = namedtuple('Loaders', 'train valid')

    
    train_loader = get_dir_loader(train_args.training_path, opt_args.batch_size, train_args.max_num_workers)
    valid_loader = get_dir_loader(train_args.validation_path, opt_args.batch_size, train_args.max_num_workers)
    train(Loaders(train_loader, valid_loader), model, optimizer, scheduler, criterion, writer,
          model_args.initial_weights_path, opt_args.max_epochs, train_args.evaluation_frequency)


def main(model_args, opt_args, train_args):
    logging.info("Parameters:")
    logging.info(model_args)
    logging.info(opt_args)
    logging.info(train_args)

    # fix random seeds
    torch.manual_seed(opt_args.random_seed)
    torch.cuda.manual_seed_all(opt_args.random_seed)
    np.random.seed(opt_args.random_seed)

    if model_args.name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, model_args.num_classes)
    elif model_args.name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=False)
        model.classifier._modules['3'] = nn.Linear(1280, model_args.num_classes)
    elif model_args.name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier._modules['3'] = nn.Linear(1024, model_args.num_classes)

    for param in model.parameters():
        param.requires_grad = True

    if torch.cuda.is_available():
        model.cuda()

    initial_training(model_args, opt_args.initial, train_args, model)


def parse_args():
    parser = ArgumentParser(description='Players labeling training')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--single_match',
                       help='Directory path for the match to process (default: None)',
                       default=None, type=lambda p: Path(p))
    group.add_argument('-m', '--matches',
                       help='Path for a file containing a matches list to process',
                       default=None, type=lambda p: Path(p))
    parser.add_argument('-c', '--conf', required=False,
                        help='JSON configuration filepath (default: config/velocity.json)',
                        default="config/training.json", type=lambda p: Path(p))
    parser.add_argument('--max_num_workers', required=False,
                        help='number of worker to load data (default: 2)',
                        default=2, type=int)
    parser.add_argument('--evaluation_frequency', required=False,
                        help='Evaluation frequency in number of epochs (default: 10)',
                        default=10, type=int)
    parser.add_argument('--weights', required=False,
                        help='Weights to load (default: None)',
                        default=None, type=str)
    parser.add_argument('-g', '--GPU', required=False,
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

    comment_tmp = f'lr:{0} batch_size:{1}'
    initial_comment = comment_tmp.format(conf.opt.initial.learning_rate,
                                         conf.opt.initial.batch_size)

    training = Dict({'max_num_workers': args.max_num_workers,
                     'evaluation_frequency': args.evaluation_frequency,
                     'weights': args.weights,
                     'initial_comment': initial_comment})

    with open(args.log_config, 'rt') as f:
        log_config = yaml.safe_load(f.read())
    logs = Dict({'log_config': log_config})

    return Dict({'match_paths': match_paths,
                 'model': conf.model,
                 'optimization': conf.optimization,
                 'training': training,
                 'logs': logs,
                 'GPU': args.GPU})


if __name__ == '__main__':
    args = parse_args()

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    for match_path in tqdm(args.match_paths, desc='Overall Progress', leave=True, position=0):
        weights_dir = match_path.joinpath('player_labeling', 'weights')
        args.model.weights_dir = weights_dir.joinpath(args.model.name)
        args.model.initial_weights_path = args.model.weights_dir.joinpath("initial_model.pth.tar")

        args.training.data_path = match_path.joinpath('player_labeling', 'data')
        args.training.training_path = args.training.data_path.joinpath('train')
        args.training.validation_path = args.training.data_path.joinpath('valid')
        args.training.log_dir = args.model.weights_dir.joinpath('runs', args.training.initial_comment),

        log_fname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
        log_fpath = args.model.weights_dir.joinpath('logs', log_fname)
        args.logs.log_config['handlers']['file_handler']['filename'] = str(log_fpath)

        log_fpath.parent.mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(args.logs.log_config)

        start = time.time()
        logging.info('Starting main function')
        main(args.model, args.optimization, args.training)
        logging.info(f'Total Execution Time is {time.time() - start} seconds')
