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
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from clustering import Kmeans
from clustering import PIC
from clustering import arrange_clustering
from clustering import cluster_assign
from util import AverageMeter
from util import Logger
from util import UnifLabelSampler
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
    writer = SummaryWriter(train_args.log_dir, train_args.comment)

    Loaders = namedtuple('Loaders', 'train valid')
    train_loader = get_dir_loader('data/train', opt_args.batch_size, train_args.max_num_workers)
    valid_loader = get_dir_loader('data/valid', opt_args.batch_size, train_args.max_num_workers)
    train(Loaders(train_loader, valid_loader), model, optimizer, scheduler, criterion, writer,
          model_args.initial_weights_path, opt_args.max_epochs, train_args.evaluation_frequency)


def compute_features(loader, model, N):
    logging.info('Computing features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()

    # discard the label information in the dataloader
    with tqdm(enumerate(loader), total=len(loader)) as t:
        for i, (input_tensor, _) in t:
            input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(loader) - 1:
                features[i * loader.batch_size: (i + 1) * loader.batch_size] = aux
            else:
                # special treatment for final batch
                features[i * loader.batch_size:] = aux

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            t.set_description(f'{i}/{len(loader)} Time: {batch_time.val:.3f} ({batch_time.avg:.3f})')
    return features


def train_deep_clustering(loader, model, criterion, optimizer, optimizer_last_fc, epoch):
    batch_time, losses, data_time = AverageMeter(), AverageMeter(), AverageMeter()

    description = 'Epoch: [{epoch}] ' \
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) ' \
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) ' \
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'

    model.train()

    end = time.time()
    with tqdm(enumerate(loader), total=len(loader)) as t:
        for i, (input_tensor, target) in t:
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = torch.autograd.Variable(input_tensor.cuda())
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

            # record loss
            losses.update(loss.item(), loader.batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            optimizer_last_fc.zero_grad()
            loss.backward()

            optimizer.step()
            optimizer_last_fc.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            t.set_description(description.format(epoch=epoch,
                                                 batch_time=batch_time,
                                                 data_time=data_time,
                                                 loss=losses))

    return losses.avg


def deepcluster_training(model_args, opt_args, train_args, model):
    criterion = nn.CrossEntropyLoss()

    # remove head
    if model_args.name == 'resnet18':
        model.fc = None
    elif 'mobilenet_v3' in model_args.name:
        model.classifier._modules['3'] = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:3])

    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=opt_args.learning_rate,
        momentum=opt_args.momentum,
        weight_decay=10 ** opt_args.weight_decay,
    )
    cluster_log = Logger(train_args.cluster_log_dir)

    dataloader = get_dir_loader('data/unlabeled', opt_args.batch_size, train_args.max_num_workers)

    deepcluster = Kmeans(5) if opt_args.clustering_algorithm == 'KMeans' else PIC(5)

    description = 'Epoch [{epoch}] ' \
                  'Time: {time:.3f}s ' \
                  'Clustering loss: {clustering_loss:.3f} ' \
                  'ConvNet loss: {loss:.3f}'

    for epoch in range(opt_args.max_epochs):

        # remove head
        if model_args.name == 'resnet18':
            model.fc = None
        elif 'mobilenet_v3' in model_args.name:
            model.classifier._modules['3'] = None
            model.classifier = nn.Sequential(*list(model.classifier.children())[:3])

        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataloader.dataset))

        logging.info('Clustering the features')
        clustering_loss = deepcluster.cluster(features)

        logging.info('Assigning pseudo labels')
        train_dataset = cluster_assign(deepcluster.images_lists, dataloader.dataset.imgs)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(opt_args.reassignment_frequency * len(train_dataset)), deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt_args.batch_size,
            num_workers=train_args.max_num_workers,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        if model_args.name == 'resnet18':
            model.fc = None
        elif 'mobilenet_v3' in model_args.name:
            input_size = 1024 if model_args.name == 'mobilenet_v3_small' else 1280
            last_fc = nn.Linear(input_size, len(deepcluster.images_lists))
            last_fc.weight.data.normal_(0, 0.01)
            last_fc.bias.data.zero_()
            last_fc.cuda()
            mlp = list(model.classifier.children())
            mlp.append(last_fc)
            model.classifier = nn.Sequential(*mlp)

            optimizer_last_fc = torch.optim.SGD(
                model.classifier._modules['3'].parameters(),
                lr=opt_args.learning_rate,
                weight_decay=10 ** opt_args.weight_decay,
            )

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train_deep_clustering(train_dataloader, model, criterion, optimizer, optimizer_last_fc, epoch)

        logging.info(description.format(epoch=epoch,
                                        time=time.time() - end,
                                        clustering_loss=clustering_loss,
                                        loss=loss))
        try:
            nmi = normalized_mutual_info_score(
                arrange_clustering(deepcluster.images_lists),
                arrange_clustering(cluster_log.data[-1])
            )
            logging.info(f'NMI against previous assignment: {nmi:.3f}')
        except IndexError:
            pass

        state = {
            'epoch': epoch + 1,
            'arch': model_args.name,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, model_args.deep_cluster_weights_path)
        cluster_log.log(deepcluster.images_lists)


def main(model_args, opt_args, train_args, main_args):
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
    
    if not main_args.only_clustering:
        initial_training(model_args, opt_args.initial, train_args, model)

    # Loading best initial trained model
    checkpoint = torch.load(model_args.initial_weights_path)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info(f"=> loaded checkpoint '{model_args.initial_weights_path}' (epoch {checkpoint['epoch']})")

    deepcluster_training(model_args, opt_args.deep_clustering, train_args, model)


def parse_args():
    parser = ArgumentParser(description='DeepClustering for players labeling training')
    parser.add_argument('conf', help='JSON model configuration filepath',
                        type=lambda p: Path(p))
    parser.add_argument('-d', '--data_path', required=True,
                        help='Path for patches data (default: None)',
                        default=None, type=lambda p: Path(p))
    parser.add_argument('--max_num_workers', required=False,
                        help='number of worker to load data (default: 2)',
                        default=2, type=int)
    parser.add_argument('--evaluation_frequency', required=False,
                        help='Evaluation frequency in number of epochs (default: 10)',
                        default=10, type=int)
    parser.add_argument('--weights_dir', required=False,
                        help='Path for weights saving directory (default: weights)',
                        default="weights", type=lambda p: Path(p))
    parser.add_argument('--weights', required=False,
                        help='Weights to load (default: None)',
                        default=None, type=str)
    parser.add_argument('--GPU', required=False,
                        help='ID of the GPU to use (default: -1)',
                        default=-1, type=int)
    parser.add_argument('--log_config', required=False,
                        help='Logging configuration file (default: config/log_config.yml)',
                        default="config/log_config.yml", type=lambda p: Path(p))
    parser.add_argument('--only_clustering', required=False,
                        help='Skip initial training (default: False)',
                        action='store_true')

    args = parser.parse_args()
    with open(args.conf) as json_file:
        conf = json.load(json_file)
        conf = Dict(conf)
    conf.model.weights_dir = args.weights_dir.joinpath(conf.model.name)
    conf.model.initial_weights_path = conf.model.weights_dir.joinpath("initial_model.pth.tar")
    conf.model.deep_cluster_weights_path = conf.model.weights_dir.joinpath("deep_cluster_model.pth.tar")

    comment_tmp = f'lr:{0} batch_size:{1}'
    comment = comment_tmp.format(conf.opt.learning_rate,
                                 conf.opt.batch_size)

    training = Dict({'data_path': args.data_path,
                     'max_num_workers': args.max_num_workers,
                     'evaluation_frequency': args.evaluation_frequency,
                     'weights': args.weights,
                     'log_dir': conf.model.weights_dir.joinpath('runs', comment),
                     'cluster_log_dir': conf.model.weights_dir.joinpath('clusters', comment),
                     'comment': comment})

    main_args = Dict({'only_clustering': args.only_clustering})

    log_fname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
    log_fpath = conf.model.weights_dir.joinpath('logs', log_fname)
    with open(args.log_config, 'rt') as f:
        log_config = yaml.safe_load(f.read())
        log_config['handlers']['file_handler']['filename'] = str(log_fpath)
    logs = Dict({'log_config': log_config,
                 'log_fpath': log_fpath})

    return Dict({'model': conf.model,
                 'optimization': conf.optimization,
                 'training': training,
                 'logs': logs,
                 'GPU': args.GPU,
                 'main': main_args})


if __name__ == '__main__':
    args = parse_args()

    args.logs.log_fpath.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(args.logs.log_config)

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    start = time.time()
    logging.info('Starting main function')
    main(args.model, args.optimization, args.training, args.main)
    logging.info(f'Total Execution Time is {time.time() - start} seconds')
