# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

import wandb
from pathlib import Path
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--exp-name', help='experiment name')
    parser.add_argument('--work-dir', help='the dir to save logs and models', default='/data/ephemeral/home/results/')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--epochs', type=int, default=20, help='# of epochs during training')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    root_path = Path(__file__).parent.parent

    config_name = args.config
    config_path = os.path.join(root_path, "mmdetection/configs/_teamconfig_/")
    config_path = os.path.join(config_path, config_name)

    cfg = Config.fromfile(config_path)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.gpu_ids = [0]
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=args.epochs)
    cfg.evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
    if args.exp_name is not None:
        exp_name = args.exp_name
    else:
        exp_name = config_name

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # # dump config

    # cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)

    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed)
    cfg.seed = seed

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        assert 'val' in [mode for (mode, _) in cfg.workflow]
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.get(
            'pipeline', cfg.data.train.dataset.get('pipeline'))
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    wandb.init(project="Boost Camp Lv2-1",
                name=f"{exp_name}",
                config={"lr": 0.02, "batch_size": 32},
                dir=args.work_dir) 

    train_detector(
        model,
        datasets[0],
        cfg,
        validate=True,
        )   


if __name__ == '__main__':
    main()
