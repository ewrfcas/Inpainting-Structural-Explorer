import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from utils import Config
from torch.utils.data import DataLoader
from dataloader import InpaintingDataset
from src.models import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./check_points',
                        help='model checkpoints path (default: ./checkpoints)')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml', config_path)

    # load config file
    config = Config(config_path)
    config.path = args.path

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids

    # init device
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.device = torch.device("cpu")
    n_gpu = torch.cuda.device_count()

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)

    # load dataset
    train_list = config.data_flist[config.dataset][0]
    val_list = config.data_flist[config.dataset][1]
    train_loader = DataLoader(
        dataset=InpaintingDataset(config, train_list, config.train_seg_path, training=True),
        batch_size=config.batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=InpaintingDataset(config, val_list, config.val_seg_path, training=False),
        batch_size=config.batch_size,
        num_workers=4,
        drop_last=False,
        shuffle=False
    )

    model = Model(config)