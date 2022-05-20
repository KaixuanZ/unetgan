#from https://github.com/ajbrock/BigGAN-PyTorch (MIT license) - some modifications
""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).
    Let's go. """
import os
import functools
import math
import numpy as np
use_tqdm=False
if use_tqdm:
    from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
####
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from PyTorchDatasets import CocoAnimals
from PyTorchDatasets import  FFHQ,Celeba
from MyDataset import Dataset
# Import my stuff
import inception_utils
import utils

from PyTorchDatasets import CocoAnimals, FFHQ, Celeba
from fid_score import calculate_fid_given_paths_or_tensor
from torchvision.datasets import ImageFolder
import pickle
from matplotlib import pyplot as plt
from mixup import CutMix
import gc
import sys
from types import ModuleType, FunctionType
from gc import get_referents

####


# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


# The main training file. Config is a dictionary specifying the configuration of this training run.
#torch.backends.cudnn.benchmark = True

def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def run(config):

    import train_fns

    if config["dataset"]=="coco_animals":
        folders = ['bird','cat','dog','horse','sheep','cow','elephant','monkey','zebra','giraffe']

    # Update the config dict as necessary This is for convenience, to add settings derived from the user-specified configuration into the
    # config-dict (e.g. inferring the number of classes and size of the images from the dataset, passing in a pytorch object for the
    # activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    print("RESOLUTION: ",config['resolution'])
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    # device = 'cuda'
    device = torch.device('cpu')
    # Seed RNG
    utils.seed_rng(config['seed'])
    # Prepare root folders if necessary
    utils.prepare_root(config)
    # Setup cudnn.benchmark for free speed, but only if not more than 4 gpus are used
    if "4" not in config["gpus"]:
        torch.backends.cudnn.benchmark = True
    print(":::::::::::/nCUDNN BENCHMARK", torch.backends.cudnn.benchmark, "::::::::::::::" )
    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)
    print("::: weights saved at ", '/'.join([config['weights_root'],experiment_name]) )
    # Next, build the model
    keys = sorted(config.keys())
    for k in keys:
        print(k, ": ", config[k])
    G = model.Generator(**config).to(device)

    D = model.Unet_Discriminator(**config).to(device)

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init':True,
                                   'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None
    # FP16?
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
        # Consider automatically reducing SN_eps?

    GD = model.G_D(G, D, config)
    print(G)
    print(D)
    print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))

    # Prepare noise and randomly sampled label arrays Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    G_batch_size = int(G_batch_size*config["num_G_accumulations"])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])



    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0,'best_FID': 999999,'config': config}
    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        if config["epoch_id"] !="":
            epoch_id = config["epoch_id"]

        try:
            print("LOADING EMA")
            utils.load_weights(G, D, state_dict,
                            config['weights_root'], experiment_name, config, epoch_id,
                            config['load_weights'] if config['load_weights'] else None,
                            G_ema if config['ema'] else None)
        except:
            print("Ema weight wasn't found, copying G weights to G_ema instead")
            utils.load_weights(G, D, state_dict,
                            config['weights_root'], experiment_name, config, epoch_id,
                            config['load_weights'] if config['load_weights'] else None,
                             None)
            G_ema.load_state_dict(G.state_dict())

        print("loaded weigths")


    if config["dataset"]=="FFHQ":

        root = config["data_folder"]
        root_perm =  config["data_folder"]

        # transform = transforms.Compose(
        #     [
        #         transforms.Resize(config["resolution"]),
        #         transforms.CenterCrop(config["resolution"]),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        #     ]
        # )

        batch_size = config['batch_size']
        print("rooooot:",root)
        # dataset = FFHQ(root = root, transform = transform, batch_size = batch_size*config["num_D_accumulations"], imsize = config["resolution"])
        dataset = Dataset(path=root)
        data_loader = DataLoader(dataset, batch_size, shuffle = True, drop_last = True)
        loaders = [data_loader]

    print("Loaded ", config["dataset"])


    fake_imgs = G(z_, y_)
    _, prob = D(fake_imgs)
    prob = torch.sigmoid(res[1])

    print('D(fake image):', prob.mean())

def main():

    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    if config["gpus"] !="":
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    random_number_string = str(int(np.random.rand()*1000000)) + "_" + config["id"]
    config["stop_it"] = 99999999999999


    if config["debug"]:
        config["save_every"] = 30
        config["sample_every"] = 20
        config["test_every"] = 20
        config["num_epochs"] = 1
        config["stop_it"] = 35
        config["slow_mixup"] = False

    config["num_gpus"] = len(config["gpus"].replace(",",""))

    config["random_number_string"] = random_number_string
    new_root = os.path.join(config["base_root"],random_number_string)
    if not os.path.isdir(new_root):
        os.makedirs(new_root)
        os.makedirs(os.path.join(new_root, "samples"))
        os.makedirs(os.path.join(new_root, "weights"))
        os.makedirs(os.path.join(new_root, "data"))
        os.makedirs(os.path.join(new_root, "logs"))
        print("created ", new_root)
    config["base_root"] = new_root


    keys = sorted(config.keys())
    print("config")
    for k in keys:
        print(str(k).ljust(30,"."), config[k] )



    run(config)
if __name__ == '__main__':
    main()
