import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random
import numpy as np
import pickle


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}_{}".format(args["model_name"],args["dataset"], init_cls, args['increment'], args["kshot"])
    saved_path = "saved_model/{}/{}/{}_{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    logfilename = "logs/{}/{}/{}/{}_{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["kshot"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    args["base_model_path"] = "saved_model/{}/{}/{}_{}/{}_{}_{}_{}.pth".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["model_prefix"],
        args["tuned_epoch"],
        args["seed"],
        args["backbone_type"],
    )
    

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    top1_curve = {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        model.incremental_train(data_manager)

        top1_accy = model.eval_task()
        model.after_task()

        top1_curve["top1"].append(top1_accy["top1"])

        logging.info("Top1 curve: {}".format(top1_curve["top1"]))
        
        Hacc, old_acc, new_acc = Harmonic_Accuracy(top1_accy["grouped"], args["init_cls"])
        logging.info("Average Accuracy (Top1): {}   (Harmonic Accuracy): {} (Old Acc): {} (New Acc): {} \n".format(sum(top1_curve["top1"])/len(top1_curve["top1"]),
                                                                            Hacc, old_acc, new_acc))

    logging.info("\n")

    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def Harmonic_Accuracy(grouped_acc, init_cls):
    old_acc, new_acc = [], []
    for key in grouped_acc.keys():
        if '-' in key:  
            if int(key.split('-')[1]) < init_cls:
                old_acc.append(grouped_acc[key])
            elif int(key.split('-')[1]) > init_cls:
                new_acc.append(grouped_acc[key])
    old_acc = sum(old_acc) / len(old_acc)

    if len(new_acc) > 0:
        new_acc = sum(new_acc) / len(new_acc)
        Hacc = 2 * old_acc * new_acc / (old_acc + new_acc)
    else:
        Hacc = None
    return Hacc, old_acc, new_acc
