#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from servers.serverGradTopK import FedTopK


from models.lenet_femnist import LeNetFEMNIST
from models.lenet_mnist import LeNetMNIST
from models.mobilenet_v2_femnist import MobileNetV2FEMNIST
from models.mobilenet_v2_mnist import MobileNetV2MNIST
from models.alexnet_femnist import AlexNetFEMNIST
from models.alexnet_mnist import AlexNetMNIST
from utils.result import average_data
from utils.memory import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):
    time_list = []
    reporter = MemReporter()
    
    for i in range(args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        print(args.model_name)
        if args.model_name == "lenet_femnist":
            args.model = LeNetFEMNIST(num_classes=args.num_classes).to(args.device)

        elif args.model_name == "lenet_mnist":
            args.model = LeNetMNIST(num_classes=args.num_classes).to(args.device)
        elif args.model_name == "mobilenetv2_mnist":
            # args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            args.model = MobileNetV2MNIST(num_classes=args.num_classes).to(args.device)
        elif args.model_name == "mobilenetv2_femnist":
            # args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            args.model = MobileNetV2FEMNIST(num_classes=args.num_classes).to(args.device)
        elif args.model_name =="alexnet_mnist":
            args.model = AlexNetMNIST(num_classes=args.num_classes).to(args.device)
        elif args.model_name =="alexnet_femnist":
            args.model = AlexNetFEMNIST(num_classes=args.num_classes).to(args.device)
        else:
            raise NotImplementedError

        # select algorithm
        if args.algorithm == "GradTopK":
            server = FedTopK(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument(
        "-go", "--goal", type=str, default="test", help="The goal for this experiment"
    )
    parser.add_argument(
        "-dev", "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )
    parser.add_argument("-did", "--device_id", type=str, default="0")
    parser.add_argument("-data", "--dataset", type=str, default="mnist")
    parser.add_argument("-nb", "--num_classes", type=int, default=10)
    parser.add_argument("-m", "--model_name", type=str, default="mobilenet_v2")
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument(
        "-lr",
        "--local_learning_rate",
        type=float,
        default=0.01,
        help="Local learning rate",
    )
    parser.add_argument("-ld", "--learning_rate_decay", type=bool, default=True)
    parser.add_argument("-ldg", "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument("-gr", "--global_rounds", type=int, default=2000)
    parser.add_argument(
        "-ls",
        "--local_epochs",
        type=int,
        default=1,
        help="Multiple update steps in one local epoch.",
    )
    parser.add_argument("-algo", "--algorithm", type=str, default="GradTopK")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument("-tk", "--topk", type=int, default=1000)
    parser.add_argument("-tkalgo", "--topk_algo", type=str, default="global")
    parser.add_argument("-nw", "--num_writers", type=int, default=1)
    parser.add_argument(
        "-nc", "--num_clients", type=int, default=3, help="Total number of clients"
    )
    parser.add_argument("-t", "--times", type=int, default=1, help="Running times")
    parser.add_argument(
        "-eg", "--eval_gap", type=int, default=1, help="Rounds gap for evaluation"
    )

    parser.add_argument("-sfn", "--save_folder_name", type=str, default="items")
    parser.add_argument("-ab", "--auto_break", type=bool, default=False)

    # practical
    parser.add_argument(
        "-cdr",
        "--client_drop_rate",
        type=float,
        default=0.0,
        help="Rate for clients that train but drop out",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("TopK Method : {}".format(args.topk_algo))
    print("TopK K : {}".format(args.topk))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Client drop rate: {}".format(args.client_drop_rate))

    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model_name))
    print("Using device: {}".format(args.device))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    print("=" * 50)

    run(args)
