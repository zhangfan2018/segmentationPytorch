
import os
import datetime

import torch
from apex import amp
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as CUDNN
from torch.utils.data import DataLoader
from apex.parallel import convert_syncbn_model
from prefetch_generator import BackgroundGenerator


# data prefetch loader.
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# base model.
class BaseModel:
    """Base model, superclass"""
    def __init__(self, args, network):
        # init model params.
        self.args = args
        self.lr = self.args.lr * self.args.batch_size if self.args.lr < self.args.batch_size*1e-3 else \
            self.args.lr
        if self.args.mode != "train":
            self.args.is_apex_train = False
            self.args.is_distributed_train = False
        if self.args.n_workers <= self.args.batch_size + 2:
            self.args.n_workers = self.args.batch_size + 2
        assert self.args.num_classes == len(self.args.label)
        self.start_epoch = self.args.start_epoch
        self.metric_non_improve_epoch = 0

        # set distribute training config.
        if self.args.is_distributed_train:
            dist.init_process_group(backend='nccl')
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            self.local_rank = local_rank
            self.is_print_out = True if local_rank == 0 else False # GPU 0, print information.
            self.network = network.to(device) if args.cuda else network
        else:
            self.network = network.cuda() if args.cuda else network
            self.is_print_out = True

        # set synchronization batch normalization.
        self.network = convert_syncbn_model(self.network) if self.args.is_sync_bn else self.network

        # init optimizer.
        self.init_optimizer()

        # set apex training.
        if self.args.is_apex_train:
            self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level="O1")

        # load model weight.
        self.load_weights()

        # set cuda
        CUDNN.benchmark = False
        CUDNN.deterministic = False

        self.model_name = "models_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def load_weights(self):
        """load model weight."""
        if os.path.exists(self.args.weight_dir):
            if self.is_print_out: print('Loading pre_trained model...')
            checkpoint = torch.load(self.args.weight_dir)
            self.network.load_state_dict({k.replace('module.', ""): v for k, v in checkpoint['state_dict'].items()})
            self.start_epoch = checkpoint["epoch"]
            self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
            self.lr = checkpoint['lr']
        else:
            if self.is_print_out: print('Failed to load pre-trained network')

    def save_weights(self, epoch, net_state_dict, optimizer_state_dict):
        """save model weight."""
        model_dir = os.path.join(os.path.join(self.args.out_dir, self.model_name), "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save({
            'epoch': epoch,
            'state_dict': net_state_dict,
            'optimizer_dict': optimizer_state_dict,
            'lr': self.lr
        }, os.path.join(model_dir, 'model_%03d.pt' % epoch))

    def convert_to_script_module(self, dummy_input):
        """convert torch model to torch script module."""
        self.network.eval()
        model_dir = os.path.join(self.args.out_dir, 'script_models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.set_requires_grad(self.network, requires_grad=False)
        self.traced_script_module = torch.jit.trace(self.network, dummy_input)
        print(self.traced_script_module.code)
        print('Convert torch model to script susses!')

        if self.args.is_save_script_model:
            self.traced_script_module.save(os.path.join(model_dir, self.args.script_model_name))

    def set_requires_grad(self, network, requires_grad=False):
        """set requires grad."""
        for param in network.parameters():
            param.requires_grad = requires_grad

    def init_optimizer(self):
        """set optimizer."""
        if self.args.opt == "adam":
            self.optimizer = optim.Adam(self.network.parameters(),
                                        lr=self.lr,
                                        betas=(0.9, 0.99),
                                        weight_decay=self.args.l2_penalty)
        elif self.args.opt == "sgd":
            self.optimizer = optim.SGD(self.network.parameters(),
                                       lr=self.lr,
                                       momentum=0.99,
                                       weight_decay=self.args.l2_penalty)

    def get_lr(self, epoch, num_epochs, init_lr):
        """custom lr decay."""
        if epoch <= num_epochs * 0.66:
            lr = init_lr
        elif epoch <= num_epochs * 0.86:
            lr = init_lr * 0.1
        else:
            lr = init_lr * 0.05

        return lr

    def get_lr_scheduler(self):
        """lr scheduler in pytorch."""
        if self.args.lr_scheduler == "stepLR":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=0.5)
        elif self.args.lr_scheduler == "cosineLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30, eta_min=3e-5)
        # TODO
        # elif self.args.lr_scheduler == "reduceLR":
        #     self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,)


    def get_rank(self):
        """get gpu id in distribution training."""
        if not dist.is_available(): return 0
        if not dist.is_initialized(): return 0
        return dist.get_rank()

    def reduce_tensor(self, tensor: torch.Tensor):
        """run All_Reduce"""
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
        rt /= torch.distributed.get_world_size()

        return rt