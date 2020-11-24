
import os
import datetime
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel

from utils.logger import get_logger
from utils.csv_tools import write_csv
from runner.loss import DiceLoss, BCELoss
from runner.base import BaseModel, DataLoaderX
from runner.metric import DiceMetric
from utils.mask_utils import smooth_mask
from data_processor.data_io import DataIO


class SegmentationModel(BaseModel):
    """segmentation model, including train, validation and test model."""
    def __init__(self, args, network):
        super(SegmentationModel, self).__init__(args, network)
        self.log_dir = os.path.join(os.path.join(self.args.out_dir, self.model_name), 'logs') \
            if self.args.mode != "test" else os.path.join(self.args.out_dir, 'logs')
        if not os.path.exists(self.log_dir) and self.is_print_out:
            os.makedirs(self.log_dir)
        if self.is_print_out:
            self.logger = get_logger(self.log_dir)
            args = vars(self.args)
            self.logger.info('\n------------ {} options -------------'.format(self.args.mode))
            for k, v in sorted(args.items()):
                self.logger.info('%s: %s' % (str(k), str(v)))
            self.logger.info('-------------- End ----------------\n')

    def train(self, train_dataset, val_dataset):
        """training procedure."""
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.network.train()

        if self.args.is_distributed_train:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        train_loader = DataLoaderX(
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.n_workers,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=True,
            sampler=train_sampler
        )

        if self.is_print_out:
            self.logger.info('preprocess parallels: {}'.format(self.args.n_workers))
            self.logger.info('train samples per epoch: {}'.format(len(self.train_dataset)))
            train_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'train'))
            val_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'val'))

        # self.get_lr_scheduler()

        if torch.cuda.device_count() > 1:
            if not self.args.is_distributed_train:
                self.network = torch.nn.DataParallel(self.network)
            elif self.args.is_apex_train and self.args.is_distributed_train:
                self.network = DistributedDataParallel(self.network, delay_allreduce=True)
            elif self.args.is_distributed_train:
                self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[self.local_rank],
                                                                         output_device=self.local_rank)
        best_dice = 0
        train_epoch = 0
        val_epoch = 0

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            if self.is_print_out:
                self.logger.info('starting training epoch {}'.format(epoch))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.get_lr(epoch, self.args.epochs, self.lr)
            # self.scheduler.step()

            self.network.train()

            # self.scheduler.step()
            start_time = datetime.datetime.now()

            dice = [0.] * self.args.num_classes
            total = 0

            for index, (images, masks) in enumerate(train_loader):
                if self.args.cuda:
                    images, masks = images.cuda(), masks.cuda()

                self.optimizer.zero_grad()
                if self.args.task == "single":
                    output_seg = self.network(Variable(images))
                elif self.args.task == "multi":
                    output_seg, output_boundary = self.network(Variable(images))

                seg_loss = 0
                dice_loss_func = DiceLoss()
                boundary_loss_func = BCELoss()
                for i, _ in enumerate(output_seg):
                    dice_loss = dice_loss_func(output_seg[i], masks, activation=self.args.activation)
                    seg_loss += dice_loss
                seg_loss /= len(output_seg)

                if self.args.task == "multi":
                    # contours
                    boundary_loss = boundary_loss_func(output_boundary, None, activation=self.args.activation)
                    seg_loss += boundary_loss

                if self.args.is_apex_train:
                    with amp.scale_loss(seg_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    seg_loss.backward()
                self.optimizer.step()

                dice_metric_func = DiceMetric()
                dice_output = dice_metric_func(output_seg[-1], masks, activation=self.args.activation, is_average=False)
                if self.args.is_distributed_train:
                    dice_output = self.reduce_tensor(dice_output.data)
                for i, dice_tmp in enumerate(dice_output):
                    dice[i] += float(dice_tmp.item())
                total += len(images)

                if self.args.is_distributed_train:
                    seg_loss = self.reduce_tensor(seg_loss.data)

                if index > 0 and index % self.args.log_interval == 0:
                    train_epoch += 1
                    if self.is_print_out:
                        self.logger.info('Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(
                        epoch, self.args.epochs, index * len(images), len(train_loader.dataset), 100. * index / len(train_loader)))
                        train_writer.add_scalar('Train/SegLoss', seg_loss.item(), train_epoch)
                        self.logger.info('SegLoss:{:.6f}'.format(seg_loss.item()))
                    if self.args.task == "multi":
                        if self.is_print_out:
                            train_writer.add_scalar('Train/BoundaryLoss', boundary_loss.item(), train_epoch)
                            self.logger.info('BoundaryLoss:{:.6f}'.format(boundary_loss.item()))

                    for i, dice_label in enumerate(dice):
                        dice_ind = dice_label / total
                        if self.is_print_out:
                            train_writer.add_scalars('Train/Dice', {self.args.label[i]: dice_ind}, train_epoch)
                            self.logger.info('{} Dice:{:.6f}'.format(self.args.label[i], dice_ind))

            val_loss, val_dice = self.validate(val_dataset)
            val_epoch += 1
            if self.is_print_out:
                val_writer.add_scalar('Val/Loss', val_loss, val_epoch)
                total_dice = 0
                for i, _ in enumerate(val_dice):
                    if self.is_print_out:
                        val_writer.add_scalars('Val/Dice', {self.args.label[i]: val_dice[i]}, val_epoch)
                    total_dice += val_dice[i]
                total_dice /= len(val_dice)

                if total_dice > best_dice:
                    self.metric_non_improve_epoch = 0
                    best_dice = total_dice
                    self.save_weights(epoch, self.network.state_dict(), self.optimizer.state_dict())
                else:
                    self.metric_non_improve_epoch += 1
            if self.is_print_out:
                self.logger.info('\nEnd of epoch {}, time: {}'.format(epoch, datetime.datetime.now() - start_time))

        if self.is_print_out:
            train_writer.close()
            val_writer.close()
            self.logger.info('\nEnd of training, best dice: {}'.format(best_dice))

    def validate(self, val_dataset):
        """validation procedure."""
        self.val_dataset = val_dataset
        self.network.eval()

        dice = [0.] * self.args.num_classes
        total = 0
        val_loss = 0

        if self.args.is_distributed_train:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            val_sampler = None

        val_loader = DataLoaderX(
            dataset=self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.n_workers,
            drop_last=False,
            pin_memory=True,
            sampler=val_sampler
        )
        if self.is_print_out:
            self.logger.info('starting validation')
            self.logger.info('val samples per epoch: {}'.format(len(self.val_dataset)))
        for images, masks in val_loader:
            if self.args.cuda:
                images, masks = images.cuda(), masks.cuda()

            with torch.no_grad():
                if self.args.task == "single":
                    output_seg = self.network(Variable(images))
                elif self.args.task == "multi":
                    output_seg, output_boundary = self.network(Variable(images))

            seg_loss = 0
            dice_loss_func = DiceLoss()
            boundary_loss_func = BCELoss()
            for i, _ in enumerate(output_seg):
                dice_loss = dice_loss_func(output_seg[i], masks, activation=self.args.activation, is_average=False)
                seg_loss += dice_loss
            seg_loss /= len(output_seg)

            if self.args.task == "multi":
                # contours
                boundary_loss = boundary_loss_func(output_boundary, None, activation=self.args.activation, is_average=False)
                seg_loss += boundary_loss

            if self.args.is_distributed_train:
                seg_loss = self.reduce_tensor(seg_loss.data)
            val_loss += seg_loss.item()

            dice_metric_func = DiceMetric()
            dice_output = dice_metric_func(output_seg[-1], masks, activation=self.args.activation, is_average=False)
            if self.args.is_distributed_train:
                dice_output = self.reduce_tensor(dice_output.data)
            for i, dice_tmp in enumerate(dice_output):
                dice[i] += float(dice_tmp.item())

            total += len(images)

        val_loss /= total
        if self.is_print_out:
            self.logger.info('the loss of validation is {}'.format(val_loss))

        for idx, _ in enumerate(dice):
            dice[idx] /= total
            if self.is_print_out:
                self.logger.info('{} Dice:{:.6f}'.format(self.args.label[idx], dice[idx]))

        return val_loss, dice

    def test(self, test_dataset):
        """test procedure."""
        self.network.eval()

        out_image_dir = os.path.join(self.args.out_dir, 'image_visual')
        out_mask_dir = os.path.join(self.args.out_dir, 'mask_visual')
        if not os.path.exists(out_mask_dir) or not os.path.exists(out_image_dir):
            os.makedirs(out_mask_dir)
            os.makedirs(out_image_dir)
        csv_path = os.path.join(self.args.out_dir, 'infer_dice.csv')
        contents = ["SeriesUID"]
        for label in self.args.label:
            contents.append(label)
        write_csv(csv_path, contents, mul=False, mod="w")

        dice = [0.] * self.args.num_classes

        if self.args.is_save_script_model:
            example = torch.rand(1, 1, 128, 128, 128).float().cuda()
            example = example.half() if self.args.is_fp16 else example
            self.network = self.network.half() if self.args.is_fp16 else self.network
            self.convert_to_script_module(example)

        self.logger.info('starting test')
        self.logger.info('test samples per epoch: {}'.format(len(test_dataset)))
        for batchidx, data_dict in enumerate(test_dataset):
            self.logger.info('process: {}/{}'.format(batchidx, len(test_dataset)))
            uid = data_dict['uid']
            image_array = data_dict['image']
            mask_array = data_dict['mask'] if "mask" in data_dict else None

            if self.args.cuda:
                image = torch.from_numpy(image_array.copy()).float().cuda()
                image = image.half() if self.args.is_fp16 else image

            try:
                with torch.no_grad():
                    if self.args.is_save_script_model:
                        pred_mask_czyx = self.traced_script_module(image)
                    else:
                        pred_mask_czyx = self.network(image)
            except:
                self.logger.warning("out of memory, the shape of image is {}".format(image_array.shape))
                continue

            if data_dict["is_exist_mask"]:
                mask = torch.from_numpy(mask_array.copy()).float().cuda()
                pred_mask = pred_mask_czyx.float()
                compute_dice = DiceMetric()
                dice_output = compute_dice(pred_mask, mask)
                contents = [uid]
                for i, dice_tmp in enumerate(dice_output):
                    dice[i] += float(dice_tmp.item())
                    contents.append(dice_tmp.cpu().numpy())
                    self.logger.info("{}: {}".format(self.args.label[i], dice_tmp))
                write_csv(csv_path, contents, mul=False, mod="a+")

            if self.args.is_upsample_mask:
                pred_mask_czyx = F.interpolate(pred_mask_czyx, size=data_dict["image_shape_ori"],
                                               mode='trilinear', align_corners=True)
            pred_mask_czyx = F.sigmoid(pred_mask_czyx)
            pred_mask_czyx = pred_mask_czyx.cpu().numpy().squeeze()
            pred_mask_czyx[pred_mask_czyx >= 0.5] = 1
            pred_mask_czyx[pred_mask_czyx < 0.5] = 0
            pred_mask_czyx = pred_mask_czyx[np.newaxis, ] if self.args.num_classes == 1 else pred_mask_czyx
            pred_mask_zyx = np.zeros(pred_mask_czyx.shape[1:], dtype=np.int8)
            for i in range(self.args.num_classes):
                out_mask = pred_mask_czyx[i, ].squeeze()
                if self.args.is_post_process:
                    out_mask = smooth_mask(out_mask, area_least=1000, is_binary_close=False)
                pred_mask_zyx[out_mask != 0] = i + 1

            if self.args.is_save_mask:
                data_save = DataIO()
                image_path = os.path.join(out_image_dir, uid) + ".nii.gz"
                mask_path = os.path.join(out_mask_dir, uid) + ".nii.gz"
                data_save.save_medical_info_and_data(image_array.squeeze(), data_dict['origin'], data_dict["spacing"],
                                                     data_dict["direction"], image_path)
                data_save.save_medical_info_and_data(pred_mask_zyx, data_dict['origin'], data_dict["spacing"],
                                                     data_dict["direction"], mask_path)

        for idx, _ in enumerate(dice):
            dice[idx] /= len(test_dataset)
            self.logger.info('the average of {} Dice:{:.6f}'.format(self.args.label[idx], dice[idx]))