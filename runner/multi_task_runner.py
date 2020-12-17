
import os
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel

from utils.logger import get_logger
from utils.csv_tools import write_csv
from runner.loss import DiceLoss, BCELoss, ClsSegLoss
from runner.base import BaseModel, DataLoaderX
from runner.metric import DiceMetric, LabeledDiceMetric
from utils.mask_utils import smooth_mask, extract_candidates_bbox
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

        sample_weights = [1, 6]
        weights = []
        for i, type_ in enumerate(train_dataset):
            if i > len(train_dataset) - 1:
                break
            weights.append(sample_weights[type_])
        weights = np.array(weights)
        sampler_weights = torch.from_numpy(weights).double()
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sampler_weights, int(weights.shape[0] * 1))
        shuffle = False

        self.train_dataset.is_sample = False
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

        if torch.cuda.device_count() > 1:
            self.network = torch.nn.DataParallel(self.network)

        best_dice = 0
        best_recall = 0
        train_epoch = 0
        val_epoch = 0

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            if self.is_print_out:
                self.logger.info('starting training epoch {}'.format(epoch))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.get_lr(epoch, self.args.epochs, self.lr)

            self.network.train()

            start_time = datetime.datetime.now()

            dice = [0.] * self.args.num_classes
            total = 0.
            correct = 0.

            for index, (images, masks, labels) in enumerate(train_loader):
                if self.args.cuda:
                    images, masks, labels = images.cuda(), masks.cuda(), labels.cuda()

                self.optimizer.zero_grad()
                if self.args.task == "single":
                    output_seg = self.network(Variable(images))
                elif self.args.task == "multi":
                    output_cls, output_seg = self.network(Variable(images))

                if self.args.task == "multi":
                    loss_cls = 0
                    loss_seg = 0
                    loss_func = ClsSegLoss()
                    for i in range(len(output_cls)):
                        cls_loss, seg_loss = loss_func(torch.squeeze(output_cls[i]), output_seg[i], labels, masks,
                                                       is_average=True)
                        loss_cls += cls_loss
                        loss_seg += seg_loss
                    loss_cls /= len(output_cls)
                    loss_seg /= len(output_seg)
                    if epoch < self.args.epochs * 0.66:
                        loss = loss_cls + loss_seg * 0.6
                    else:
                        loss = loss_cls + loss_seg
                else:
                    loss_seg = 0
                    dice_loss_func = DiceLoss()
                    for i, _ in enumerate(output_seg):
                        dice_loss = dice_loss_func(output_seg[i], masks, activation=self.args.activation)
                        loss_seg += dice_loss
                    loss_seg /= len(output_seg)
                    loss = loss_seg

                if self.args.is_apex_train:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                predicted_cls = torch.squeeze(output_cls[-1])
                predicted_cls[predicted_cls < 0.5] = 0
                predicted_cls[predicted_cls >= 0.5] = 1
                correct_cls = torch.sum(predicted_cls.long() == labels)
                correct += correct_cls

                dice_metric_func = LabeledDiceMetric()
                dice_output = dice_metric_func(output_seg[-1], masks, predicted_cls, labels, is_average=False)

                for i, dice_tmp in enumerate(dice_output):
                    dice[i] += float(dice_tmp.item())
                total += len(images)

                if index > 0 and index % self.args.log_interval == 0:
                    train_epoch += 1
                    if self.is_print_out:
                        self.logger.info('Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(
                        epoch, self.args.epochs, index * len(images), len(train_loader.dataset), 100. * index / len(train_loader)))
                        train_writer.add_scalar('Train/Loss', loss.item(), train_epoch)
                        train_writer.add_scalar('Train/Cls_loss', loss_cls.item(), train_epoch)
                        train_writer.add_scalar('Train/Seg_loss', loss_seg.item(), train_epoch)
                        self.logger.info('Loss:{:.6f}'.format(loss.item()))
                        self.logger.info('Cls_loss:{:.6f}'.format(loss_cls.item()))
                        self.logger.info('Seg_loss:{:.6f}'.format(loss_seg.item()))

                        if self.args.task == "multi":
                            train_writer.add_scalar('Train/Acc', 100.0 * float(correct) / total, train_epoch)
                            self.logger.info('Acc:{:.6f}'.format(100.0 * float(correct) / total))

                        for i, dice_label in enumerate(dice):
                            dice_ind = dice_label / total
                            train_writer.add_scalars('Train/Dice', {self.args.label[i]: dice_ind}, train_epoch)
                            self.logger.info('{} Dice:{:.6f}'.format(self.args.label[i], dice_ind))

            val_acc, all_loss, val_recall, val_dice = self.validate(val_dataset)
            val_epoch += 1
            if self.is_print_out:
                val_writer.add_scalar('Val/Loss', all_loss["loss"], val_epoch)
                val_writer.add_scalar('Val/Cls_loss', all_loss["cls_loss"], val_epoch)
                val_writer.add_scalar('Val/Seg_loss', all_loss["seg_loss"], val_epoch)
                val_writer.add_scalar('Val/Acc', val_acc, val_epoch)

                labels = ['normal', 'unnormal']
                for label in range(self.args.num_classes+1):
                    val_writer.add_scalars('Val/recall', {labels[label]: val_recall[label]}, val_epoch)

                total_dice = 0
                for i, _ in enumerate(val_dice):
                    val_writer.add_scalars('Val/Dice', {self.args.label[i]: val_dice[i]}, val_epoch)
                    total_dice += val_dice[i]
                total_dice /= len(val_dice)

                if val_recall[1] > best_recall:
                    self.metric_non_improve_epoch = 0
                    best_recall = val_recall[1]
                    self.save_weights(epoch, self.network.state_dict(), self.optimizer.state_dict())
                else:
                    self.metric_non_improve_epoch += 1

                # if total_dice > best_dice:
                #     self.metric_non_improve_epoch = 0
                #     best_dice = total_dice
                #     self.save_weights(epoch, self.network.state_dict(), self.optimizer.state_dict())
                # else:
                #     self.metric_non_improve_epoch += 1

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
        total = 0.
        correct = 0.
        val_loss = 0.
        all_cls_loss = 0.
        all_seg_loss = 0.

        tp = [0.] * (self.args.num_classes + 1)
        tpfp = [0.] * (self.args.num_classes + 1)
        tpfn = [0.] * (self.args.num_classes + 1)
        precision = [0.] * (self.args.num_classes + 1)
        recall = [0.] * (self.args.num_classes + 1)
        f1 = [0.] * (self.args.num_classes + 1)

        val_loader = DataLoaderX(
            dataset=self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.n_workers,
            drop_last=False,
            pin_memory=True
        )
        if self.is_print_out:
            self.logger.info('starting validation')
            self.logger.info('val samples per epoch: {}'.format(len(self.val_dataset)))
        for images, masks, labels in val_loader:
            if self.args.cuda:
                images, masks, labels = images.cuda(), masks.cuda(), labels.cuda()

            with torch.no_grad():
                if self.args.task == "single":
                    output_seg = self.network(Variable(images))
                elif self.args.task == "multi":
                    output_cls, output_seg = self.network(Variable(images))

            if self.args.task == "multi":
                loss_cls = 0
                loss_seg = 0
                loss_func = ClsSegLoss()
                for i, _ in enumerate(output_seg):
                    cls_loss, seg_loss = loss_func(torch.squeeze(output_cls[i]), output_seg[i], labels, masks,
                                                   is_average=False)
                    loss_cls += cls_loss
                    loss_seg += seg_loss
                loss_cls /= len(output_cls)
                loss_seg /= len(output_seg)
                loss = loss_cls.item() + loss_seg.item()
            else:
                loss_seg = 0
                dice_loss_func = DiceLoss()
                for i, _ in enumerate(output_seg):
                    dice_loss = dice_loss_func(output_seg[i], masks, activation=self.args.activation)
                    loss_seg += dice_loss
                loss_seg /= len(output_seg)
                loss = loss_seg.item()

            val_loss += loss
            all_cls_loss += loss_cls.item()
            all_seg_loss += loss_seg.item()

            predicted_cls = torch.squeeze(output_cls[-1])
            predicted_cls[predicted_cls < 0.5] = 0
            predicted_cls[predicted_cls >= 0.5] = 1
            correct_cls = torch.sum(predicted_cls.long() == labels)
            correct += correct_cls

            for label in range(self.args.num_classes + 1):
                t_labels = labels == label
                p_labels = predicted_cls == label
                tp[label] += torch.sum(t_labels == (p_labels * 2 - 1))
                tpfp[label] += torch.sum(p_labels)
                tpfn[label] += torch.sum(t_labels)

            dice_metric_func = LabeledDiceMetric()
            dice_output = dice_metric_func(output_seg[-1], masks, predicted_cls, labels, is_average=False)
            for i, dice_tmp in enumerate(dice_output):
                dice[i] += float(dice_tmp.item())

            total += len(images)

        for label in range(self.args.num_classes + 1):
            precision[label] = tp[label].float() / (tpfp[label].float() + 1e-8)
            recall[label] = tp[label].float() / (tpfn[label].float() + 1e-8)
            f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label] + 1e-8)

        acc = 100.0 * float(correct) / total
        val_loss /= total
        all_cls_loss /= total
        all_seg_loss /= total
        if self.is_print_out:
            self.logger.info('the loss of validation is {:.6f}'.format(val_loss))
            self.logger.info('the cls loss of validation is {:.6f}'.format(all_cls_loss))
            self.logger.info('the seg loss of validation is {:.6f}'.format(all_seg_loss))
            self.logger.info('the acc of validation is {:.6f}'.format(acc))

            labels = ['normal', 'unnormal']
            for label in range(self.args.num_classes + 1):
                self.logger.info('{}:  \t Precision: {:.2f},  Recall: {:.2f},  F1: {:.2f}'.format(
                    labels[label],
                    precision[label],
                    recall[label],
                    f1[label]
                ))

            for idx, _ in enumerate(dice):
                dice[idx] /= total
                self.logger.info('{} Dice:{:.6f}'.format(self.args.label[idx], dice[idx]))

        return acc, {"loss": val_loss, "cls_loss": all_cls_loss, "seg_loss": all_seg_loss}, recall, dice

    def test(self, test_dataset):
        """test procedure."""
        self.network.eval()
        self.network = self.network.half()

        csv_path = os.path.join(self.args.out_dir, 'infer_result.csv')
        contents = ['uid', 'coordX', 'coordY', 'coordZ',
                    'detector_diameterX', 'detector_diameterY', 'detector_diameterZ',
                    'boneNo', 'boneType', 'frac_type', 'det_probability', 'candidate_type']
        write_csv(csv_path, contents, mul=False, mod="w")

        self.logger.info('starting test')
        self.logger.info('the number of test dataset: {}'.format(len(test_dataset)))

        for data_dict in tqdm(test_dataset):
            uid = data_dict["uid"]
            candidates = data_dict["candidates"]
            images = []
            for candidate in candidates:
                image = candidate["image"]
                images.append(image)
            images = np.stack(images, axis=0)
            images = torch.from_numpy(images).float().half()
            if self.args.cuda:
                images = images.cuda()

            with torch.no_grad():
                output_cls, output_seg = self.network(images)
            pred_cls = output_cls[-1]
            pred_cls = pred_cls.cpu().numpy().squeeze()
            pred_cls[pred_cls < 0.3] = 0
            pred_cls[pred_cls >= 0.3] = 1

            pred_seg = output_seg[-1]
            pred_seg = F.sigmoid(pred_seg)
            pred_seg = pred_seg.cpu().numpy().squeeze()
            pred_seg[pred_seg < 0.5] = 0
            pred_seg[pred_seg >= 0.5] = 1
            for i in range(len(candidates)):
                if pred_cls[i] == 0:
                    continue
                out_candidate_seg = pred_seg[i, ...].squeeze()
                out_candidates = extract_candidates_bbox(out_candidate_seg, area_least=25)
                zoom_factor = candidates[i]["zoom_factor"]
                crop_bbox = candidates[i]["crop_bbox"]
                candidate_type = candidates[i]["candidate_type"]
                for out_candidate in out_candidates:
                    centroid = out_candidate["centroid"]
                    bbox = out_candidate["bbox"]
                    raw_centroid = [centroid[j] * zoom_factor[j] + crop_bbox[j] for j in range(3)]
                    raw_diameter = [(bbox[3] - bbox[0]) * zoom_factor[0],
                                    (bbox[4] - bbox[1]) * zoom_factor[1],
                                    (bbox[5] - bbox[2]) * zoom_factor[2]]
                    contents = [uid, raw_centroid[2], raw_centroid[1], raw_centroid[0],
                                raw_diameter[2], raw_diameter[1], raw_diameter[0],
                                -1, -1, -1, 1, candidate_type]
                    write_csv(csv_path, contents, mul=False, mod="a+")