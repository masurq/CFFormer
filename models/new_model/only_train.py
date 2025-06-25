import os
import json
import random
import time
from interfaces import dataset_interface
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from utils import Core, average_meter, metrics, common
import utils.Info as Info
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.Logger import Logger
from torch.cuda import amp

from builder import EncoderDecoder as segmodel
from init_func import group_weight

# from datasets import Opt_SarDataset as Dataset
from datasets import VaihingenDataset as Dataset
# from datasets import PotsdamDataset as Dataset

# from config.opt_sar_config import config
# from config.opt_sar_config512 import config

from config.Vaihingen_config import config
# from config.Potsdam_config import config


class Trainer(object):
    def __init__(self):
        Info.info_logger(logger)
        logger.log('ZMSegmentation', '{}_{}_{}'.format(config.decoder, config.backbone, config.env), show_time=True,
                   print_type='print')

        rlog_env = '{}/{}_{}_{}_reduced'.format(log_env, config.decoder, config.backbone, config.env)
        common.check_path(rlog_env)
        self.rlogger_lr = Logger(
            '{}/{}_{}_{}_reduced_lr.rlg'.format(rlog_env, config.decoder, config.backbone, config.env),
            append=config.reduced_log_append)
        self.rlogger_train_loss = Logger(
            '{}/{}_{}_{}_reduced_train_loss.rlg'.format(rlog_env, config.decoder, config.backbone, config.env),
            append=config.reduced_log_append)
        self.rlogger_train_acc = Logger(
            '{}/{}_{}_{}_reduced_train_acc.rlg'.format(rlog_env, config.decoder, config.backbone, config.env),
            append=config.reduced_log_append)

        if config.use_cuda:
            logger.log('INFO', 'Numbers of GPUs:{}'.format(len(config.gpu_ids)), show_time=False)
        else:
            logger.log('INFO', 'Using CPU', show_time=False)

        self.use_aux_loss = False
        if config.decoder == 'UPernet' or config.decoder == 'deeplabv3+':
            self.use_aux_loss = True

        if config.random_seed >= 0:
            Core.init_random_seed(config.random_seed)
        else:
            random.seed()

        dataset_dict = dataset_interface(config)
        aug_params = dataset_dict['aug_params']

        dataset_name = config.dataset_name
        class_name = []
        if dataset_name == 'five':
            from utils import five_classes
            class_name = five_classes()
        if dataset_name == 'six':
            from utils import six_classes
            class_name = six_classes()
        if dataset_name == 'seven':
            from utils import seven_classes
            class_name = seven_classes()
        if dataset_name == 'eight':
            from utils import eight_classes
            class_name = eight_classes()
        if dataset_name == 'fifteen':
            from utils import fifteen_classes
            class_name = fifteen_classes()

        self.train_dataset = Dataset(class_name,
                                     root=config.train_data_root,
                                     mode='train',
                                     aug_params=aug_params)
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=config.train_batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=True,
                                       drop_last=True)

        logger.log('INFO', 'Number samples {}'.format(len(self.train_dataset)), show_time=False)

        self.num_classes = len(self.train_dataset.class_names)
        logger.log('INFO', '类别数：{}'.format(self.num_classes), show_time=False)
        logger.log('INFO', '{}'.format(self.train_dataset.class_names), show_time=False)

        self.class_loss_weight = torch.Tensor(config.class_loss_weight)
        if not config.loss_weight_with_background:
            self.class_loss_weight[0] = 0
            logger.log('INFO', 'loss_weight_without_background', show_time=False)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_loss_weight, reduction='mean',
                                             ignore_index=config.loss_ignore_index,
                                             label_smoothing=config.label_smoothing).cuda()
        # self.criterion = UnetFormerLoss(ignore_index=config.loss_ignore_index)

        model = segmodel(cfg=config, logger=logger, norm_layer=nn.BatchNorm2d)
        base_lr = config.lr
        params_list = []
        params_list = group_weight(params_list, model, nn.BatchNorm2d, base_lr)

        if config.resume_model:
            print('resume model', config.resume_model)
            state_dict = torch.load(config.resume_model_path)
            model.load_state_dict(state_dict)

            logger.log('INFO', '=====> resume model success from {}'.format(config.resume_model_path), show_time=True)

        if config.use_cuda:
            if len(config.gpu_ids) == 1:
                self.model = model.cuda()
            else:
                model = model.cuda()
                self.model = nn.DataParallel(model, device_ids=config.gpu_ids)

        if config.optimizer_name == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(params_list,
                                                  lr=base_lr,
                                                  weight_decay=config.weight_decay)
        if config.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(params_list,
                                              lr=base_lr,
                                              weight_decay=config.weight_decay)
        if config.optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(params_list,
                                               lr=base_lr,
                                               betas=(0.9, 0.999),
                                               weight_decay=config.weight_decay)
        if config.optimizer_name == 'Adan':
            from timm.optim.adan import Adan
            self.optimizer = Adan(params_list,
                                  lr=base_lr,
                                  weight_decay=config.weight_decay)

        if config.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(params=params_list,
                                             lr=base_lr,
                                             momentum=config.momentum,
                                             weight_decay=config.weight_decay)

        # 在训练最开始定义GradScalar的实例
        if config.use_amp:
            self.scaler = amp.GradScaler(enabled=True)
            logger.log('INFO', 'Automatically Mixed Precision', show_time=False)

        logger.log('INFO', '=====> model {}_{} =============== '.format(config.decoder, config.backbone),
                   show_time=True)

        self.max_iter = config.total_epoch * len(self.train_loader)

        if config.lr_scheduler == 'ReduceLROnPlateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(self.optimizer,
                                               mode=config.lr_reduce_mode,
                                               factor=config.lr_reduce_factor,
                                               patience=config.lr_reduce_patience,
                                               threshold=config.lr_reduce_threshold,
                                               cooldown=config.lr_reduce_cooldown,
                                               min_lr=config.lr_reduce_min_lr)
        elif config.lr_scheduler == 'CosineAnnealingLR':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.lr_T_max, eta_min=config.lr_eta_min)

        elif config.lr_scheduler == 'CosineAnnealingWarmRestarts':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=config.lr_T_0, T_mult=config.lr_T_mult,
                                                         eta_min=config.lr_eta_min)
        elif config.lr_scheduler == 'MultiStepLR':
            from torch.optim.lr_scheduler import MultiStepLR
            self.scheduler = MultiStepLR(self.optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma)

        elif config.lr_scheduler == 'StepLR':
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(self.optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

        elif config.lr_scheduler == 'ExponentialLR':
            from torch.optim.lr_scheduler import ExponentialLR
            self.scheduler = ExponentialLR(self.optimizer, gamma=config.lr_gamma)

        self.num_warmup_iter = 0
        if config.use_lr_warmup:
            self.num_warmup_iter = config.lr_warmup_epoch * len(self.train_loader) - 1

        self.best_acc = 0.0

    def training(self, epoch):
        logger.log('INFO', 'Epoch {:d} started.'.format(epoch + 1), show_time=True, print_type='print')
        self.model.train()
        if not (config.use_lr_warmup and epoch < config.lr_warmup_epoch):
            if self.scheduler and config.lr_scheduler != 'ReduceLROnPlateau':
                self.scheduler.step()

        train_loss = average_meter.AverageMeter()

        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.train_loader)
        for index, data in enumerate(tbar):
            imgs_opt = Variable(data[0])
            imgs_sar = Variable(data[1])
            masks = Variable(data[2])

            if config.use_cuda:
                imgs_opt = imgs_opt.cuda()
                imgs_sar = imgs_sar.cuda()
                masks = masks.cuda()

            self.optimizer.zero_grad()

            if config.use_amp:
                with amp.autocast(enabled=True):
                    outputs = self.model(imgs_opt, imgs_sar)

                    if len(outputs) == 2:
                        outputs_main, outputs_aux = outputs
                        main_loss = self.criterion(outputs_main, masks)
                        aux_loss = self.criterion(outputs_aux, masks)
                        loss = main_loss + aux_loss * config.aux_rate
                    else:
                        loss = self.criterion(outputs, masks)

                    # loss = self.criterion(outputs, masks)

                    if self.use_aux_loss:
                        preds = torch.argmax(outputs[0], 1)
                    else:
                        preds = torch.argmax(outputs, 1)

                    preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)

                curr_iter = epoch * len(self.train_loader) + index

                if config.use_lr_warmup and epoch < config.lr_warmup_epoch:
                    lr = config.lr_warmup_init * (config.lr / config.lr_warmup_init) ** (
                            curr_iter / self.num_warmup_iter)

                    for i in range(len(self.optimizer.param_groups)):
                        self.optimizer.param_groups[i]['lr'] = lr
                else:
                    lr = self.optimizer.param_groups[0]['lr']

                train_loss.update(loss, config.train_batch_size)
                writer.add_scalar('02 train_loss', train_loss.avg, curr_iter)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                outputs = self.model(imgs_opt, imgs_sar)

                if len(outputs) == 2:
                    outputs_main, outputs_aux = outputs
                    main_loss = self.criterion(outputs_main, masks)
                    aux_loss = self.criterion(outputs_aux, masks)
                    loss = main_loss + aux_loss * config.aux_rate
                else:
                    loss = self.criterion(outputs, masks)


                if self.use_aux_loss:
                    preds = torch.argmax(outputs[0], 1)
                else:
                    preds = torch.argmax(outputs, 1)

                preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)

                curr_iter = epoch * len(self.train_loader) + index

                if config.use_lr_warmup and epoch < config.lr_warmup_epoch:
                    lr = config.lr_warmup_init * (config.lr / config.lr_warmup_init) ** (
                            curr_iter / self.num_warmup_iter)

                    for i in range(len(self.optimizer.param_groups)):
                        self.optimizer.param_groups[i]['lr'] = lr
                else:
                    lr = self.optimizer.param_groups[0]['lr']

                train_loss.update(loss, config.train_batch_size)
                writer.add_scalar('02 train_loss', train_loss.avg, curr_iter)

                loss.backward()
                self.optimizer.step()

            tbar.set_description(
                'epoch {}/{}, training loss {}, with learning rate {}.'.format(epoch + 1, config.total_epoch,
                                                                               train_loss.avg,
                                                                               lr))

            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)

            conf_mat += metrics.confusion_matrix(pred=preds.flatten(),
                                                 label=masks.flatten(),
                                                 num_classes=self.num_classes)

        self.rlogger_lr.log('None',
                            'Learning_Rate/Value_{:e}/Epoch_{:d}'.format(lr, epoch + 1),
                            show_time=False, not_to_screen=True)

        train_acc, train_kappa, train_mean_IoU, train_mean_f1, train_mean_precision, train_mean_recall, \
        train_IoU, train_f1_score, train_precision, train_recall = metrics.evaluate(conf_mat, 1,
                                                                                    config.val_with_background)

        # 非warm_up且为ReduceLROnPlateau学习策略的学习率更新
        if not (config.use_lr_warmup and epoch < config.lr_warmup_epoch):
            if self.scheduler and config.lr_scheduler == 'ReduceLROnPlateau':
                if config.lr_reduce_metrics == 'Pixel_Acc':
                    self.scheduler.step(train_acc)
                elif config.lr_reduce_metrics == 'mIoU':
                    self.scheduler.step(train_mean_IoU)
                elif config.lr_reduce_metrics == 'Val_Loss':
                    self.scheduler.step(train_loss.avg)

        writer.add_scalar(tag='01 lr_per_epoch', scalar_value=lr, global_step=epoch + 1, walltime=None)
        writer.add_scalar(tag='03 train_loss_per_epoch', scalar_value=train_loss.avg, global_step=epoch + 1, walltime=None)
        writer.add_scalar(tag='04 train_oa', scalar_value=train_acc, global_step=epoch + 1, walltime=None)
        writer.add_scalar(tag='05 train_mean_IoU', scalar_value=train_mean_IoU, global_step=epoch + 1)

        _acc = 0.0
        if config.best_acc_metric == 'Pixel_Acc':
            _acc = train_acc
        elif config.best_acc_metric == 'Kappa':
            _acc = train_kappa
        elif config.best_acc_metric == 'mIoU':
            _acc = train_mean_IoU
        elif config.best_acc_metric == 'mF_score':
            _acc = train_mean_f1
        is_best = _acc > self.best_acc
        if is_best:
            self.best_acc = _acc

        logger.log('INFO', 'Epoch {:d}: Lr = {:f}'.format(epoch + 1, lr), show_time=True)
        logger.log('INFO', 'Epoch {:d}: Train Loss = {:f}'.format(epoch + 1, train_loss.avg), show_time=True)
        logger.log('INFO', 'Epoch {:d}: Train Acc = {:f}'.format(epoch + 1, train_acc), show_time=True)

        self.rlogger_train_loss.log('None',
                                    'Training_Loss/Value_{:f}/Epoch_{:d}'.format(train_loss.avg, epoch + 1),
                                    show_time=False, not_to_screen=True)
        self.rlogger_train_acc.log('None',
                                   'Training_Acc/Value_{:f}/Epoch_{:d}'.format(train_acc, epoch + 1), show_time=False,
                                   not_to_screen=True)

        model_name = '{}_{}_{}_epoch{}.pth'.format(config.decoder, config.backbone, config.env, epoch + 1)
        torch.save(self.model.state_dict(),
                   os.path.join(directory, model_name))
        logger.log('INFO', 'Saving Model', show_time=True, print_type='print')

        if is_best:
            torch.save(self.model.state_dict(), os.path.join(directory,
                                                             '{}_{}_{}_{}_best_model.pth'.format(config.decoder,
                                                                                                 config.backbone,
                                                                                                 config.env,
                                                                                                 config.best_acc_metric)))
            logger.log('INFO', 'Saving Best Model', show_time=True, print_type='print')


if __name__ == "__main__":

    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    directory = config.save_root + "/%s_%s/%s/%s/" % (config.decoder, config.backbone, config.env,
                                                      config.experiment_start_time)
    if not os.path.exists(directory):
        os.makedirs(directory)
    config_file = os.path.join(directory, 'config.json')
    with open(config_file, 'w') as file:
        json.dump(vars(config), file, indent=4)

    log_env = '{}/train_only_results/{}_{}_{}'.format(config.log_path, config.decoder, config.backbone, config.env)
    common.check_path(log_env)
    log_filename = '{}/{}_{}_{}_train.log'.format(log_env, config.decoder, config.backbone, config.env)
    logger = Logger(log_filename, append=config.log_append)

    writer = SummaryWriter(directory)
    trainer = Trainer()

    t_start = time.time()

    if config.resume_model:
        config.start_epoch = config.resume_start_epoch
        config.total_epoch = config.resume_total_epoch
    for epoch in range(config.start_epoch, config.total_epoch):
        trainer.training(epoch)

    t_end = time.time()
    elapsed_time = t_end - t_start
    hours = int(elapsed_time / 3600)
    minutes = int((elapsed_time % 3600) / 60)
    seconds = int((elapsed_time % 3600) % 60)
    logger.log('INFO', 'Total time consuming: {}时{}分{}秒'.format(hours, minutes, seconds),
               show_time=True)
