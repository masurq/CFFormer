import os
import shutil
import time
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
from utils import metrics
from utils.Common import common
from utils.Logger import Logger
import utils.Info as Info
from utils import colorize_mask

# from datasets.Opt_SarDataset import Opt_SarDataset as Dataset
from datasets.VaihingenDataset import VaihingenDataset as Dataset

# from config.opt_sar_config512 import config

from config.Vaihingen_config import config


from builder import EncoderDecoder as segmodel

if config.dataset_name == 'eight':
    from utils import eight_classes

    class_name = eight_classes()
elif config.dataset_name == 'six':
    from utils import six_classes

    class_name = six_classes()


def snapshot_forward(model, logger, dataloader, num_classes, save_path=None):
    model.eval()

    conf_mat = np.zeros((num_classes, num_classes)).astype(np.int64)
    tbar = tqdm(dataloader)
    for index, data in enumerate(tbar):
        with torch.no_grad():
            imgs = Variable(data[0])
            imgs_X = Variable(data[1])
            masks = Variable(data[2])
            output_names = data[3]

            imgs = imgs.cuda()
            imgs_X = imgs_X.cuda()
            masks = masks.cuda()

            outputs = model(imgs, imgs_X)
            preds = torch.argmax(outputs, 1)

            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)

            if save_path is not None:
                for i in range(masks.shape[0]):
                    pred_vis_pil = colorize_mask(preds[i], dataset_name=config.dataset_name)
                    pred_save_path = os.path.join(save_path, 'predict')

                    if not os.path.exists(pred_save_path):
                        os.makedirs(pred_save_path)
                    pred_vis_pil.save(os.path.join(pred_save_path, output_names[i].replace('.tif', '.png')))

            conf_mat += metrics.confusion_matrix(pred=preds.flatten(),
                                                 label=masks.flatten(),
                                                 num_classes=num_classes)

    test_acc, test_kappa, test_mean_IoU, test_mean_f1, test_mean_precision, test_mean_recall, \
        test_IoU, test_f1_score, test_precision, test_recall = metrics.evaluate(conf_mat, 1, config.val_with_background)

    logger.log('None', ' Pixel Acc:      {}'.format(str(test_acc)), show_time=False)
    logger.log('None', ' Kappa:          {}'.format(str(test_kappa)), show_time=False)
    logger.log('None', ' Mean IoU:       {}'.format(str(test_mean_IoU)), show_time=False)
    logger.log('None', ' Mean F1 score:  {}'.format(str(test_mean_f1)), show_time=False)
    logger.log('None', ' Mean Precision: {}'.format(str(test_mean_precision)), show_time=False)
    logger.log('None', ' Mean Recall:    {}'.format(str(test_mean_recall)), show_time=False)

    table = PrettyTable(["序号", "名称", "IoU", "F1 Score", "Precision", "Recall"])
    _class_name = class_name
    if config.val_with_background == 0:
        _class_name = _class_name[1:]
    for i in range(len(_class_name)):
        table.add_row(
            [i, _class_name[i], '{:.8f}'.format(test_IoU[i]), '{:.8f}'.format(test_f1_score[i]),
             '{:.8f}'.format(test_precision[i]), '{:.8f}'.format(test_recall[i])])

    logger.log('None', '{}'.format(table), show_time=False)


def reference():
    log_env = '{}/test_results/{}_{}_{}'.format(config.log_path, config.decoder, config.backbone, config.env)
    common.check_path(log_env)
    log_filename = '{}/{}_{}_{}_test.log'.format(log_env, config.decoder, config.backbone, config.env)
    logger = Logger(log_filename, append=config.log_append)

    Info.info_logger(logger)
    logger.log('ZMSegmentation', '{}_{}_{}'.format(config.decoder, config.backbone, config.env), show_time=True,
               print_type='print')

    if config.use_cuda:
        logger.log('INFO', 'Numbers of GPUs:{}'.format(len(config.gpu_ids)), show_time=False)
    else:
        logger.log('INFO', 'Using CPU', show_time=False)

    dataset = Dataset(class_name=class_name, root=config.test_data_root, mode='test')

    dataloader = DataLoader(dataset=dataset, batch_size=config.test_batch_size, shuffle=False,
                            num_workers=config.num_workers)

    logger.log('INFO', '类别数：{}'.format(len(class_name)), show_time=False)
    logger.log('INFO', '{}'.format(class_name), show_time=False)

    model = segmodel(cfg=config, logger=logger, norm_layer=nn.BatchNorm2d)

    logger.log('INFO', '=====> model {}_{} =============== '.format(config.decoder, config.backbone),
               show_time=True)

    state_dict = torch.load(config.model_path)
    logger.log('INFO', 'Loading Model from {}'.format(config.model_path), show_time=False, print_type='print')
    model.load_state_dict(state_dict)

    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)

    logger.log('INFO', '===============> load model successfully', show_time=True, print_type='print')
    model = model.cuda()

    if not config.pred_path:
        pass
    elif not os.path.exists(config.pred_path):
        os.makedirs(config.pred_path)
    else:
        shutil.rmtree(config.pred_path)
        os.makedirs(config.pred_path)

    t_start = time.time()

    snapshot_forward(model, logger, dataloader, len(class_name), config.pred_path)

    t_end = time.time()
    elapsed_time = t_end - t_start
    hours = int(elapsed_time / 3600)
    minutes = int((elapsed_time % 3600) / 60)
    seconds = int((elapsed_time % 3600) % 60)
    logger.log('INFO', 'Test task finished.', show_time=True, print_type='print')
    logger.log('INFO', 'Total time consuming: {}时{}分{}秒'.format(hours, minutes, seconds),
               show_time=True)


if __name__ == '__main__':
    reference()
