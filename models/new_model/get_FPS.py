import time
import argparse

import torch
from torchvision import transforms
from PIL import Image
from utils.Common import common
from utils.Logger import Logger
import utils.Info as Info

from config.opt_sar_config512 import config
# from config.Vaihingen_config import config
# from config.Potsdam_config import config

from builder import EncoderDecoder as segmodel

if config.dataset_name == 'eight':
    from utils import eight_classes

    class_name = eight_classes()

elif config.dataset_name == 'six':
    from utils import six_classes

    class_name = six_classes()


def parse_args():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument('--test_interval', type=int, default=1000, help='inference_num')

    parser.add_argument('--base-size', type=int, default=512, help='base image size')

    args = parser.parse_args()
    return args


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def get_FPS():
    args = parse_args()
    log_env = '{}/FPS/{}_{}_{}'.format(config.log_path, config.decoder, config.backbone, config.env)
    common.check_path(log_env)
    log_filename = '{}/{}_{}_{}_fps.log'.format(log_env, config.decoder, config.backbone, config.env)
    logger = Logger(log_filename, append=config.log_append)

    Info.info_logger(logger)
    logger.log('ZMSegmentation', '{}_{}_{}'.format(config.decoder, config.backbone, config.env), show_time=True,
               print_type='print')

    weights_path = config.model_path
    img_path1 = config.test_data_root + '/opt/NH49E001014_0_0.tif'
    img_path2 = config.test_data_root + '/sar/NH49E001014_0_0.tif'

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.log('INFO', 'using {} device.'.format(device))

    # create model
    model = segmodel(cfg=config, logger=logger)

    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    model.to(device)

    # load image
    original_img1 = Image.open(img_path1)
    original_img2 = Image.open(img_path2)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(args.base_size),
                                         transforms.ToTensor()])

    img1 = data_transform(original_img1)
    img2 = data_transform(original_img2)
    # expand batch dimension
    img1 = torch.unsqueeze(img1, dim=0)
    img2 = torch.unsqueeze(img2, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img1.shape[-2:]
        init_img1 = torch.zeros((1, 4, img_height, img_width), device=device)
        init_img2 = torch.zeros((1, 1, img_height, img_width), device=device)
        model(init_img1, init_img2)  # 模型预热

        t_start = time_synchronized()
        for _ in range(args.test_interval):
            with torch.no_grad():
                output = model(img1.to(device), img2.to(device))
        t_end = time_synchronized()
        tact_time = (t_end - t_start) / args.test_interval
        logger.log('INFO', 'inference time: {} seconds, {} FPS, @batch_size 1'.format(tact_time, 1 / tact_time),
                   show_time=True)


if __name__ == '__main__':
    get_FPS()
