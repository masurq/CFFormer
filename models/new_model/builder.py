import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from functools import partial

from models.new_model.init_func import init_weight, nostride_dilate, patch_first_conv_mit, patch_first_conv_swin, \
    patch_first_conv_biformer, patch_first_conv_SMT, patch_first_conv_DilateFormer, patch_first_conv_resnet, \
    patch_first_conv_cswin, patch_first_conv_mobilenet, patch_first_conv_EMO, patch_first_conv_single_biformer

from utils.Common import common
from utils.Logger import Logger

from thop import clever_format, profile
from config.opt_sar_config512 import config


class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, logger=None, norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        self.logger = logger
        # import backbone and decoder
        if cfg.backbone == 'swin_t':
            logger.log('INFO', 'Using backbone: Swin-Transformer-Tiny', show_time=False)
            from models.new_model.encoders.Transformer.dual_swin import swin_t as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'swin_s':
            logger.log('INFO', 'Using backbone: Swin-Transformer-Small', show_time=False)
            from models.new_model.encoders.Transformer.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'swin_b':
            logger.log('INFO', 'Using backbone: Swin-Transformer-Base', show_time=False)
            from models.new_model.encoders.Transformer.dual_swin import swin_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'swin_l':
            logger.log('INFO', 'Using backbone: Swin-Transformer-Large', show_time=False)
            from models.new_model.encoders.Transformer.dual_swin import swin_l as backbone
            self.channels = [192, 384, 768, 1536]
            self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'CSwin_t':
            logger.log('INFO', 'Using backbone: CSwin-Tiny', show_time=False)
            from models.new_model.encoders.Transformer.dual_cswin import cswin_t as backbone
            self.channels = [64, 128, 256, 512]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'CSwin_s':
            logger.log('INFO', 'Using backbone: CSwin-Small', show_time=False)
            from models.new_model.encoders.Transformer.dual_cswin import cswin_s as backbone
            self.channels = [64, 128, 256, 512]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'CSwin_b':
            logger.log('INFO', 'Using backbone: CSwin-Base', show_time=False)
            from models.new_model.encoders.Transformer.dual_cswin import cswin_b as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'mit_b5':
            logger.log('INFO', 'Using backbone: Segformer-B5', show_time=False)
            from models.new_model.encoders.Transformer.dual_segformer import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b4':
            logger.log('INFO', 'Using backbone: Segformer-B4', show_time=False)
            from models.new_model.encoders.Transformer.dual_segformer import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b3':
            logger.log('INFO', 'Using backbone: Segformer-B3', show_time=False)
            from models.new_model.encoders.Transformer.dual_segformer import mit_b3 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.log('INFO', 'Using backbone: Segformer-B2', show_time=False)
            from models.new_model.encoders.Transformer.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b1':
            logger.log('INFO', 'Using backbone: Segformer-B1', show_time=False)
            from models.new_model.encoders.Transformer.dual_segformer import mit_b1 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b0':
            logger.log('INFO', 'Using backbone: Segformer-B0', show_time=False)
            self.channels = [32, 64, 160, 256]
            from models.new_model.encoders.Transformer.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'pvt_t':
            logger.log('INFO', 'Using backbone: PVT-Tiny', show_time=False)
            from models.new_model.encoders.Transformer.PVT.PVTv1 import pvt_tiny as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'pvt_s':
            logger.log('INFO', 'Using backbone: PVT-Small', show_time=False)
            from models.new_model.encoders.Transformer.PVT.PVTv1 import pvt_small as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'pvt_m':
            logger.log('INFO', 'Using backbone: PVT-Medium', show_time=False)
            from models.new_model.encoders.Transformer.PVT.PVTv1 import pvt_medium as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'pvt_l':
            logger.log('INFO', 'Using backbone: PVT-Large', show_time=False)
            from models.new_model.encoders.Transformer.PVT.PVTv1 import pvt_large as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'pvt_v2_b0':
            logger.log('INFO', 'Using backbone: PVTv2-B0', show_time=False)
            self.channels = [32, 64, 160, 256]
            from models.new_model.encoders.Transformer.PVT.PVTv2 import pvt_v2_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'pvt_v2_b1':
            logger.log('INFO', 'Using backbone: PVTv2-B1', show_time=False)
            from models.new_model.encoders.Transformer.PVT.PVTv2 import pvt_v2_b1 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'pvt_v2_b2':
            logger.log('INFO', 'Using backbone: PVTv2-B2', show_time=False)
            from models.new_model.encoders.Transformer.PVT.PVTv2 import pvt_v2_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'pvt_v2_b3':
            logger.log('INFO', 'Using backbone: PVTv2-B3', show_time=False)
            from models.new_model.encoders.Transformer.PVT.PVTv2 import pvt_v2_b3 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'pvt_v2_b4':
            logger.log('INFO', 'Using backbone: PVTv2-B4', show_time=False)
            from models.new_model.encoders.Transformer.PVT.PVTv2 import pvt_v2_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'pvt_v2_b5':
            logger.log('INFO', 'Using backbone: PVTv2-B5', show_time=False)
            from models.new_model.encoders.Transformer.PVT.PVTv2 import pvt_v2_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'dilateformer_t':
            logger.log('INFO', 'Using backbone: DilateFormer-Tiny', show_time=False)
            self.channels = [144, 288, 576, 576]
            from models.new_model.encoders.Transformer.dual_dilateformer import dilateformer_t as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'dilateformer_s':
            logger.log('INFO', 'Using backbone: DilateFormer-Small', show_time=False)
            self.channels = [144, 288, 576, 576]
            from models.new_model.encoders.Transformer.dual_dilateformer import dilateformer_s as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'dilateformer_b':
            logger.log('INFO', 'Using backbone: DilateFormer-Base', show_time=False)
            self.channels = [192, 384, 768, 768]
            from models.new_model.encoders.Transformer.dual_dilateformer import dilateformer_b as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'biformer_t':
            logger.log('INFO', 'Using backbone: Biformer-Tiny', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.Transformer.BiFormer.dual_biformer import biformer_t as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'biformer_s':
            logger.log('INFO', 'Using backbone: Biformer-Small', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.Transformer.BiFormer.dual_biformer import biformer_s as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'biformer_b':
            logger.log('INFO', 'Using backbone: Biformer-Base', show_time=False)
            self.channels = [96, 192, 384, 768]
            from models.new_model.encoders.Transformer.BiFormer.dual_biformer import biformer_b as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        # elif cfg.backbone == 'biformer_t':
        #     logger.log('INFO', 'Using backbone: Biformer-Tiny', show_time=False)
        #     self.channels = [64, 128, 256, 512]
        #     from models.new_model.encoders.Transformer.BiFormer.biformer import biformer_t as backbone
        #     self.backbone = backbone(norm_fuse=norm_layer)
        # elif cfg.backbone == 'biformer_s':
        #     logger.log('INFO', 'Using backbone: Biformer-Small', show_time=False)
        #     self.channels = [64, 128, 256, 512]
        #     from models.new_model.encoders.Transformer.BiFormer.biformer import biformer_s as backbone
        #     self.backbone = backbone(norm_fuse=norm_layer)
        # elif cfg.backbone == 'biformer_b':
        #     logger.log('INFO', 'Using backbone: Biformer-Base', show_time=False)
        #     self.channels = [96, 192, 384, 768]
        #     from models.new_model.encoders.Transformer.BiFormer.biformer import biformer_b as backbone
        #     self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'uniformer_s':
            logger.log('INFO', 'Using backbone: Uniformer-Small', show_time=False)
            from models.new_model.encoders.Transformer.dual_uniformer import uniformer_s as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'uniformer_s_plus':
            logger.log('INFO', 'Using backbone: Uniformer-Small-Plus', show_time=False)
            from models.new_model.encoders.Transformer.dual_uniformer import uniformer_s_plus as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'uniformer_s_plus_64':
            logger.log('INFO', 'Using backbone: Uniformer-Small-Plus-dim64', show_time=False)
            from models.new_model.encoders.Transformer.dual_uniformer import uniformer_s_plus_dim64 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'uniformer_b':
            logger.log('INFO', 'Using backbone: Uniformer-Base', show_time=False)
            from models.new_model.encoders.Transformer.dual_uniformer import uniformer_b as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'uniformer_b_ls':
            logger.log('INFO', 'Using backbone: Uniformer-Base-LargeScale', show_time=False)
            from models.new_model.encoders.Transformer.dual_uniformer import uniformer_b_ls as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'vitaev2_s':
            logger.log('INFO', 'Using backbone: ViTAEv2-S', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.Transformer.ViTAE.dual_ViTAEv2 import ViTAEv2_S as backbone
            self.backbone = backbone(in_chans1=3, in_chans2=1, norm_fuse=norm_layer)
        elif cfg.backbone == 'vitaev2_48m':
            logger.log('INFO', 'Using backbone: ViTAEv2-48M', show_time=False)
            self.channels = [96, 192, 384, 768]
            from models.new_model.encoders.Transformer.ViTAE.dual_ViTAEv2 import ViTAEv2_48M as backbone
            self.backbone = backbone(in_chans1=3, in_chans2=1, norm_fuse=norm_layer)
        elif cfg.backbone == 'vitaev2_b':
            logger.log('INFO', 'Using backbone: ViTAEv2-B', show_time=False)
            self.channels = [128, 256, 512, 1024]
            from models.new_model.encoders.Transformer.ViTAE.dual_ViTAEv2 import ViTAEv2_B as backbone
            self.backbone = backbone(in_chans1=3, in_chans2=1, norm_fuse=norm_layer)

        elif cfg.backbone == 'SMT_t':
            logger.log('INFO', 'Using backbone: SMT-Tiny', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.CNN_Transformer.dual_SMT import SMT_t as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'SMT_s':
            logger.log('INFO', 'Using backbone: SMT-Small', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.CNN_Transformer.dual_SMT import SMT_s as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'SMT_b':
            logger.log('INFO', 'Using backbone: SMT-Base', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.CNN_Transformer.dual_SMT import SMT_b as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'SMT_l':
            logger.log('INFO', 'Using backbone: SMT-Large', show_time=False)
            self.channels = [96, 192, 384, 768]
            from models.new_model.encoders.CNN_Transformer.dual_SMT import SMT_l as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'emo_1m':
            logger.log('INFO', 'Using backbone: EMO-1M', show_time=False)
            self.channels = [32, 48, 80, 168]
            from models.new_model.encoders.CNN_Transformer.dual_EMO import emo_1M as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'emo_2m':
            logger.log('INFO', 'Using backbone: EMO-2M', show_time=False)
            self.channels = [32, 48, 120, 200]
            from models.new_model.encoders.CNN_Transformer.dual_EMO import emo_2M as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'emo_5m':
            logger.log('INFO', 'Using backbone: EMO-5M', show_time=False)
            self.channels = [48, 72, 160, 288]
            from models.new_model.encoders.CNN_Transformer.dual_EMO import emo_5M as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'resnet18':
            logger.log('INFO', 'Using backbone: resnet18', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.CNN.dual_resnet import resnet18 as backbone
            self.backbone = backbone(deep_stem=False)
            # self.dilate = 2
            # for m in self.backbone.layer4.children():
            #     m.apply(partial(nostride_dilate, dilate=self.dilate))
            #     self.dilate *= 2
        elif cfg.backbone == 'resnet34':
            logger.log('INFO', 'Using backbone: resnet34', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.CNN.dual_resnet import resnet34 as backbone
            self.backbone = backbone(deep_stem=False)
            self.dilate = 2
            for m in self.backbone.layer4.children():
                m.apply(partial(nostride_dilate, dilate=self.dilate))
                self.dilate *= 2
        elif cfg.backbone == 'resnet50':
            logger.log('INFO', 'Using backbone: resnet50', show_time=False)
            self.channels = [256, 512, 1024, 2048]
            from models.new_model.encoders.CNN.dual_resnet import resnet50 as backbone
            self.backbone = backbone(deep_stem=False)
            self.dilate = 2
            for m in self.backbone.layer4.children():
                m.apply(partial(nostride_dilate, dilate=self.dilate))
                self.dilate *= 2
        elif cfg.backbone == 'resnet101_deep':
            logger.log('INFO', 'Using backbone: resnet101_deep', show_time=False)
            self.channels = [256, 512, 1024, 2048]
            from models.new_model.encoders.CNN.dual_resnet import resnet101 as backbone
            self.backbone = backbone(deep_stem=True)
            self.dilate = 2
            for m in self.backbone.layer4.children():
                m.apply(partial(nostride_dilate, dilate=self.dilate))
                self.dilate *= 2
        elif cfg.backbone == 'resnet101':
            logger.log('INFO', 'Using backbone: resnet101', show_time=False)
            self.channels = [256, 512, 1024, 2048]
            from models.new_model.encoders.CNN.dual_resnet import resnet101 as backbone
            self.backbone = backbone(deep_stem=False)
            self.dilate = 2
            for m in self.backbone.layer4.children():
                m.apply(partial(nostride_dilate, dilate=self.dilate))
                self.dilate *= 2

        elif cfg.backbone == 'convnext_t':
            logger.log('INFO', 'Using backbone: ConvNext-Tiny', show_time=False)
            from models.new_model.encoders.CNN.dual_convnext import convnext_t as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'convnext_s':
            logger.log('INFO', 'Using backbone: ConvNext-Small', show_time=False)
            from models.new_model.encoders.CNN.dual_convnext import convnext_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'convnext':
            logger.log('INFO', 'Using backbone: ConvNext-Base', show_time=False)
            from models.new_model.encoders.CNN.dual_convnext import convnext_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'convnext_l':
            logger.log('INFO', 'Using backbone: ConvNext-Large', show_time=False)
            from models.new_model.encoders.CNN.dual_convnext import convnext_l as backbone
            self.channels = [192, 384, 768, 1536]
            self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'mobilenet_v2':
            logger.log('INFO', 'Using backbone: MobileNet-V2', show_time=False)
            from models.new_model.encoders.CNN.dual_mobilenetv2 import DualMobileNetV2 as backbone
            self.channels = [24, 32, 96, 320]
            self.backbone = backbone(norm_fuse=norm_layer)

        else:
            logger.log('INFO', 'Using backbone: Biformer-Tiny', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.Transformer.BiFormer.dual_biformer import biformer_t as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if cfg.decoder == 'MLPAlignDecoder':
            logger.log('INFO', 'Using MLP Aligned Decoder', show_time=False)
            from decoders.MLPAlignDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes,
                                           norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        elif cfg.decoder == 'AlignedFPN':
            logger.log('INFO', 'Using Decoder: AlignedFPN', show_time=False)
            from decoders.fpn_head import AlignedFPNDecoder
            self.decode_head = AlignedFPNDecoder(self.channels, cfg.num_classes)

        # b0/b1 256, b2-b5 768
        elif cfg.decoder == 'MLPDecoder':
            logger.log('INFO', 'Using MLP Decoder', show_time=False)
            from decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes,
                                           norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        elif cfg.decoder == 'UPernet':
            logger.log('INFO', 'Using UPernet Decoder', show_time=False)
            from decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer,
                                        channels=512)
            from decoders.fcnhead import AuxFCNHead
            self.aux_index = 2
            self.aux_head = AuxFCNHead(self.channels[2], num_classes=cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == 'deeplabv3+':
            logger.log('INFO', 'Using Decoder: DeepLabV3+', show_time=False)
            from decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from decoders.fcnhead import AuxFCNHead
            self.aux_index = 2
            self.aux_head = AuxFCNHead(self.channels[2], num_classes=cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == 'fapn':
            logger.log('INFO', 'Using Decoder: fapn', show_time=False)
            from decoders.fapn import FaPNHead as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes)

        elif cfg.decoder == 'lawin':
            logger.log('INFO', 'Using Decoder: lawin', show_time=False)
            from decoders.lawin import LawinHead as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes)
        elif cfg.decoder == 'fpn':
            logger.log('INFO', 'Using Decoder: FPNHead', show_time=False)
            from decoders.fpn import FPNHead as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes)
        elif cfg.decoder == 'condnet':
            logger.log('INFO', 'Using Decoder: CondNet', show_time=False)
            from decoders.condnet import CondHead as Head
            self.decode_head = Head(in_channel=self.channels[-1], num_classes=cfg.num_classes)

        else:
            logger.log('INFO', 'No decoder(FCN-32s)', show_time=False)
            from decoders.fcnhead import MainFCNHead
            self.decode_head = MainFCNHead(in_channels=self.channels[-1], num_classes=cfg.num_classes,
                                           norm_layer=norm_layer)

        self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.logger.log('INFO', 'Loading pretrained model: {}'.format(pretrained), show_time=False)
            self.backbone.init_weights(pretrained=pretrained)

        if 'swin' in cfg.backbone:
            patch_first_conv_swin(self, cfg.in_channel1, cfg.in_channel2)
        elif 'CSwin' in cfg.backbone:
            patch_first_conv_cswin(self, cfg.in_channel1, cfg.in_channel2)
        elif 'mit' in cfg.backbone:
            patch_first_conv_mit(self, cfg.in_channel1, cfg.in_channel2)
        elif 'dilate' in cfg.backbone or 'pvt' in cfg.backbone:
            patch_first_conv_DilateFormer(self, cfg.in_channel1, cfg.in_channel2)
        elif 'biformer' in cfg.backbone or 'convnext' in cfg.backbone:
            # patch_first_conv_single_biformer(self, cfg.in_channel1, cfg.in_channel2)
            patch_first_conv_biformer(self, cfg.in_channel1, cfg.in_channel2)
        elif 'SMT' in cfg.backbone or 'uniformer' in cfg.backbone:
            patch_first_conv_SMT(self, cfg.in_channel1, cfg.in_channel2)
        elif 'emo' in cfg.backbone:
            patch_first_conv_EMO(self, cfg.in_channel1, cfg.in_channel2)
        elif 'resnet' in cfg.backbone:
            patch_first_conv_resnet(self, cfg.in_channel1, cfg.in_channel2)
        elif 'mobile' in cfg.backbone:
            patch_first_conv_mobilenet(self, cfg.in_channel1, cfg.in_channel2)

        self.logger.log('INFO', 'Initing weights ...', show_time=False)
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                        self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                        mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = x1.shape
        x = self.backbone(x1, x2)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head and self.training:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out


if __name__ == '__main__':
    log_env = '{}/params_statistic/{}_{}_{}'.format(config.log_path, config.decoder, config.backbone, config.env)
    common.check_path(log_env)
    log_filename = '{}/{}_{}_{}_train.log'.format(log_env, config.decoder, config.backbone, config.env)
    logger = Logger(log_filename, append=config.log_append)
    model = EncoderDecoder(cfg=config, logger=logger, norm_layer=nn.BatchNorm2d).cuda()

    left = torch.randn(1, 4, 256, 256).cuda()
    right = torch.randn(1, 1, 256, 256).cuda()

    flops = FlopCountAnalysis(model, (left, right))
    print(flop_count_table(flops))

    # summary(model, [(4, 256, 256), (1, 256, 256)])
    flops, params = profile(model, (left, right), verbose=False)

    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % flops)
    print('Total params: %s' % params)
