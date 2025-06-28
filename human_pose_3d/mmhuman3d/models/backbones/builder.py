# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

# from .convnext import ConvNeXt
from .hornet import HorNet
from .mobilenet_v2 import MobileNetV2
from .moganet import MogaNet
from .hrnet import (
    PoseHighResolutionNet,
    PoseHighResolutionNetExpose,
    PoseHighResolutionNetPyMAFX,
)
from .resnet import PoseResNet, ResNet, ResNetV1d
# from .vision_transformer import VisionTransformer

from mmcls.models import ConvNeXt, SwinTransformer, VisionTransformer

BACKBONES = Registry('backbones')

BACKBONES.register_module(name='ConvNeXt', module=ConvNeXt)
BACKBONES.register_module(name='HorNet', module=HorNet)
BACKBONES.register_module(name='MobileNetV2', module=MobileNetV2)
BACKBONES.register_module(name='MogaNet', module=MogaNet)
BACKBONES.register_module(name='ResNet', module=ResNet)
BACKBONES.register_module(name='ResNetV1d', module=ResNetV1d)
BACKBONES.register_module(name='PoseResNet', module=PoseResNet)
BACKBONES.register_module(
    name='PoseHighResolutionNet', module=PoseHighResolutionNet)
BACKBONES.register_module(
    name='PoseHighResolutionNetExpose', module=PoseHighResolutionNetExpose)
BACKBONES.register_module(
    name='PoseHighResolutionNetPyMAFX', module=PoseHighResolutionNetPyMAFX)
BACKBONES.register_module(name='SwinTransformer', module=SwinTransformer)
BACKBONES.register_module(name='VisionTransformer', module=VisionTransformer)


def build_backbone(cfg):
    """Build backbone."""
    if cfg is None:
        return None
    return BACKBONES.build(cfg)
