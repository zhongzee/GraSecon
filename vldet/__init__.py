# Copyright (c) Facebook, Inc. and its affiliates.
from .modeling.meta_arch import custom_rcnn
from .modeling.roi_heads import vldet_roi_heads
from .modeling.roi_heads import res5_roi_heads
from .modeling.backbone import swintransformer
from .modeling.backbone import timm

# When we import each dataset package, they will be registered automatically
# because the register all method is written directly at the end of each script
from .data import datasets

# from .data.datasets import lvis_v1
# from .data.datasets import imagenet
# from .data.datasets import cc
# from .data.datasets import objects365
# from .data.datasets import oid
# from .data.datasets import coco_zeroshot


