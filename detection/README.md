# Applying MogaNet to Object Detection

This repo is a PyTorch implementation of applying **MogaNet** to object detaction and instance segmentation with [Mask R-CNN](https://arxiv.org/abs/1703.06870) and [RetinaNet](https://arxiv.org/abs/1708.02002) on [COCO](https://arxiv.org/abs/1405.0312). The code is based on [MMDetection](https://github.com/open-mmlab/mmdetection/tree/v2.26.0).
For more details, see [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295) (arXiv 2022).

## Note

Please note that we simply follow the hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/detection) and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), which may not be the optimal ones for MogaNet. Feel free to tune the hyper-parameters to get better performance.

## Environement Setup

Install [MMDetection](https://github.com/open-mmlab/mmdetection/) from souce code, or follow the following steps. This experiment uses MMDetection>=2.19.0, and we reproduced the results with [MMDetection v2.26.0](https://github.com/open-mmlab/mmdetection/tree/v2.26.0) and Pytorch==1.10.
```
pip install openmim
mim install mmcv-full
pip install mmdet
```

Apex (optional) for Pytorch<=1.6.0:
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext --user
```

By default, we run experiments with fp32 or fp16 (Apex). If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the configuration files:
```
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

Note: Since we write [MogaNet backbone code](../models/moganet.py) of detection, segmentation, and pose estimation in the same file, it also works for [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.29.1) and [MMPose](https://github.com/open-mmlab/mmpose/tree/v0.29.0) through `@BACKBONES.register_module()`. Please continue to install MMSegmentation or MMPose for further usage.

## Data preparation

Download [COCO2017](https://cocodataset.org/#download) and prepare COCO experiments according to the guidelines in [MMDetection](https://github.com/open-mmlab/mmdetection/).

<p align="right">(<a href="#top">back to top</a>)</p>

## Results and models on COCO

**Notes**: All the models can also be downloaded by [**Baidu Cloud**](https://pan.baidu.com/s/1d5MTTC66gegehmfZvCQRUA?pwd=z8mf) (z8mf) at `MogaNet/COCO_Detection`. The params (M) and FLOPs (G) are measured by [get_flops](get_flops.py) with 1280 $\times$ 800 resolutions.
```bash
python get_flops.py /path/to/config --shape 1280 800
```

### MogaNet + RetinaNet

| Method | Backbone | Pretrain | Params | FLOPs | Lr schd | box mAP | Config | Download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RetinaNet | MogaNet-XT | ImageNet-1K | 12.1M | 167.2G | 1x | 39.7 | [config](configs/retinanet_moganet_xtiny_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_xtiny_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_xtiny_fpn_1x_coco.pth) |
| RetinaNet | MogaNet-T | ImageNet-1K | 14.4M | 173.4G | 1x | 41.4 | [config](configs/retinanet_moganet_tiny_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_tiny_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_tiny_fpn_1x_coco.pth) |
| RetinaNet | MogaNet-S | ImageNet-1K | 35.1M | 253.0G | 1x | 45.8 | [config](configs/retinanet_moganet_small_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_small_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_small_fpn_1x_coco.pth) |
| RetinaNet | MogaNet-B | ImageNet-1K | 53.5M | 354.5G | 1x | 47.7 | [config](configs/retinanet_moganet_base_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_base_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_base_fpn_1x_coco.pth) |
| RetinaNet | MogaNet-L | ImageNet-1K | 92.4M | 476.8G | 1x | 48.7 | [config](configs/retinanet_moganet_large_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_large_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_large_fpn_1x_coco.pth) |

### MogaNet + Mask R-CNN

| Method | Backbone | Pretrain | Params | FLOPs | Lr schd | box mAP | mask mAP | Config | Download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mask R-CNN | MogaNet-XT | ImageNet-1K | 22.8M | 185.4G | 1x | 40.7 | 37.6 | [config](configs/mask_rcnn_moganet_xtiny_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_xtiny_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_xtiny_fpn_1x_coco.pth) |
| Mask R-CNN | MogaNet-T | ImageNet-1K | 25.0M | 191.7G | 1x | 42.6 | 39.1 | [config](configs/mask_rcnn_moganet_tiny_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_tiny_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_tiny_fpn_1x_coco.pth) |
| Mask R-CNN | MogaNet-S | ImageNet-1K | 45.0M | 271.6G | 1x | 46.6 | 42.2 | [config](configs/mask_rcnn_moganet_small_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_small_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_small_fpn_1x_coco.pth) |
| Mask R-CNN | MogaNet-B | ImageNet-1K | 63.4M | 373.1G | 1x | 49.0 | 43.8 | [config](configs/mask_rcnn_moganet_base_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_base_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_base_fpn_1x_coco.pth) |
| Mask R-CNN | MogaNet-L | ImageNet-1K | 102.1M | 495.3G | 1x | 49.4 | 44.2 | [config](configs/mask_rcnn_moganet_large_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_large_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_large_fpn_1x_coco.pth) |

## Demo

We provide some demos according to [MMDetection](https://github.com/open-mmlab/mmdetection/demo). Please use [inference_demo](./demo/inference_demo.ipynb) or run the following script:
```bash
cd demo
python image_demo.py demo.png ../configs/moganet/mask_rcnn_moganet_small_fpn_1x_coco.py ../../work_dirs/checkpoints/mask_rcnn_moganet_small_fpn_1x_coco.pth --out-file pred.png
```

## Training

We train the model on a single node with 8 GPUs (a batch size of 16) by default. Start training with the config as:
```bash
PORT=29001 bash dist_train.sh /path/to/config 8
```

## Evaluation

To evaluate the trained model on a single node with 8 GPUs, run:
```bash
bash dist_test.sh /path/to/config /path/to/checkpoint 8 --out results.pkl --eval bbox # or `bbox segm`
```

## Citation

If you find this repository helpful, please consider citing:
```
@article{Li2022MogaNet,
  title={Efficient Multi-order Gated Aggregation Network},
  author={Siyuan Li and Zedong Wang and Zicheng Liu and Cheng Tan and Haitao Lin and Di Wu and Zhiyuan Chen and Jiangbin Zheng and Stan Z. Li},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.03295}
}
```

## Acknowledgment

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [PVT detection](https://github.com/whai362/PVT/tree/v2/detection)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [PoolFormer](https://github.com/sail-sg/poolformer)

<p align="right">(<a href="#top">back to top</a>)</p>
