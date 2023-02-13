# Applying MogaNet to Pose Estimation

This repo is a PyTorch implementation of applying **MogaNet** to 2D human pose estimation on COCO. The code is based on [MMPose](https://github.com/open-mmlab/mmpose/tree/v0.29.0).
For more details, see [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295) (arXiv 2022).

## Note

Please note that we simply follow the hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/detection) and [Swin](https://github.com/microsoft/Swin-Transformer) which may not be the optimal ones for MogaNet. Feel free to tune the hyper-parameters to get better performance.

## Environement Setup

Install [MMPose](https://github.com/open-mmlab/mmpose/) from souce code, or follow the following steps. This experiment uses MMPose>=0.29.0, and we reproduced the results with [MMPose v0.29.0](https://github.com/open-mmlab/mmpose/tree/v0.29.0) and Pytorch==1.10.
```
pip install openmim
mim install mmcv-full
pip install mmpose
```

Note: Since we write [MogaNet backbone code](../models/moganet.py) of detection, segmentation, and pose estimation in the same file, it also works for [MMDetection](https://github.com/open-mmlab/mmdetection/tree/v2.26.0) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.29.1) through `@BACKBONES.register_module()`. Please continue to install MMDetection or MMSegmentation for further usage.

## Data preparation

Download [COCO2017](https://cocodataset.org/#download) and prepare COCO experiments according to the guidelines in [MMPose](https://github.com/open-mmlab/mmpose/).

<p align="right">(<a href="#top">back to top</a>)</p>

## Results and models on COCO

**Notes**: All the models use ImageNet-1K pre-trained backbones and can also be downloaded by [**Baidu Cloud**](https://pan.baidu.com/s/1d5MTTC66gegehmfZvCQRUA?pwd=z8mf) (z8mf) at `MogaNet/COCO_Pose`. The params (M) and FLOPs (G) are measured by [get_flops](get_flops.sh) with 256 $\times$ 192 or 384 $\times$ 288 resolutions.
```bash
bash get_flops.sh /path/to/config --shape 256 192
```

### MogaNet + Top-Down

| Backbone | Input Size | Params | FLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>M</sup> | AR<sup>L</sup> | Config | Download |
|---|:---:|:---:|:---:|:---:|---|---|---|---|---|:---:|:---:|
| MogaNet-XT | 256x192 | 5.6M | 1.8G | 72.1 | 89.7 | 80.1 | 77.7 | 73.6 | 83.6 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_xt_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_xt_coco_256x192.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_xt_coco_256x192.pth) |
| MogaNet-XT | 384x288 | 5.6M | 4.2G | 74.7 | 90.1 | 81.3 | 79.9 | 75.9 | 85.9 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_xt_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_xt_coco_384x288.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_xt_coco_384x288.pth) |
| MogaNet-T | 256x192 | 8.1M | 2.2G | 73.2 | 90.1 | 81.0 | 78.8 | 74.9 | 84.4 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_t_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_t_coco_256x192.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_t_coco_256x192.pth) |
| MogaNet-T | 384x288 | 8.1M | 4.9G | 75.7 | 90.6 | 82.6 | 80.9 | 76.8 | 86.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_t_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_t_coco_384x288.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_t_coco_384x288.pth) |
| MogaNet-S | 256x192 | 29.0M | 6.0G | 74.8 | 90.7 | 82.8 | 80.1 | 75.7 | 86.3 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_s_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_s_coco_256x192.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_s_coco_256x192.pth) |
| MogaNet-S | 384x288 | 29.0M | 13.5G | 76.4 | 91.0 | 83.3 | 81.4 | 77.1 | 87.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_s_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_s_coco_384x288.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_s_coco_384x288.pth) |
| MogaNet-B | 256x192 | 47.4M | 10.9G | 75.3 | 90.9 | 83.3 | 80.7 | 76.4 | 87.1 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_b_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_b_coco_256x192.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_b_coco_256x192.pth) |
| MogaNet-B | 384x288 | 47.4M | 24.4G | 77.3 | 91.4 | 84.0 | 82.2 | 77.9 | 88.5 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_b_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_b_coco_384x288.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_b_coco_384x288.pth) |

## Training

We train the model on a single node with 8 GPUs by default (a batch size of 32 $\times$ 8 for Top-Down). Start training with the config as:
```bash
PORT=29001 bash dist_train.sh /path/to/config 8
```

## Evaluation

To evaluate the trained model on a single node with 8 GPUs, run:
```bash
bash dist_test.sh /path/to/config /path/to/checkpoint 8 --out results.pkl --eval mAP
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

Our segmentation implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

- [MMPose](https://github.com/open-mmlab/mmpose)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [PVT segmentation](https://github.com/whai362/PVT/tree/v2/segmentation)
- [PoolFormer](https://github.com/sail-sg/poolformer)

<p align="right">(<a href="#top">back to top</a>)</p>
