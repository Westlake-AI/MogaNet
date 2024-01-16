# Applying MogaNet to Pose Estimation

This repo is a PyTorch implementation of applying **MogaNet** to 2D human pose estimation on COCO. The code is based on [MMPose](https://github.com/open-mmlab/mmpose/tree/v0.29.0).
For more details, see [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295) (ICLR 2024).

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

**Notes**: All the models use ImageNet-1K pre-trained backbones and can also be downloaded by [**Baidu Cloud**](https://pan.baidu.com/s/1d5MTTC66gegehmfZvCQRUA?pwd=z8mf) (z8mf) at `MogaNet/COCO_Pose`. The params (M) and FLOPs (G) are measured by [get_flops](get_flops.py) with 256 $\times$ 192 or 384 $\times$ 288 resolutions.
```bash
python get_flops.py /path/to/config --shape 256 192
```

### MogaNet + Top-Down

We provide results of MogaNet and popular architectures (Swin, ConvNeXt, and Uniformer) in comparison.

| Backbone | Input Size | Params | FLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>M</sup> | AR<sup>L</sup> | Config | Download |
|---|:---:|:---:|:---:|:---:|---|---|---|---|---|:---:|:---:|
| MogaNet-XT | 256x192 | 5.6M | 1.8G | 72.1 | 89.7 | 80.1 | 77.7 | 73.6 | 83.6 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_xt_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_xt_coco_256x192.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_xt_coco_256x192.pth) |
| MogaNet-XT | 384x288 | 5.6M | 4.2G | 74.7 | 90.1 | 81.3 | 79.9 | 75.9 | 85.9 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_xt_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_xt_coco_384x288.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_xt_coco_384x288.pth) |
| MogaNet-T | 256x192 | 8.1M | 2.2G | 73.2 | 90.1 | 81.0 | 78.8 | 74.9 | 84.4 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_t_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_t_coco_256x192.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_t_coco_256x192.pth) |
| MogaNet-T | 384x288 | 8.1M | 4.9G | 75.7 | 90.6 | 82.6 | 80.9 | 76.8 | 86.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_t_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_t_coco_384x288.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_t_coco_384x288.pth) |
| MogaNet-S | 256x192 | 29.0M | 6.0G | 74.9 | 90.7 | 82.8 | 80.1 | 75.7 | 86.3 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_s_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_s_coco_256x192.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_s_coco_256x192.pth) |
| MogaNet-S | 384x288 | 29.0M | 13.5G | 76.4 | 91.0 | 83.3 | 81.4 | 77.1 | 87.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_s_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_s_coco_384x288.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_s_coco_384x288.pth) |
| MogaNet-B | 256x192 | 47.4M | 10.9G | 75.3 | 90.9 | 83.3 | 80.7 | 76.4 | 87.1 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_b_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_b_coco_256x192.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_b_coco_256x192.pth) |
| MogaNet-B | 384x288 | 47.4M | 24.4G | 77.3 | 91.4 | 84.0 | 82.2 | 77.9 | 88.5 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_b_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_b_coco_384x288.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_b_coco_384x288.pth) |

### MetaFormers + Top-Down

| Backbone | Input Size | Params | FLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>M</sup> | AR<sup>L</sup> | Config | Download |
|---|:---:|:---:|:---:|:---:|---|---|---|---|---|:---:|:---:|
| Swin-T | 256x192 | 32.8M | 6.1G | 72.4 | 90.1 | 80.6 | 78.2 | 74.0 | 84.3 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_t_p4_w7_coco_256x192.py) | [model](https://download.openmmlab.com/mmpose/top_down/swin/swin_t_p4_w7_coco_256x192-eaefe010_20220503.pth) \| [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_t_p4_w7_coco_256x192_20220503.log.json) |
| Swin-B | 256x192 | 93.0M | 18.6G | 73.7 | 90.4 | 82.0 | 79.8 | 74.9 | 85.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_b_p4_w7_coco_256x192.py) | [model](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_256x192-7432be9e_20220705.pth) \| [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_256x192_20220705.log.json) |
| Swin-B | 384x288 | 93.0M | 40.1G | 75.9 | 91.0 | 83.2 | 78.8 | 76.5 | 87.5 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_b_p4_w7_coco_384x288.py) | [model](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_384x288-3abf54f9_20220705.pth) \| [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_384x288_20220705.log.json) |
| Swin-L | 256x192 | 203.4M | 40.3G | 74.3 | 90.6 | 82.1 | 79.8 | 75.5 | 86.2 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_l_p4_w7_coco_256x192.py) | [model](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_256x192-642a89db_20220705.pth) \| [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_256x192_20220705.log.json) |
| Swin-L | 384x288 | 203.4M | 86.9G | 76.3 | 91.2 | 83.0 | 81.4 | 77.0 | 87.9 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_l_p4_w7_coco_384x288.py) | [model](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_384x288-c36b7845_20220705.pth) \| [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_384x288_20220705.log.json) |
| ConvNeXt-T | 256x192 | 33.0M | 5.5G | 73.2 | 90.0 | 80.9 | 78.8 | 74.5 | 85.1 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/convnext_t_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_t_coco_256x192.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_t_coco_256x192.pth) |
| ConvNeXt-T | 384x288 | 33.0M | 12.5G | 75.3 | 90.4 | 82.1 | 80.5 | 76.1 | 86.8 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/convnext_t_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_t_coco_384x288.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_t_coco_384x288.pth) |
| ConvNeXt-S | 256x192 | 54.7M | 9.7G | 73.7 | 90.3 | 81.9 | 79.3 | 75.0 | 85.5 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/convnext_s_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_s_coco_256x192.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_s_coco_256x192.pth) |
| ConvNeXt-S | 384x288 | 54.7M | 21.8G | 75.8 | 90.7 | 83.1 | 81.0 | 76.8 | 87.1 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/convnext_s_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_s_coco_384x288.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_s_coco_384x288.pth) |
| ConvNeXt-B | 256x192 | 93.9M | 16.3G | 74.0 | 90.7 | 82.1 | 79.5 | 75.2 | 85.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/convnext_b_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_b_coco_256x192.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_b_coco_256x192.pth) |
| ConvNeXt-B | 384x288 | 93.9M | 36.6G | 75.9 | 90.6 | 83.1 | 81.1 | 76.5 | 87.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/convnext_b_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_b_coco_384x288.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/convnext_b_coco_384x288.pth) |
| UniFormer-S | 256x192 | 25.2M | 4.7G | 74.0 | 90.3 | 82.2 | 79.5 | 66.8 | 76.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/uniformer_s_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/uniformer_s_coco_256x192.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/uniformer_s_coco_256x192.pth) |
| UniFormer-S | 384x288 | 25.2M | 11.1G | 75.9 | 90.6 | 83.4 | 81.4 | 68.6 | 79.0 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/uniformer_s_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/uniformer_s_coco_384x288.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/uniformer_s_coco_384x288.pth) |
| UniFormer-B | 256x192 | 53.5M | 9.2G | 75.0 | 90.6 | 83.0 | 80.4 | 67.8 | 77.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/uniformer_b_coco_256x192.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/uniformer_b_coco_256x192.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/uniformer_b_coco_256x192.pth) |
| UniFormer-B | 384x288 | 53.5M | 14.8G | 76.7 | 90.8 | 84.0 | 81.4 | 69.3 | 79.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/uniformer_b_coco_384x288.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/uniformer_b_coco_384x288.log.json) \| [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/uniformer_b_coco_384x288.pth) |

## Demo

We provide some demos according to [MMPose](https://github.com/open-mmlab/mmpose/demo). Please use [inference_demo](./demo/inference_demo.ipynb) or run the python tools with following script:
```bash
cd demo
python top_down_img_demo.py path/to/config path/to/checkpoint --img-root coco2017_val --json-file ../data/coco/annotations/person_keypoints_val2017.json --show
```

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
@inproceedings{iclr2024MogaNet,
  title={Efficient Multi-order Gated Aggregation Network},
  author={Siyuan Li and Zedong Wang and Zicheng Liu and Cheng Tan and Haitao Lin and Di Wu and Zhiyuan Chen and Jiangbin Zheng and Stan Z. Li},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

## Acknowledgment

Our segmentation implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

- [MMPose](https://github.com/open-mmlab/mmpose)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [PVT segmentation](https://github.com/whai362/PVT/tree/v2/segmentation)
- [PoolFormer](https://github.com/sail-sg/poolformer)

<p align="right">(<a href="#top">back to top</a>)</p>
