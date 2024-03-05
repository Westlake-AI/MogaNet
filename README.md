<div align="center">
<!-- <h1>MogaNet: Efficient Multi-order Gated Aggregation Network</h1> -->
<h2><a href="https://arxiv.org/abs/2211.03295">MogaNet: Multi-order Gated Aggregation Network (ICLR 2024)</a></h2>

[Siyuan Li](https://lupin1998.github.io/)<sup>\*,1,2</sup>, [Zedong Wang](https://jacky1128.github.io)<sup>\*,1</sup>, [Zicheng Liu](https://pone7.github.io/)<sup>1,2</sup>, [Chen Tan](https://chengtan9907.github.io/)<sup>1,2</sup>, [Haitao Lin](https://bird-tao.github.io/)<sup>1,2</sup>, [Di Wu](https://scholar.google.com/citations?user=egz8bGQAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Zhiyuan Chen](https://zyc.ai/)<sup>1</sup>, [Jiangbin Zheng](https://scholar.google.com/citations?user=egz8bGQAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ&hl=zh-CN)<sup>†,1</sup>

<sup>1</sup>[Westlake University](https://westlake.edu.cn/), <sup>2</sup>[Zhejiang University](https://www.zju.edu.cn/english/)
</div>

<p align="center">
<a href="https://arxiv.org/abs/2211.03295" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2211.03295-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/Westlake-AI/MogaNet/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23B7A800" /></a>
<a href="https://colab.research.google.com/github/Westlake-AI/MogaNet/blob/main/demo.ipynb" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
<a href="https://huggingface.co/MogaNet" alt="Huggingface">
    <img src="https://img.shields.io/badge/huggingface-MogaNet-blueviolet" /></a>
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/44519745/202308950-00708e25-9ac7-48f0-af12-224d927ac1ae.jpg" width=100% height=100% 
class="center">
</p>

We propose **MogaNet**, a new family of efficient ConvNets designed through the lens of multi-order game-theoretic interaction, to pursue informative context mining with preferable complexity-performance trade-offs. It shows excellent scalability and attains competitive results among state-of-the-art models with more efficient use of model parameters on ImageNet and multifarious typical vision benchmarks, including COCO object detection, ADE20K semantic segmentation, 2D\&3D human pose estimation, and video prediction.

This repository contains PyTorch implementation for MogaNet (ICLR 2024).

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#catalog">Catalog</a></li>
    <li><a href="#image-classification">Image Classification</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>

## Catalog

We plan to release implementations of MogaNet in a few months. Please watch us for the latest release. Currently, this repo is reimplemented according to our official implementations in [OpenMixup](https://github.com/Westlake-AI/openmixup), and we are working on cleaning up experimental results and code implementations. Models are released in [GitHub](https://github.com/Westlake-AI/MogaNet/releases) / [Baidu Cloud](https://pan.baidu.com/s/1d5MTTC66gegehmfZvCQRUA?pwd=z8mf) / [Hugging Face](https://huggingface.co/MogaNet).

- [x] **ImageNet-1K** Training and Validation Code with [timm](https://github.com/rwightman/pytorch-image-models) [[code](#image-classification)] [[models](https://github.com/Westlake-AI/MogaNet/releases/tag/moganet-in1k-weights)] [[Hugging Face 🤗](https://huggingface.co/MogaNet)]
- [x] **ImageNet-1K** Training and Validation Code in [OpenMixup](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet) / [MMPretrain (TODO)](https://github.com/open-mmlab/mmpretrain)
- [x] Downstream Transfer to **Object Detection and Instance Segmentation on COCO** [[code](detection/)] [[models](https://github.com/Westlake-AI/MogaNet/releases/tag/moganet-det-weights)] [[demo](detection/demo/)]
- [x] Downstream Transfer to **Semantic Segmentation on ADE20K** [[code](segmentation/)] [[models](https://github.com/Westlake-AI/MogaNet/releases/tag/moganet-seg-weights)] [[demo](segmentation/demo/)]
- [x] Downstream Transfer to **2D Human Pose Estimation on COCO** [[code](pose_estimation/)] (baselines supported) [[models](https://github.com/Westlake-AI/MogaNet/releases/tag/moganet-pose-weights)] [[demo](pose_estimation/demo/)]
 - [ ] Downstream Transfer to **3D Human Pose Estimation** (baseline models will be supported) <!--[[code](human_pose_3d/)] (baseline models will be supported) -->
- [x] Downstream Transfer to **Video Prediction on MMNIST Variants** [[code](video_prediction/)] (baselines supported)
- [x] Image Classification on Google Colab and Notebook Demo [[demo](demo.ipynb)]

<p align="center">
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/239330216-a93e71ee-7909-485d-8257-1b34abcd61c6.jpg" width=100% height=100% 
class="center">
</p>


## Image Classification

### 1. Installation

Please check [INSTALL.md](INSTALL.md) for installation instructions.

### 2. Training and Validation

See [TRAINING.md](TRAINING.md) for ImageNet-1K training and validation instructions, or refer to our [OpenMixup](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/) implementations. We released pre-trained models on [OpenMixup](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/) in [moganet-in1k-weights](https://github.com/Westlake-AI/openmixup/releases/tag/moganet-in1k-weights). We have also reproduced ImageNet results with this repo and released `args.yaml` / `summary.csv` / `model.pth.tar` in [moganet-in1k-weights](https://github.com/Westlake-AI/MogaNet/releases/tag/moganet-in1k-weights). The parameters in the trained model can be extracted by [code](extract_ckpt.py).

Here is a notebook [demo](demo.ipynb) of MogaNet which run the steps to perform inference with MogaNet for image classification.

### 3. ImageNet-1K Trained Models

| Model | Resolution | Params (M) | Flops (G) | Top-1 / top-5 (%) | Script | Download |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| MogaNet-XT | 224x224 | 2.97 | 0.80 | 76.5 \| 93.4 | [args](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_xtiny_sz224_8xbs128_ep300_args.yaml) \| [script](TRAINING.md) | [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_xtiny_sz224_8xbs128_ep300.pth.tar) \| [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_xtiny_sz224_8xbs128_ep300_summary.csv) |
| MogaNet-XT | 256x256 | 2.97 | 1.04 | 77.2 \| 93.8 | [args](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_xtiny_sz256_8xbs128_ep300_args.yaml) \| [script](TRAINING.md) | [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_xtiny_sz256_8xbs128_ep300.pth.tar) \| [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_xtiny_sz256_8xbs128_ep300_summary.csv) |
| MogaNet-T | 224x224 | 5.20 | 1.10 | 79.0 \| 94.6 | [args](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_tiny_sz224_8xbs128_ep300_args.yaml) \| [script](TRAINING.md) | [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_tiny_sz224_8xbs128_ep300.pth.tar) \| [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_tiny_sz224_8xbs128_ep300_summary.csv) |
| MogaNet-T | 256x256 | 5.20 | 1.44 | 79.6 \| 94.9 | [args](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_tiny_sz256_8xbs128_ep300_args.yaml) \| [script](TRAINING.md) | [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_tiny_sz256_8xbs128_ep300.pth.tar) \| [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_tiny_sz256_8xbs128_ep300_summary.csv) |
| MogaNet-T\* | 256x256 | 5.20 | 1.44 | 80.0 \| 95.0 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_tiny_deit3_sz256_lr2e_3_8xb128_fp16_ep300.py) \| [script](TRAINING.md) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_tiny_deit3_sz256_lr2e_3_8xb128_fp16_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_tiny_deit3_sz256_lr2e_3_8xb128_fp16_ep300.log.json) |
| MogaNet-S | 224x224 | 25.3 | 4.97 | 83.4 \| 96.9 | [args](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_small_sz224_8xbs128_ep300_args.yaml) \| [script](TRAINING.md) | [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_small_sz224_8xbs128_ep300.pth.tar) \| [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_small_sz224_8xbs128_ep300_summary.csv) |
| MogaNet-B | 224x224 | 43.9 | 9.93 | 84.3 \| 97.0 | [args](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_base_sz224_8xbs128_ep300_args.yaml) \| [script](TRAINING.md) | [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_base_sz224_8xbs128_ep300.pth.tar) \| [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_base_sz224_8xbs128_ep300_summary.csv) |
| MogaNet-L | 224x224 | 82.5 | 15.9 | 84.7 \| 97.1 | [args](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_large_sz224_8xbs64_ep300_args.yaml) \| [script](TRAINING.md) | [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_large_sz224_8xbs64_ep300.pth.tar) \| [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_large_sz224_8xbs64_ep300_summary.csv) |
| MogaNet-XL | 224x224 | 180.8 | 34.5 | 85.1 \| 97.4 | [args](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_xlarge_sz224_8xbs64_ep300_args.yaml) \| [script](TRAINING.md) | [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_xlarge_sz224_8xbs64_ep300.pth.tar) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_xlarge_ema_sz224_8xb32_accu2_ep300.log.json) |

### 4. Analysis Tools

(1) The [code](get_flops.py) to count MACs of MogaNet variants.

```
python get_flops.py --model moganet_tiny
```
<p align="center">
<img src="https://user-images.githubusercontent.com/44519745/212429257-f0b09d7a-7503-4945-9517-68ea36d10e00.png" width=100% height=100% 
class="center">
</p>

(2) The [code](cam_image.py) to visualize Grad-CAM activation maps (or variants of Grad-CAM) of MogaNet and other popular architectures.

```
python cam_image.py --use_cuda --image_path /path/to/image.JPEG --model moganet_tiny --method gradcam
```

<p align="right">(<a href="#top">back to top</a>)</p>

### 5. Downstream Tasks

<details>
  <summary>Object Detection and Instance Segmentation on COCO</summary>
  <li><a href="https://github.com/Westlake-AI/MogaNet/tree/main/detection">MogaNet + Mask R-CNN</a></li>

  | Method | Backbone | Pretrain | Params | FLOPs | Lr schd | box mAP | mask mAP | Config | Download |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | Mask R-CNN | MogaNet-XT | ImageNet-1K | 22.8M | 185.4G | 1x | 40.7 | 37.6 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/mask_rcnn_moganet_xtiny_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_xtiny_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_xtiny_fpn_1x_coco.pth) |
  | Mask R-CNN | MogaNet-T | ImageNet-1K | 25.0M | 191.7G | 1x | 42.6 | 39.1 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/mask_rcnn_moganet_tiny_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_tiny_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_tiny_fpn_1x_coco.pth) |
  | Mask R-CNN | MogaNet-S | ImageNet-1K | 45.0M | 271.6G | 1x | 46.6 | 42.2 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/mask_rcnn_moganet_small_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_small_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_small_fpn_1x_coco.pth) |
  | Mask R-CNN | MogaNet-B | ImageNet-1K | 63.4M | 373.1G | 1x | 49.0 | 43.8 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/mask_rcnn_moganet_base_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_base_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_base_fpn_1x_coco.pth) |
  | Mask R-CNN | MogaNet-L | ImageNet-1K | 102.1M | 495.3G | 1x | 49.4 | 44.2 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/mask_rcnn_moganet_t_fpn_mstrain_480-800_3x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_large_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_large_fpn_1x_coco.pth) |
  | Mask R-CNN | MogaNet-T | ImageNet-1K | 25.0M | 191.7G | MS 3x | 45.3 | 40.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/mask_rcnn_moganet_t_fpn_mstrain_480-800_3x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_t_fpn_mstrain_480-800_3x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_t_fpn_mstrain_480-800_3x_coco.pth) |
  | Mask R-CNN | MogaNet-S | ImageNet-1K | 45.0M | 271.6G | MS 3x | 48.5 | 43.1 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/mask_rcnn_moganet_s_fpn_mstrain_480-800_3x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_s_fpn_mstrain_480-800_3x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_s_fpn_mstrain_480-800_3x_coco.pth) |
  | Mask R-CNN | MogaNet-B | ImageNet-1K | 63.4M | 373.1G | MS 3x | 50.3 | 44.4 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/mask_rcnn_moganet_b_fpn_mstrain_480-800_3x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_b_fpn_mstrain_480-800_3x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_b_fpn_mstrain_480-800_3x_coco.pth) |
  | Mask R-CNN | MogaNet-L | ImageNet-1K | 63.4M | 373.1G | MS 3x | 50.6 | 44.6 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/mask_rcnn_moganet_l_fpn_mstrain_480-800_3x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_l_fpn_mstrain_480-800_3x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/mask_rcnn_moganet_l_fpn_mstrain_480-800_3x_coco.pth) |

  <li><a href="https://github.com/Westlake-AI/MogaNet/tree/main/detection">MogaNet + RetinaNet</a></li>

  | Method | Backbone | Pretrain | Params | FLOPs | Lr schd | box mAP | Config | Download |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | RetinaNet | MogaNet-XT | ImageNet-1K | 12.1M | 167.2G | 1x | 39.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/retinanet_moganet_xtiny_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_xtiny_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_xtiny_fpn_1x_coco.pth) |
  | RetinaNet | MogaNet-T | ImageNet-1K | 14.4M | 173.4G | 1x | 41.4 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/retinanet_moganet_tiny_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_tiny_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_tiny_fpn_1x_coco.pth) |
  | RetinaNet | MogaNet-S | ImageNet-1K | 35.1M | 253.0G | 1x | 45.8 | [config](configs/moganet/retinanet_moganet_small_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_small_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_small_fpn_1x_coco.pth) |
  | RetinaNet | MogaNet-B | ImageNet-1K | 53.5M | 354.5G | 1x | 47.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/retinanet_moganet_base_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_base_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_base_fpn_1x_coco.pth) |
  | RetinaNet | MogaNet-L | ImageNet-1K | 92.4M | 476.8G | 1x | 48.7 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/retinanet_moganet_large_fpn_1x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_large_fpn_1x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/retinanet_moganet_large_fpn_1x_coco.pth) |

  <li><a href="https://github.com/Westlake-AI/MogaNet/tree/main/detection">MogaNet + Cascade Mask R-CNN</a></li>

  | Method | Backbone | Pretrain | Params | FLOPs | Lr schd | box mAP | mask mAP | Config | Download |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | Cascade Mask R-CNN | MogaNet-S | ImageNet-1K | 77.9M | 405.4G | MS 3x | 51.4 | 44.9 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/cascade_mask_rcnn_moganet_s_fpn_ms_3x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/cascade_mask_rcnn_moganet_s_fpn_ms_3x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/cascade_mask_rcnn_moganet_s_fpn_ms_3x_coco.pth) |
  | Cascade Mask R-CNN | MogaNet-S | ImageNet-1K | 82.8M | 750.2G | GIOU+MS 3x | 51.7 | 45.1 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/cascade_mask_rcnn_moganet_s_fpn_giou_4conv1f_ms_3x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/cascade_mask_rcnn_moganet_s_fpn_giou_4conv1f_ms_3x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/cascade_mask_rcnn_moganet_s_fpn_giou_4conv1f_ms_3x_coco.pth) |
  | Cascade Mask R-CNN | MogaNet-B | ImageNet-1K | 101.2M | 851.6G | GIOU+MS 3x | 52.6 | 46.0 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/cascade_mask_rcnn_moganet_b_fpn_giou_4conv1f_ms_3x_coco.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/cascade_mask_rcnn_moganet_b_fpn_giou_4conv1f_ms_3x_coco.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-det-weights/cascade_mask_rcnn_moganet_b_fpn_giou_4conv1f_ms_3x_coco.pth) |
  | Cascade Mask R-CNN | MogaNet-L | ImageNet-1K | 139.9M | 973.8G | GIOU+MS 3x | 53.3 | 46.1 | [config](https://github.com/Westlake-AI/MogaNet/tree/main/detection/configs/moganet/cascade_mask_rcnn_moganet_l_fpn_giou_4conv1f_ms_3x_coco.py) | - |
</details>
<details>
  <summary>Semantic Segmentation on ADE20K</summary>
  <li><a href="https://github.com/Westlake-AI/MogaNet/tree/main/segmentation">MogaNet + Semantic FPN</a></li>

  | Method | Backbone | Pretrain | Params | FLOPs | Iters | mIoU | mAcc | Config | Download |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | Semantic FPN | MogaNet-XT | ImageNet-1K | 6.9M | 101.4G | 80K | 40.3 | 52.4 | [config](configs/sem_fpn/moganet/fpn_moganet_xtiny_80k_ade20k.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/fpn_moganet_xtiny_80k_ade20k.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/fpn_moganet_xtiny_80k_ade20k.pth) |
  | Semantic FPN | MogaNet-T | ImageNet-1K | 9.1M | 107.8G | 80K | 43.1 | 55.4 | [config](configs/sem_fpn/moganet/fpn_moganet_tiny_80k_ade20k.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/fpn_moganet_tiny_80k_ade20k.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/fpn_moganet_tiny_80k_ade20k.pth) |
  | Semantic FPN | MogaNet-S | ImageNet-1K | 29.1M | 189.7G | 80K | 47.7 | 59.8 | [config](configs/sem_fpn/moganet/fpn_moganet_small_80k_ade20k.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/fpn_moganet_small_80k_ade20k.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/fpn_moganet_small_80k_ade20k.pth) |
  | Semantic FPN | MogaNet-B | ImageNet-1K | 47.5M | 293.6G | 80K | 49.3 | 61.6 | [config](configs/sem_fpn/moganet/fpn_moganet_base_80k_ade20k.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/fpn_moganet_base_80k_ade20k.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/fpn_moganet_base_80k_ade20k.pth) |
  | Semantic FPN | MogaNet-L | ImageNet-1K | 86.2M | 418.7G | 80K | 50.2 | 63.0 | [config](configs/sem_fpn/moganet/fpn_moganet_large_80k_ade20k.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/fpn_moganet_large_80k_ade20k.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/fpn_moganet_large_80k_ade20k.pth) |

  <li><a href="https://github.com/Westlake-AI/MogaNet/tree/main/segmentation">MogaNet + UperNet</a></li>

  | Method | Backbone | Pretrain | Params | FLOPs | Iters | mIoU | mAcc | Config | Download |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | UperNet | MogaNet-XT | ImageNet-1K | 30.4M | 855.7G | 160K | 42.2 | 55.1 | [config](configs/upernet/moganet/upernet_moganet_xtiny_512x512_160k_ade20k.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/upernet_moganet_xtiny_512x512_160k_ade20k.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/upernet_moganet_xtiny_512x512_160k_ade20k.pth) |
  | UperNet | MogaNet-T | ImageNet-1K | 33.1M | 862.4G | 160K | 43.7 | 57.1 | [config](configs/upernet/moganet/upernet_moganet_tiny_512x512_160k_ade20k.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/upernet_moganet_tiny_512x512_160k_ade20k.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/upernet_moganet_tiny_512x512_160k_ade20k.pth) |
  | UperNet | MogaNet-S | ImageNet-1K | 55.3M | 946.4G | 160K | 49.2 | 61.6 | [config](configs/upernet/moganet/upernet_moganet_small_512x512_160k_ade20k.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/upernet_moganet_small_512x512_160k_ade20k.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/upernet_moganet_small_512x512_160k_ade20k.pth) |
  | UperNet | MogaNet-B | ImageNet-1K | 73.7M | 1050.4G | 160K | 50.1 | 63.4 | [config](configs/upernet/moganet/upernet_moganet_base_512x512_160k_ade20k.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/upernet_moganet_base_512x512_160k_ade20k.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/upernet_moganet_base_512x512_160k_ade20k.pth) |
  | UperNet | MogaNet-L | ImageNet-1K | 113.2M | 1176.1G | 160K | 50.9 | 63.5 | [config](configs/upernet/moganet/upernet_moganet_large_512x512_160k_ade20k.py) | [log](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/upernet_moganet_large_512x512_160k_ade20k.log.json) / [model](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-seg-weights/upernet_moganet_large_512x512_160k_ade20k.pth) |
</details>
<details>
  <summary>2D Human Pose Estimation on COCO</summary>
  <li><a href="https://github.com/Westlake-AI/MogaNet/tree/main/pose_estimation">MogaNet + Top-Down</a></li>

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
</details>
<details>
  <summary>Video Prediction on Moving MNIST</summary>

  | Architecture     |   Setting  | Params |  FLOPs | FPS |  MSE  |  MAE  |  SSIM  |  PSNR |   Download   |
  |------------------|:----------:|:------:|:------:|:---:|:-----:|:-----:|:------:|:-----:|:------------:|
  | IncepU (SimVPv1) |  200 epoch |  58.0M |  19.4G | 209 | 32.15 | 89.05 | 0.9268 | 21.84 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_incepu_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_incepu_one_ep200.log) |
  | gSTA (SimVPv2)   |  200 epoch |  46.8M |  16.5G | 282 | 26.69 | 77.19 | 0.9402 | 22.78 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_gsta_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_gsta_one_ep200.log) |
  | ViT              |  200 epoch |  46.1M |  16.9G | 290 | 35.15 | 95.87 | 0.9139 | 21.67 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_vit_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_vit_one_ep200.log) |
  | Swin Transformer |  200 epoch |  46.1M |  16.4G | 294 | 29.70 | 84.05 | 0.9331 | 22.22 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_swin_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_swin_one_ep200.log) |
  | Uniformer        |  200 epoch |  44.8M |  16.5G | 296 | 30.38 | 85.87 | 0.9308 | 22.13 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_uniformer_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_uniformer_one_ep200.log) |
  | MLP-Mixer        |  200 epoch |  38.2M |  14.7G | 334 | 29.52 | 83.36 | 0.9338 | 22.22 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_mlpmixer_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_mlpmixer_one_ep200.log) |
  | ConvMixer        |  200 epoch |   3.9M |   5.5G | 658 | 32.09 | 88.93 | 0.9259 | 21.93 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_convmixer_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_convmixer_one_ep200.log) |
  | Poolformer       |  200 epoch |  37.1M |  14.1G | 341 | 31.79 | 88.48 | 0.9271 | 22.03 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_poolformer_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_poolformer_one_ep200.log) |
  | ConvNeXt         |  200 epoch |  37.3M |  14.1G | 344 | 26.94 | 77.23 | 0.9397 | 22.74 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_convnext_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_convnext_one_ep200.log) |
  | VAN              |  200 epoch |  44.5M |  16.0G | 288 | 26.10 | 76.11 | 0.9417 | 22.89 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_van_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_van_one_ep200.log) |
  | HorNet           |  200 epoch |  45.7M |  16.3G | 287 | 29.64 | 83.26 | 0.9331 | 22.26 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_hornet_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_hornet_one_ep200.log) |
  | **MogaNet**      |  200 epoch |  46.8M |  16.5G | 255 | 25.57 | 75.19 | 0.9429 | 22.99 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_moganet_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_moganet_one_ep200.log) |
  | IncepU (SimVPv1) | 2000 epoch |  58.0M |  19.4G | 209 | 21.15 | 64.15 | 0.9536 | 23.99 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_incepu_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_incepu_one_ep2000.log) |
  | gSTA (SimVPv2)   | 2000 epoch |  46.8M |  16.5G | 282 | 15.05 | 49.80 | 0.9675 | 25.97 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_gsta_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_gsta_one_ep2000.log) |
  | ViT              | 2000 epoch |  46.1M | 16.9.G | 290 | 19.74 | 61.65 | 0.9539 | 24.59 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_vit_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_vit_one_ep2000.log) |
  | Swin Transformer | 2000 epoch |  46.1M |  16.4G | 294 | 19.11 | 59.84 | 0.9584 | 24.53 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_swin_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_swin_one_ep2000.log) |
  | Uniformer        | 2000 epoch |  44.8M |  16.5G | 296 | 18.01 | 57.52 | 0.9609 | 24.92 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_uniformer_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_uniformer_one_ep2000.log) |
  | MLP-Mixer        | 2000 epoch |  38.2M |  14.7G | 334 | 18.85 | 59.86 | 0.9589 | 24.58 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_mlpmixer_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_mlpmixer_one_ep2000.log) |
  | ConvMixer        | 2000 epoch |   3.9M |   5.5G | 658 | 22.30 | 67.37 | 0.9507 | 23.73 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_convmixer_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_convmixer_one_ep2000.log) |
  | Poolformer       | 2000 epoch |  37.1M |  14.1G | 341 | 20.96 | 64.31 | 0.9539 | 24.15 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_poolformer_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_poolformer_one_ep2000.log) |
  | ConvNeXt         | 2000 epoch |  37.3M |  14.1G | 344 | 17.58 | 55.76 | 0.9617 | 25.06 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_convnext_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_convnext_one_ep2000.log) |
  | VAN              | 2000 epoch |  44.5M |  16.0G | 288 | 16.21 | 53.57 | 0.9646 | 25.49 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_van_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_van_one_ep2000.log) |
  | HorNet           | 2000 epoch |  45.7M |  16.3G | 287 | 17.40 | 55.70 | 0.9624 | 25.14 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_hornet_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_hornet_one_ep2000.log) |
  | **MogaNet**      | 2000 epoch |  46.8M |  16.5G | 255 | 15.67 | 51.84 | 0.9661 | 25.70 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_moganet_one_ep2000.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mmnist-weights/mmnist_simvp_s_moganet_one_ep2000.log) |

  <summary>Video Prediction on Moving FMNIST</summary>

  | Architecture     |   Setting  | Params |  FLOPs | FPS |  MSE  |   MAE  |  SSIM  |  PSNR |   Download   |
  |------------------|:----------:|:------:|:------:|:---:|:-----:|:------:|:------:|:-----:|:------------:|
  | IncepU (SimVPv1) |  200 epoch |  58.0M |  19.4G | 209 | 30.77 | 113.94 | 0.8740 | 21.81 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_incepu_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_incepu_one_ep200.log) |
  | gSTA (SimVPv2)   |  200 epoch |  46.8M |  16.5G | 282 | 25.86 | 101.22 | 0.8933 | 22.61 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_gsta_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_gsta_one_ep200.log) |
  | ViT              |  200 epoch |  46.1M | 16.9.G | 290 | 31.05 | 115.59 | 0.8712 | 21.83 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_vit_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_vit_one_ep200.log) |
  | Swin Transformer |  200 epoch |  46.1M |  16.4G | 294 | 28.66 | 108.93 | 0.8815 | 22.08 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_swin_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_swin_one_ep200.log) |
  | Uniformer        |  200 epoch |  44.8M |  16.5G | 296 | 29.56 | 111.72 | 0.8779 | 21.97 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_uniformer_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_uniformer_one_ep200.log) |
  | MLP-Mixer        |  200 epoch |  38.2M |  14.7G | 334 | 28.83 | 109.51 | 0.8803 | 22.01 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_mlpmixer_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_mlpmixer_one_ep200.log) |
  | ConvMixer        |  200 epoch |   3.9M |   5.5G | 658 | 31.21 | 115.74 | 0.8709 | 21.71 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_convmixer_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_convmixer_one_ep200.log) |
  | Poolformer       |  200 epoch |  37.1M |  14.1G | 341 | 30.02 | 113.07 | 0.8750 | 21.95 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_poolformer_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_poolformer_one_ep200.log) |
  | ConvNeXt         |  200 epoch |  37.3M |  14.1G | 344 | 26.41 | 102.56 | 0.8908 | 22.49 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_convnext_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_convnext_one_ep200.log) |
  | VAN              |  200 epoch |  44.5M |  16.0G | 288 | 31.39 | 116.28 | 0.8703 | 22.82 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_van_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_van_one_ep200.log) |
  | HorNet           |  200 epoch |  45.7M |  16.3G | 287 | 29.19 | 110.17 | 0.8796 | 22.03 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_hornet_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_hornet_one_ep200.log) |
  | **MogaNet**      |  200 epoch |  46.8M |  16.5G | 255 | 25.14 |  99.69 | 0.8960 | 22.73 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_moganet_one_ep200.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/mfmnist-weights/fmnist_simvp_moganet_one_ep200.log) |
</details>

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

- [pytorch-image-models (timm)](https://github.com/rwightman/pytorch-image-models): PyTorch image models, scripts, pretrained weights.
- [PoolFormer](https://github.com/sail-sg/poolformer): Official PyTorch implementation of MetaFormer.
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt): Official PyTorch implementation of ConvNeXt.
- [OpenMixup](https://github.com/Westlake-AI/openmixup): Open-source toolbox for visual representation learning.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab Detection Toolbox and Benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab Semantic Segmentation Toolbox and Benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab Pose Estimation Toolbox and Benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D Human Parametric Model Toolbox and Benchmark.
- [OpenSTL](https://github.com/chengtan9907/OpenSTL): A Comprehensive Benchmark of Spatio-Temporal Predictive Learning.

## Citation

If you find this repository helpful, please consider citing:
```
@inproceedings{iclr2024MogaNet,
  title={MogaNet: Multi-order Gated Aggregation Network},
  author={Siyuan Li and Zedong Wang and Zicheng Liu and Cheng Tan and Haitao Lin and Di Wu and Zhiyuan Chen and Jiangbin Zheng and Stan Z. Li},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
