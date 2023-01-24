# Applying MogaNet to Semantic Segmentation

This repo is a PyTorch implementation of applying **MogaNet** to semantic segmentation on ADE20K. The code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.29.1).
For more details, see [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295) (arXiv 2022).

## Note

Please note that we just simply follow the hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/detection), [Swin](https://github.com/microsoft/Swin-Transformer), and [VAN](https://github.com/Visual-Attention-Network/VAN-Segmentation), which may not be the optimal ones for MogaNet. Feel free to tune the hyper-parameters to get better performance.

## Environement Setup

Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/) from souce code, or follow the following steps. This experiment uses MMSegmentation>=0.19.0, and we reproduced the results with [MMSegmentation v0.29.1](https://github.com/open-mmlab/mmsegmentation/tree/v0.29.1) and Pytorch==1.10.
```
pip install openmim
mim install mmcv-full
pip install mmseg
```

Note: Since we write [MogaNet backbone code](../models/moganet.py) of detection, segmentation, and pose estimation in the same file, it also works for [MMDetection](https://github.com/open-mmlab/mmdetection/tree/v2.26.0) and [MMPose](https://github.com/open-mmlab/mmpose/tree/v0.29.0) through `@BACKBONES.register_module()`. Please continue to install MMDetection or MMPose for further usage.

## Data preparation

Prepare ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets) in MMSegmentation. Please use the 2016 version of ADE20K dataset, which can be downloaded from [ADEChallengeData2016](data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) or [**Baidu Cloud**](https://pan.baidu.com/s/1EIrXVTOxX-cdhYVfqd9Vng?pwd=7ycz) (7ycz).

<p align="right">(<a href="#top">back to top</a>)</p>

## Results and models on ADE20K

**Notes**: All the models can also be downloaded by [**Baidu Cloud**](https://pan.baidu.com/s/1d5MTTC66gegehmfZvCQRUA?pwd=z8mf) (z8mf). The params (M) and FLOPs (G) are measured by [get_flops](get_flops.sh) with 2048 $\times$ 512 resolutions.
```bash
bash get_flops.sh /path/to/config --shape 2048 512
```

### MogaNet + Semantic FPN

| Method | Backbone | Pretrain | Params | FLOPs | Iters | mIoU | mAcc | Config | Download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Semantic FPN | MogaNet-XT | ImageNet-1K | 6.9M | 101.4G | 40K |  |  | [config](configs/sem_fpn/moganet/fpn_moganet_xtiny_40k_ade20k.py) | log / model |
| Semantic FPN | MogaNet-T | ImageNet-1K | 9.1M | 107.8G | 40K |  |  | [config](configs/sem_fpn/moganet/fpn_moganet_xtiny_80k_ade20k.py) | log / model |
| Semantic FPN | MogaNet-S | ImageNet-1K | 29.1M | 189.7G | 40K |  |  | [config](configs/sem_fpn/moganet/fpn_moganet_tiny_40k_ade20k.py) | log / model |
| Semantic FPN | MogaNet-B | ImageNet-1K | 47.5M | 293.6G | 40K |  |  | [config](configs/sem_fpn/moganet/fpn_moganet_tiny_80k_ade20k.py) | log / model |
| Semantic FPN | MogaNet-L | ImageNet-1K | 86.2M | 418.7G | 40K |  |  | [config](configs/sem_fpn/moganet/fpn_moganet_small_40k_ade20k.py) | log / model |
| Semantic FPN | MogaNet-XT | ImageNet-1K | 6.9M | 101.4G | 80K |  |  | [config](configs/sem_fpn/moganet/fpn_moganet_small_80k_ade20k.py) | log / model |
| Semantic FPN | MogaNet-T | ImageNet-1K | 9.1M | 107.8G | 80K |  |  | [config](configs/sem_fpn/moganet/fpn_moganet_base_40k_ade20k.py) | log / model |
| Semantic FPN | MogaNet-S | ImageNet-1K | 29.1M | 189.7G | 80K |  |  | [config](configs/sem_fpn/moganet/fpn_moganet_base_80k_ade20k.py) | log / model |
| Semantic FPN | MogaNet-B | ImageNet-1K | 47.5M | 293.6G | 80K |  |  | [config](configs/sem_fpn/moganet/fpn_moganet_large_40k_ade20k.py) | log / model |
| Semantic FPN | MogaNet-L | ImageNet-1K | 86.2M | 418.7G | 80K |  |  | [config](configs/sem_fpn/moganet/fpn_moganet_large_80k_ade20k.py) | log / model |

### MogaNet + UperNet

| Method | Backbone | Pretrain | Params | FLOPs | Iters | mIoU | mAcc | Config | Download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| UperNet | MogaNet-XT | ImageNet-1K | 30.4M | 855.7G | 160K |  |  | [config](configs/upernet/moganet/upernet_moganet_xtiny_512x512_160k_ade20k.py) | log / model |
| UperNet | MogaNet-T | ImageNet-1K | 33.1M | 862.4G | 160K |  |  | [config](configs/upernet/moganet/upernet_moganet_tiny_512x512_160k_ade20k.py) | log / model |
| UperNet | MogaNet-S | ImageNet-1K | 55.3M | 946.4G | 160K |  |  | [config](configs/upernet/moganet/upernet_moganet_small_512x512_160k_ade20k.py) | log / model |
| UperNet | MogaNet-B | ImageNet-1K | 73.7M | 1050.4G | 160K |  |  | [config](configs/upernet/moganet/upernet_moganet_base_512x512_160k_ade20k.py) | log / model |
| UperNet | MogaNet-L | ImageNet-1K | 113.2M | 1176.1G | 160K |  |  | [config](configs/upernet/moganet/upernet_moganet_large_512x512_160k_ade20k.py) | log / model |

## Training

We train the model on a single node with 8 GPUs by default (a batch size of 32 / 16 for Semantic FPN / UperNet). Start training with the config as:
```bash
PORT=29001 bash dist_train.sh /path/to/config 8
```

## Evaluation

To evaluate the trained model on a single node with 8 GPUs, run:
```bash
bash dist_test.sh /path/to/config /path/to/checkpoint 8 --out results.pkl --eval mIoU
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

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [PVT segmentation](https://github.com/whai362/PVT/tree/v2/segmentation)
- [PoolFormer](https://github.com/sail-sg/poolformer)
- [VAN-Segmentation](https://github.com/Visual-Attention-Network/VAN-Segmentation)

<p align="right">(<a href="#top">back to top</a>)</p>
