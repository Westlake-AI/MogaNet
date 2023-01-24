# Applying MogaNet to Pose Estimation

This repo is a PyTorch implementation of applying **MogaNet** to 2D human pose estimation on COCO. The code is based on [MMPose](https://github.com/open-mmlab/mmpose/tree/v0.29.0).
For more details, see [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295) (arXiv 2022).

## Note

Please note that we just simply follow the hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/detection) and [Swin](https://github.com/microsoft/Swin-Transformer) which may not be the optimal ones for MogaNet. Feel free to tune the hyper-parameters to get better performance.

## Environement Setup

Install [MMPose](https://github.com/open-mmlab/mmpose/) from souce code, or follow the following steps. This experiment uses MMPose>=0.29.0, and we reproduced the results with [MMPose v0.29.0](https://github.com/open-mmlab/mmpose/tree/v0.29.0) and Pytorch==1.10.
```
pip install openmim
mim install mmcv-full
pip install mmpose
```

Note: Since we write [MogaNet backbone code](../models/moganet.py) of detection, segmentation, and pose estimation in the same file, it also works for [MMDetection](https://github.com/open-mmlab/mmdetection/tree/v2.26.0) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.29.1) through `@BACKBONES.register_module()`. Please continue to install MMDetection or MMSegmentation for further usage.

## Data preparation

Prepare COCO according to the guidelines in [MMPose](https://github.com/open-mmlab/mmpose/).

<p align="right">(<a href="#top">back to top</a>)</p>

## Results and models on COCO

**Notes**: All the models can also be downloaded by [**Baidu Cloud**](https://pan.baidu.com/s/1d5MTTC66gegehmfZvCQRUA?pwd=z8mf) (z8mf). The params (M) and FLOPs (G) are measured by [get_flops](get_flops.sh) with 256 $\times$ 192 or 384 $\times$ 288 resolutions.
```bash
bash get_flops.sh /path/to/config --shape 256 192
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
