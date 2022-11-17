# [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295)

<p align="left">
<a href="https://arxiv.org/abs/2210.13452" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2210.13452-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/Westlake-AI/MogaNet/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23B7A800" /></a>
<!-- <a href="https://colab.research.google.com/github/sail-sg/metaformer/blob/main/misc/demo_metaformer_baselines.ipynb" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a> -->
</p>

This is a PyTorch implementation of **MogaNet** from our paper:

[Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295)\
[Siyuan Li](https://lupin1998.github.io/), [Zedong Wang](https://zedongwang.netlify.app/), [Zicheng Liu](https://pone7.github.io/), [Chen Tan](https://chengtan9907.github.io/), [Haitao Lin](https://bird-tao.github.io/), [Di Wu](https://scholar.google.com/citations?user=egz8bGQAAAAJ&hl=zh-CN), [Zhiyuan Chen](https://zyc.ai/), [Jiangbin Zheng](https://scholar.google.com/citations?user=egz8bGQAAAAJ&hl=zh-CN), and [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ&hl=zh-CN). In arXiv, 2022.


<p align="center">
<img src="https://user-images.githubusercontent.com/44519745/202308950-00708e25-9ac7-48f0-af12-224d927ac1ae.jpg" width=100% height=100% 
class="center">
</p>

We propose **MogaNet**, a new family of efficient ConvNets, to pursue informative context mining with preferable complexity-performance trade-offs.

## Catalog

We plan to release implementations of MogaNet in a few months. Please watch us for the latest release. Currently, this repo is reimplemented according to our official implementations in [OpenMixup](https://github.com/Westlake-AI/openmixup/), and we are working on cleaning up experimental results and code implementations.

- [x] ImageNet-1K Training Code
- [ ] ImageNet-1K Fine-tuning Code
- [ ] Downstream Transfer to Object Detection and Instance Segmentation on COCO
- [ ] Downstream Transfer to Semantic Segmentation on ADE20K
- [ ] Image Classification on Google Colab and Web Demo

<!-- ✅ ⬜️  -->

## Results and Pre-trained Models

### ImageNet-1K trained models

| Model | resolution | Params(M) | Flops(G) | Top-1 (%) | Config | Download |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| MogaNet-XT | 224x224 | 2.97 | 0.80 | 76.5 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_xtiny_sz224_8xb128_fp16_ep300.py) / [script](TRAINING.md) | model / log |
| MogaNet-T | 224x224 | 5.20 | 1.10 | 79.0 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_tiny_sz224_8xb128_fp16_ep300.py) / [script](TRAINING.md) | model / log |
| MogaNet-T | 256x256 | 5.20 | 1.44 | 79.6 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_tiny_sz256_8xb128_fp16_ep300.py) / [script](TRAINING.md) | model / log |
| MogaNet-T\* | 256x256 | 5.20 | 1.44 | 80.0 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_tiny_deit3_sz256_lr2e_3_8xb128_fp16_ep300.py) / [script](TRAINING.md) | model / log |
| MogaNet-S | 224x224 | 25.3 | 4.97 | 83.4 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_small_sz224_8xb128_ep300.py) / [script](TRAINING.md) | model / log |
| MogaNet-B | 224x224 | 43.9 | 9.93 | 84.2 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_base_sz224_8xb128_ep300.py) / [script](TRAINING.md) | model / log |
| MogaNet-L | 224x224 | 82.5 | 15.9 | 84.6 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_large_sz224_8xb64_accu2_ep300.py) / [script](TRAINING.md) | model / log |


## Installation

Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Training

See [TRAINING.md](TRAINING.md) for ImageNet-1K training instructions, or refer to our [OpenMixup](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/) implementations.

## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit) and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) repositories.

## License

This project is released under the [Apache 2.0 license](LICENSE).

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
