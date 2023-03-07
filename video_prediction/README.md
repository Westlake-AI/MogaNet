# Applying MogaNet to Video Prediction

This repo is a PyTorch implementation of applying **MogaNet** to unsupervised video prediction with [SimVP](https://arxiv.org/abs/2206.05099) on [Moving MNIST](http://arxiv.org/abs/1502.04681). The code is based on [SimVPv2](https://github.com/chengtan9907/SimVPv2). It is worth noticing that the Translator module in [SimVP](https://arxiv.org/abs/2206.05099) can be replaced by any [MetaFormer](https://arxiv.org/abs/2111.11418) block, which can benchmark the video prediction performance of MetaFormers.
For more details, see [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295) (arXiv 2022).

## Environement Setup

Install [SimVPv2](https://github.com/chengtan9907/SimVPv2) with pipe as follow. It can also be installed with `environment.yml`
```
python setup.py develop
```

## Data preparation

Prepare [Moving MNIST](http://arxiv.org/abs/1502.04681) with [script](tools/prepare_data/download_mmnist.sh) according to the [guidelines](docs/en/get_started.md).

<p align="right">(<a href="#top">back to top</a>)</p>

## Results and models on MMNIST

**Notes**: All the models are trained 200 and 2000 epochs by Adam optimizer and Onecycle learning rate scheduler. The trained models can also be downloaded by [**Baidu Cloud**](https://pan.baidu.com/s/1d5MTTC66gegehmfZvCQRUA?pwd=z8mf) (z8mf) at `MogaNet/MMNIST_VP`. The params (M) and FLOPs (G) are measured by [non_dist_train.py](tools/non_dist_train.py) by setting `--fps`.

| Architecture | Setting | Params | FLOPs | FPS | MSE | MAE | SSIM |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| IncepU (SimVPv1) | 200 epoch | 58.0M | 19.4G | 209s | 32.15 | 89.05 | 0.9268 |
| ViT | 200 epoch | 46.1M | 16.9.G | 290s | 35.15 | 95.87 | 0.9139 |
| Swin Transformer | 200 epoch | 46.1M | 16.4G | 294s | 29.70 | 84.05 | 0.9331 |
| Uniformer | 200 epoch | 44.8M | 16.5G | 296s | 30.38 | 85.87 | 0.9308 |
| MLP-Mixer | 200 epoch | 38.2M | 14.7G | 334s | 29.52 | 83.36 | 0.9338 |
| ConvMixer | 200 epoch | 3.9M | 5.5G | 658s | 32.09 | 88.93 | 0.9259 |
| Poolformer | 200 epoch | 37.1M | 14.1G | 341s | 31.79 | 88.48 | 0.9271 |
| ConvNeXt | 200 epoch | 37.3M | 14.1G | 344s | 26.94 | 77.23 | 0.9397 |
| VAN | 200 epoch | 44.5M | 16.0G | 288s | 26.10 | 76.11 | 0.9417 |
| HorNet | 200 epoch | 45.7M | 16.3G | 287s | 29.64 | 83.26 | 0.9331 |
| MogaNet | 200 epoch | 46.8M | 16.5G | 255s | 25.57 | 75.19 | 0.9429 |
| IncepU (SimVPv1) | 2000 epoch | 58.0M | 19.4G | 209s | 21.15 | 64.15 | 0.9536 |
| ViT | 2000 epoch | 46.1M | 16.9.G | 290s | 19.74 | 61.65 | 0.9539 |
| Swin Transformer | 2000 epoch | 46.1M | 16.4G | 294s | 19.11 | 59.84 | 0.9584 |
| Uniformer | 2000 epoch | 44.8M | 16.5G | 296s | 18.01 | 57.52 | 0.9609 |
| MLP-Mixer | 2000 epoch | 38.2M | 14.7G | 334s | 18.85 | 59.86 | 0.9589 |
| ConvMixer | 2000 epoch | 3.9M | 5.5G | 658s | 22.30 | 67.37 | 0.9507 |
| Poolformer | 2000 epoch | 37.1M | 14.1G | 341s | 20.96 | 64.31 | 0.9539 |
| ConvNeXt | 2000 epoch | 37.3M | 14.1G | 344s | 17.58 | 55.76 | 0.9617 |
| VAN | 2000 epoch | 44.5M | 16.0G | 288s | 16.21 | 53.57 | 0.9646 |
| HorNet | 2000 epoch | 45.7M | 16.3G | 287s | 17.40 | 55.70 | 0.9624 |
| MogaNet | 2000 epoch | 46.8M | 16.5G | 255s | 15.67 | 51.84 | 0.9661 |

## Training

We train the model on a single GPU by default (a batch size of 16 for SimVP). Start training with the bash script as:
```bash
python tools/non_dist_train.py -d mmnist -m SimVP --model_type moga -c configs/mmnist/simvp/SimVP_MogaNet.py --lr 1e-3 --ex_name mmnist_simvp_moga
```

## Evaluation

We test the trained model on a single GPU with the bash script as:
```bash
python tools/non_dist_test.py -d mmnist -m SimVP --model_type moga -c configs/mmnist/simvp/SimVP_MogaNet.py --ex_name /path/to/exp_name
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

- [SimVPv2](https://github.com/chengtan9907/SimVPv2)

<p align="right">(<a href="#top">back to top</a>)</p>
