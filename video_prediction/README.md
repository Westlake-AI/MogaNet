# Applying MogaNet to Video Prediction

This repo is a PyTorch implementation of applying **MogaNet** to unsupervised video prediction with [SimVP](https://arxiv.org/abs/2206.05099) on [Moving MNIST](http://arxiv.org/abs/1502.04681). The code is based on [SimVPv2](https://github.com/chengtan9907/SimVPv2) (or its latest version [OpenSTL](https://github.com/chengtan9907/OpenSTL)). It is worth noticing that the Translator module in [SimVP](https://arxiv.org/abs/2206.05099) can be replaced by any [MetaFormer](https://arxiv.org/abs/2111.11418) block, which can benchmark the video prediction performance of MetaFormers.
For more details, see [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295) (ICLR 2024).

## Environement Setup

Install [SimVPv2](https://github.com/chengtan9907/SimVPv2) with pipe as follow. It can also be installed with `environment.yml`.
```
python setup.py develop
```

## Data preparation

Prepare [Moving MNIST](http://arxiv.org/abs/1502.04681) with [script](tools/prepare_data/download_mmnist.sh) according to the [guidelines](docs/en/get_started.md).

<p align="right">(<a href="#top">back to top</a>)</p>

## Results and models on MMNIST

**Notes**: All the models are trained 200 and 2000 epochs by Adam optimizer and Onecycle learning rate scheduler. The trained models can also be downloaded by [**Baidu Cloud**](https://pan.baidu.com/s/1d5MTTC66gegehmfZvCQRUA?pwd=z8mf) (z8mf) at `MogaNet/MMNIST_VP`. The params (M) and FLOPs (G) are measured by [non_dist_train.py](tools/non_dist_train.py) by setting `--fps`. Please refer to [video_benchmarks](https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/docs/en/model_zoos/video_benchmarks.md) for the full results.

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

## Results on OpenSTL Benchmarks

- [Video Benchmarks](https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/docs/en/model_zoos/video_benchmarks.md) and [visualizations](https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/docs/en/visualization/video_visualization.md).
- [Weather Benchmarks](https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/docs/en/model_zoos/weather_benchmarks.md) and [visualizations](https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/docs/en/visualization/weather_visualization.md).
- [Traffic Benchmarks](https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/docs/en/model_zoos/traffic_benchmarks.md) and [visualizations](https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/docs/en/visualization/traffic_visualization.md).

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
@inproceedings{iclr2024MogaNet,
  title={Efficient Multi-order Gated Aggregation Network},
  author={Siyuan Li and Zedong Wang and Zicheng Liu and Cheng Tan and Haitao Lin and Di Wu and Zhiyuan Chen and Jiangbin Zheng and Stan Z. Li},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

## Acknowledgment
Our segmentation implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

- [SimVPv2](https://github.com/chengtan9907/SimVPv2)
- [OpenSTL](https://github.com/chengtan9907/OpenSTL)

<p align="right">(<a href="#top">back to top</a>)</p>
