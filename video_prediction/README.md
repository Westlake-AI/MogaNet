# Applying MogaNet to Video Prediction

This repo is a PyTorch implementation of applying **MogaNet** to unsupervised video prediction with [SimVP](https://arxiv.org/abs/2206.05099) on [Moving MNIST](http://arxiv.org/abs/1502.04681). The code is based on [SimVPv2](https://github.com/chengtan9907/SimVPv2). It is worth noticing that the Translator module in [SimVP](https://arxiv.org/abs/2206.05099) can be replaced by any [MetaFormer](https://arxiv.org/abs/2111.11418) block, which can benchmark the video prediction performance of MetaFormers.
For more details, see [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295) (arXiv 2022).

## Environement Setup

Install [SimVPv2](https://github.com/chengtan9907/SimVPv2) with conda.
```
conda env create -f environment.yml
conda activate SimVP
python setup.py develop
```

## Data preparation

Prepare ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets) in MMSegmentation. Please use the 2016 version of ADE20K dataset, which can be downloaded from [ADEChallengeData2016](data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) or [**Baidu Cloud**](https://pan.baidu.com/s/1EIrXVTOxX-cdhYVfqd9Vng?pwd=7ycz) (7ycz).

<p align="right">(<a href="#top">back to top</a>)</p>

## Results and models on ADE20K

**Notes**: All the models are evaluated at a single scale (SS), you can modify `test_pipeline` in config files to evaluate the multi-scale performance (MS). The trained models can also be downloaded by [**Baidu Cloud**](https://pan.baidu.com/s/1d5MTTC66gegehmfZvCQRUA?pwd=z8mf) (z8mf) at `MogaNet/ADE20K_Segmentation`. The params (M) and FLOPs (G) are measured by [get_flops](get_flops.sh) with 2048 $\times$ 512 resolutions.
```bash
bash get_flops.sh /path/to/config --shape 2048 512
```

## Training

We train the model on a single GPU by default (a batch size of 16 for SimVP). Start training with the bash script as:
```bash
python tools/non_dist_train.py -d mmnist -m SimVP --model_type gsta --lr 1e-3 --ex_name mmnist_simvp_gsta
```

## Evaluation

TODO

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
