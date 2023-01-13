# Installation

We provide installation instructions for ImageNet classification experiments here.

## Dependency Setup
Create an new conda virtual environment
```
conda create -n moganet python=3.8 -y
conda activate moganet
```

Install [Pytorch](https://pytorch.org/)>=1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.9.0 following official instructions. For example:
```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Clone this repo and install required packages. You can install [apex-amp](https://github.com/NVIDIA/apex) if you want to use fp16 with Pytorch<=1.6.0.
```
git clone https://github.com/Westlake-AI/MogaNet
pip install timm fvcore
```

The results in this repository are produced with `torch==1.10.0+cu111 torchvision==0.11.0+cu111 timm==0.6.12`, and we adopt amp fp16 for fast training.

## Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset ([train](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar) and [val](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)) and structure the data as follows. You can extract ImageNet with this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
