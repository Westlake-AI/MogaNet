<br/>

<div align="center">
    <img src="resources/mmhuman3d-logo.png" width="600"/>
</div>

<br/>

<div align="center">

[![Documentation](https://readthedocs.org/projects/mmhuman3d/badge/?version=latest)](https://mmhuman3d.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmhuman3d/workflows/build/badge.svg)](https://github.com/open-mmlab/mmhuman3d/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmhuman3d/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmhuman3d)
[![PyPI](https://img.shields.io/pypi/v/mmhuman3d)](https://pypi.org/project/mmhuman3d/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmhuman3d.svg)](https://github.com/open-mmlab/mmhuman3d/blob/main/LICENSE)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmhuman3d.svg)](https://github.com/open-mmlab/mmhuman3d/issues)

</div>

## 简介

[English](README.md) | 简体中文

MMHuman3D 是一款基于 PyTorch 的人体参数化模型的开源工具箱，是 [OpenMMLab](https://openmmlab.com/) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.7** 以上的版本。

如果您对多视角动作捕捉感兴趣，请参考[XRMoCap](https://github.com/openxrlab/xrmocap)了解更多详情。

https://user-images.githubusercontent.com/62529255/144362861-e794b404-c48f-4ebe-b4de-b91c3fbbaa3b.mp4

### 主要特性

- **使用模块化的框架复现流行的算法**

  MMHuman3D 重新实现了流行的算法，帮助用户只需一行代码即可完成复现。模块化的框架适合快速验证：用户可以在不修改代码的情况下调试不同的超参数甚至神经网络结构。

- **通过一个统一的数据规范 HumanData 支持多种数据集**

  通过一个规范工具箱，我们将所有的支持的数据集都对齐到统一的数据格式 *HumanData* . 我们同时也提供预处理完成的数据文件。

- **多功能可视化工具箱**

  一整套可微的可视化工具支持人体参数化模型的渲染（包括部分分割，深度图以及点云）和传统 2D/3D 关键点的可视化。

## 最新进展
- 2022-10-12: MMHuman3D [v0.10.0](https://github.com/open-mmlab/mmhuman3d/releases/tag/v0.10.0) 已经发布. 主要更新包括:
  - 支持调用本地摄像头实时渲染
  - 更新数据载入脚本，进而实现训练加速
  - 加入[balanced MSE](https://github.com/jiawei-ren/BalancedMSE) 损失函数，可进一步提升HMR训练精度
- 2022-07-08: MMHuman3D [v0.9.0](https://github.com/open-mmlab/mmhuman3d/releases/tag/v0.9.0) 已经发布. 主要更新包括:
  - 支持 SMPL-X 估计方法 [ExPose](https://expose.is.tue.mpg.de/)，同时重建脸、手、身体
  - 支持新的人体参数化模型 [STAR](https://star.is.tue.mpg.de/)
  - 开源 [GTA-Human](https://caizhongang.github.io/projects/GTA-Human/) 数据集，以及 SPIN-FT (51.98 mm) 和 PARE-FT (46.84 mm) 基线! (官方开源)
  - 重构配准管线并提升SPIN至 57.54 mm
- 2022-05-31: MMHuman3D [v0.8.0](https://github.com/open-mmlab/mmhuman3d/releases/tag/v0.8.0) 已经发布. 主要更新包括:
  - 支持 SmoothNet（由论文作者添加）
  - 修复循环引用问题，获得最多2.5倍速度提升
  - 增加中文版文档

## 基准与模型库

更多详情可见 [模型库](docs/model_zoo.md)。

已支持的人体参数化模型:

<details open>
<summary>(click to collapse)</summary>

- [x] [SMPL](https://smpl.is.tue.mpg.de/) (SIGGRAPH Asia'2015)
- [x] [SMPL-X](https://smpl-x.is.tue.mpg.de/) (CVPR'2019)
- [x] [MANO](https://mano.is.tue.mpg.de/) (SIGGRAPH ASIA'2017)
- [x] [FLAME](https://flame.is.tue.mpg.de/) (SIGGRAPH ASIA'2017)
- [x] [STAR](https://star.is.tue.mpg.de/) (ECCV'2020)

</details>

已支持的算法：

<details open>
<summary>(click to collapse)</summary>

- [x] [SMPLify](https://smplify.is.tue.mpg.de/) (ECCV'2016)
- [x] [SMPLify-X](https://smpl-x.is.tue.mpg.de/) (CVPR'2019)
- [x] [HMR](https://akanazawa.github.io/hmr/) (CVPR'2018)
- [x] [SPIN](https://www.seas.upenn.edu/~nkolot/projects/spin/) (ICCV'2019)
- [x] [VIBE](https://github.com/mkocabas/VIBE) (CVPR'2020)
- [x] [HybrIK](https://jeffli.site/HybrIK/) (CVPR'2021)
- [x] [PARE](https://pare.is.tue.mpg.de/) (ICCV'2021)
- [x] [DeciWatch](https://ailingzeng.site/deciwatch) (ECCV'2022)
- [x] [SmoothNet](https://ailingzeng.site/smoothnet) (ECCV'2022)
- [x] [ExPose](https://expose.is.tue.mpg.de) (ECCV'2020)
- [x] [BalancedMSE](https://sites.google.com/view/balanced-mse/home) (CVPR'2022)

</details>

已支持的数据集：

<details open>
<summary>(click to collapse)</summary>

- [x] [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) (ECCV'2018)
- [x] [AGORA](https://agora.is.tue.mpg.de/) (CVPR'2021)
- [x] [AMASS](https://amass.is.tue.mpg.de/) (ICCV'2019)
- [x] [COCO](https://cocodataset.org/#home) (ECCV'2014)
- [x] [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody) (ECCV'2020)
- [x] [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) (CVPR'2019)
- [x] [EFT](https://github.com/facebookresearch/eft) (3DV'2021)
- [x] [GTA-Human](https://caizhongang.github.io/projects/GTA-Human/) (arXiv'2021)
- [x] [Human3.6M](http://vision.imar.ro/human3.6m/description.php) (TPAMI'2014)
- [x] [InstaVariety](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md) (CVPR'2019)
- [x] [LSP](https://sam.johnson.io/research/lsp.html) (BMVC'2010)
- [x] [LSP-Extended](https://sam.johnson.io/research/lspet.html) (CVPR'2011)
- [x] [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) (3DC'2017)
- [x] [MPII](http://human-pose.mpi-inf.mpg.de/) (CVPR'2014)
- [x] [Penn Action](http://dreamdragon.github.io/PennAction/) (ICCV'2012)
- [x] [PoseTrack18](https://posetrack.net/users/download.php) (CVPR'2018)
- [x] [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/) (CVPR'2017)
- [x] [UP3D](https://files.is.tuebingen.mpg.de/classner/up/) (CVPR'2017)
- [x] [FreiHand](https://lmb.informatik.uni-freiburg.de/projects/freihand/) (ICCV'2019)
- [x] [EHF](https://smpl-x.is.tue.mpg.de/) (CVPR'2019)
- [x] [Stirling/ESRC-Face3D](http://pics.psych.stir.ac.uk/ESRC/index.htm) (FG'2018)

</details>

我们将跟进学界的最新进展，并支持更多算法和框架。

如果您对MMHuman3D有任何功能需求，请随时在[愿望清单](https://github.com/open-mmlab/mmhuman3d/discussions/47)中留言。


## 快速入门

请参考[快速入门](docs/getting_started.md)文档学习 MMHuman3D 的基本使用。如下配置data相关文件，一些cache文件可从[Baidu Cloud](https://pan.baidu.com/s/1d5MTTC66gegehmfZvCQRUA?pwd=z8mf)下载。
```
./human_pose_3d
│   ├── data
│   │   ├── body_models
│   │   |   ├── flame
│   │   |   ├── mano
│   │   |   ├── smpl
│   │   |   ├── smplx
│   │   └── ...
│   │   ├── cache
│   │   ├── datasets
│   │   |   ├── crowpose
│   │   |   ├── ffhq
│   │   |   ├── stirling
│   │   ├── preprocessed_datasets
│   │   ├── pretrained_models
```

### Vision Network Baselines
View [config files](configs/expose) under `configs/expose` and reproduce the popular vision networks as baselines for the Face and Hand tasks, e.g.,
```bash
PORT=29501 bash tools/dist_train.sh configs/expose/swin_face/swin_t_hmr_expose_hand_4xb64_ep100.py work_dirs/expose/swin_face/swin_t_hmr_expose_hand_4xb64_ep100 4
PORT=29501 bash tools/dist_train.sh configs/expose/swin_face/swin_s_hmr_expose_hand_4xb64_ep100.py work_dirs/expose/swin_face/swin_s_hmr_expose_hand_4xb64_ep100 4
PORT=29501 bash tools/dist_train.sh configs/expose/swin_hand/swin_t_hmr_expose_hand_4xb64_ep100.py work_dirs/expose/swin_hand/swin_t_hmr_expose_hand_4xb64_ep100 4
PORT=29501 bash tools/dist_train.sh configs/expose/swin_hand/swin_s_hmr_expose_hand_4xb64_ep100.py work_dirs/expose/swin_hand/swin_s_hmr_expose_hand_4xb64_ep100 4
```

### MogaNet
Reproduce MogaNet variants with the Face and Hand tasks:
```bash
PORT=29501 bash tools/dist_train.sh configs/expose/moganet_face/moganet_t_hmr_expose_face_4xb64_ep100.py work_dirs/expose/moganet_face/moganet_t_hmr_expose_face_4xb64_ep100 4
PORT=29501 bash tools/dist_train.sh configs/expose/moganet_face/moganet_s_hmr_expose_face_4xb64_ep100.py work_dirs/expose/moganet_face/moganet_s_hmr_expose_face_4xb64_ep100 4
PORT=29501 bash tools/dist_train.sh configs/expose/moganet_face/moganet_b_hmr_expose_face_4xb64_ep100.py work_dirs/expose/moganet_face/moganet_b_hmr_expose_face_4xb64_ep100 4
PORT=29501 bash tools/dist_train.sh configs/expose/moganet_hand/moganet_t_hmr_expose_hand_4xb64_ep100.py work_dirs/expose/moganet_hand/moganet_t_hmr_expose_hand_4xb64_ep100 4
PORT=29501 bash tools/dist_train.sh configs/expose/moganet_hand/moganet_s_hmr_expose_hand_4xb64_ep100.py work_dirs/expose/moganet_hand/moganet_s_hmr_expose_hand_4xb64_ep100 4
PORT=29501 bash tools/dist_train.sh configs/expose/moganet_hand/moganet_b_hmr_expose_hand_4xb64_ep100.py work_dirs/expose/moganet_hand/moganet_b_hmr_expose_hand_4xb64_ep100 4
```

## 许可

该项目采用 [Apache 2.0 license](LICENSE) 开源协议。部分支持的算法可能采用了[额外协议](docs/additional_licenses.md)。

## 引用

如果您觉得 MMHuman3D 对您的研究有所帮助，请考虑引用它：

```bibtex
@misc{mmhuman3d,
    title={OpenMMLab 3D Human Parametric Model Toolbox and Benchmark},
    author={MMHuman3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmhuman3d}},
    year={2021}
}
```

## 参与贡献

我们非常欢迎用户对于 MMHuman3D 做出的任何贡献，可以参考 [CONTRIBUTION.md](.github/CONTRIBUTING.md) 文件了解更多细节。

## 致谢

MMHuman3D是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## OpenMMLab的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱与测试基准
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用3D目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 新一代生成模型工具箱
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d)：OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMDeploy](https://github.com/open-mmlab/mmdeploy):OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码联系管理员来加入我们的微信社区

<div align="center">
<img src="docs_zh-CN/imgs/wechat_assistant_qrcode.jpg" height="200" />
</div>

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 MMHuman3D 团队的[官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=EKJmcIiO)

<div align="center">
<img src="docs_zh-CN/imgs/zhihu_qrcode.jpg" height="400" />  <img src="docs_zh-CN/imgs/mmhuman3d_qq_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
