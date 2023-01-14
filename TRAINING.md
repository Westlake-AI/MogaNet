# Training

We provide ImageNet-1K training commands here. Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## ImageNet-1K Training

Taking MogaNet-T as an example, you can use the following command to run this experiment on a single machine (8GPUs): 
```
python -m torch.distributed.launch --nproc_per_node=8 train.py \
--model moganet_tiny --input_size 224 --drop_path 0.1 \
--epochs 300 --batch_size 128 --lr 1e-3 --weight_decay 0.04 \
--aa rand-m7-mstd0.5-inc1 --crop_pct 0.9 --mixup 0.1 \
--amp --native_amp \
--data_dir /path/to/imagenet-1k \
--experiment /path/to/save_results
```

- Batch size scaling. The effective batch size is equal to `--nproc_per_node` * `--batch_size`. In the example above, the effective batch size is `8*128 = 1024`. Running on one machine, we can reduce `--batch_size` and use `--amp` to avoid OOM issues while keeping the total batch size unchanged.
- Learning rate scaling. The default learning rate setting is `lr=1e-3 / bs1024`. We find that `lr=2e-3 / bs1024` and `lr=1e-3 / bs512` produce better performances and more stable training for MogaNet-XT/T and MogaNet-S/B/L.
- EMA. We adopt the EMA trick for MogaNet-S/B/L using `--model_ema` and `--model_ema_decay 0.9999`.

To train other MogaNet variants, `--model` and `--drop_path` need to be changed. Examples with single-machine commands are given below:


<details>
<summary>
MogaNet-XT
</summary>
Single-machine (8GPUs) with the input size of 224:

```
python -m torch.distributed.launch --nproc_per_node=8 train.py \
--model moganet_xtiny --input_size 224 --drop_path 0.05 \
--epochs 300 --batch_size 128 --lr 1e-3 --weight_decay 0.03 \
--aa rand-m7-mstd0.5-inc1 --crop_pct 0.9 --mixup 0.1 \
--amp --native_amp \
--data_dir /path/to/imagenet-1k \
--experiment /path/to/save_results
```
</details>

<details>
<summary>
MogaNet-Tiny
</summary>
Single-machine (8GPUs) with the input size of 224:

```
python -m torch.distributed.launch --nproc_per_node=8 train.py \
--model moganet_tiny --input_size 224 --drop_path 0.1 \
--epochs 300 --batch_size 128 --lr 1e-3 --weight_decay 0.04 \
--aa rand-m7-mstd0.5-inc1 --crop_pct 0.9 --mixup 0.1 \
--amp --native_amp \
--data_dir /path/to/imagenet-1k \
--experiment /path/to/save_results
```

Single-machine (8GPUs) with the input size of 256:

```
python -m torch.distributed.launch --nproc_per_node=8 train.py \
--model moganet_tiny --input_size 256 --drop_path 0.1 \
--epochs 300 --batch_size 128 --lr 1e-3 --weight_decay 0.04 \
--aa rand-m7-mstd0.5-inc1 --crop_pct 0.9 --mixup 0.1 \
--amp --native_amp \
--data_dir /path/to/imagenet-1k \
--experiment /path/to/save_results
```
</details>

<details>
<summary>
MogaNet-Small
</summary>
Single-machine (8GPUs) with the input size of 224 with EMA (you can evaluate it without EMA):

```
python -m torch.distributed.launch --nproc_per_node=8 train.py \
--model moganet_small --input_size 224 --drop_path 0.1 \
--epochs 300 --batch_size 128 --lr 1e-3 --weight_decay 0.05 \
--crop_pct 0.9 --min_lr 1e-5 \
--model_ema --model_ema_decay 0.9999 \
--data_dir /path/to/imagenet-1k \
--experiment /path/to/save_results
```
</details>

<details>
<summary>
MogaNet-Base
</summary>
Single-machine (8GPUs) with the input size of 224 with EMA:

```
python -m torch.distributed.launch --nproc_per_node=8 train.py \
--model moganet_base --input_size 224 --drop_path 0.2 \
--epochs 300 --batch_size 128 --lr 1e-3 --weight_decay 0.05 \
--crop_pct 0.9 --min_lr 1e-5 \
--model_ema --model_ema_decay 0.9999 \
--data_dir /path/to/imagenet-1k \
--experiment /path/to/save_results
```
</details>

<details>
<summary>
MogaNet-Large
</summary>
Single-machine (8GPUs) with the input size of 224 with EMA:

```
python -m torch.distributed.launch --nproc_per_node=8 train.py \
--model moganet_large --input_size 224 --drop_path 0.3 \
--epochs 300 --batch_size 128 --lr 1e-3 --weight_decay 0.05 \
--crop_pct 0.9 --min_lr 1e-5 \
--model_ema --model_ema_decay 0.9999 \
--data_dir /path/to/imagenet-1k \
--experiment /path/to/save_results
```
</details>


## ImageNet-1K Validation

Taking MogaNet-T as an example, you can use the following command to run the validation on ImageNet val set: 
```
python validate.py \
--model moganet_tiny --input_size 224 --crop_pct 0.9 \
--data_dir /path/to/imagenet-1k \
--checkpoint /path/to/checkpoint.tar.gz
```

- In the example above, we test the model in 224x224 resolutions (modified by `--input_size`) without using the EMA model. Please add `--use_ema` to enable EMA evaluation for MogaNet-Small, MogaNet-Base, and MogaNet-Large. Running on one machine, we can use `--num_gpu` and use `--amp` to avoid OOM issues.

To evaluate other MogaNet variants, `--model` and `--use_ema` need to be changed. Examples with single-machine commands are given below:

<details>
<summary>
MogaNet-XT
</summary>
Single-machine (8GPUs) with the input size of 224:

```
python validate.py \
--model moganet_xtiny --input_size 224 --crop_pct 0.9 --num_gpu 8 \
--data_dir /path/to/imagenet-1k \
--checkpoint /path/to/checkpoint.tar.gz
```
</details>

<details>
<summary>
MogaNet-Tiny
</summary>
Single-machine (8GPUs) with the input size of 224:

```
python validate.py \
--model moganet_tiny --input_size 224 --crop_pct 0.9 --num_gpu 8 \
--data_dir /path/to/imagenet-1k \
--checkpoint /path/to/checkpoint.tar.gz
```
</details>

<details>
<summary>
MogaNet-Small
</summary>
Single-machine (8GPUs) with the input size of 224:

```
python validate.py \
--model moganet_small --input_size 224 --crop_pct 0.9 --num_gpu 8 --use_ema \
--data_dir /path/to/imagenet-1k \
--checkpoint /path/to/checkpoint.tar.gz
```
</details>

<details>
<summary>
MogaNet-Base
</summary>
Single-machine (8GPUs) with the input size of 224:

```
python validate.py \
--model moganet_base --input_size 224 --crop_pct 0.9 --num_gpu 8 --use_ema \
--data_dir /path/to/imagenet-1k \
--checkpoint /path/to/checkpoint.tar.gz
```
</details>

<details>
<summary>
MogaNet-Large
</summary>
Single-machine (8GPUs) with the input size of 224:

```
python validate.py \
--model moganet_large --input_size 224 --crop_pct 0.9 --num_gpu 8 --use_ema \
--data_dir /path/to/imagenet-1k \
--checkpoint /path/to/checkpoint.tar.gz
```
</details>
