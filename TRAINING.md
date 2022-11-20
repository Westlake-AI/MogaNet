# Training

We provide ImageNet-1K training commands here. Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## ImageNet-1K Training 

Taking MogaNet-T as an example, you can use the following command to run this experiment on a single machine: 
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model moganet_tiny --input_size 224 --drop_path 0.1 \
--batch_size 128 --lr 1e-3 --weight_decay 0.04 --update_freq 1 \
--aa rand-m7-mstd0.5-inc1 --mixup 0.1 \
--model_ema false --model_ema_eval false \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

- Here, the effective batch size = `--nproc_per_node` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `8*128*1 = 1024`. Running on one machine, we can increase `update_freq` and reduce `--batch_size` to avoid OOM issues while keeping the total batch size unchanged.

To train other MogaNet variants, `--model` and `--drop_path` need to be changed. Examples with single-machine commands are given below:


<details>
<summary>
MogaNet-XT
</summary>
Single-machine (8GPUs) with the input size of 224:

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model moganet_xtiny --input_size 224 --drop_path 0.05 \
--batch_size 128 --lr 1e-3 --weight_decay 0.03 --update_freq 1 \
--aa rand-m7-mstd0.5-inc1 --mixup 0.1 \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
</details>

<details>
<summary>
MogaNet-Tiny
</summary>
Single-machine (8GPUs) with the input size of 224:

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model moganet_tiny --input_size 224 --drop_path 0.1 \
--batch_size 128 --lr 1e-3 --weight_decay 0.04 --update_freq 1 \
--aa rand-m7-mstd0.5-inc1 --mixup 0.1 \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

Single-machine (8GPUs) with the input size of 256:

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model moganet_tiny --input_size 256 --drop_path 0.1 \
--batch_size 128 --lr 1e-3 --weight_decay 0.04 --update_freq 1 \
--aa rand-m7-mstd0.5-inc1 --mixup 0.1 \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
</details>

<details>
<summary>
MogaNet-Small
</summary>
Single-machine (8GPUs) with the input size of 224 with EMA (you can evaluate it without EMA):

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model moganet_small --input_size 224 --drop_path 0.1 \
--batch_size 128 --lr 1e-3 --weight_decay 0.05 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
</details>

<details>
<summary>
MogaNet-Base
</summary>
Single-machine (8GPUs) with the input size of 224 with EMA:

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model moganet_base --input_size 224 --drop_path 0.2 \
--batch_size 128 --lr 1e-3 --weight_decay 0.05 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
</details>

<details>
<summary>
MogaNet-Large
</summary>
Single-machine (8GPUs) with the input size of 224 with EMA:

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model moganet_large --input_size 224 --drop_path 0.3 \
--batch_size 128 --lr 1e-3 --weight_decay 0.05 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
</details>
