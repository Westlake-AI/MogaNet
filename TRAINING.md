# Training

We provide ImageNet-1K training commands here. Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## ImageNet-1K Training 

Taking MogaNet-T as an example, you can use the following command to run this experiment on a single machine: 
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model moganet_tiny --drop_path 0.1 \
--batch_size 128 --lr 1e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

- Here, the effective batch size = `--nproc_per_node` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `8*128*1 = 1024`. Running on one machine, we can increase `update_freq` and reduce `--batch_size` to avoid OOM issues while keeping the total batch size unchanged.

To train other MogaNet variants, `--model` and `--drop_path` need to be changed. Examples with single-machine commands are given below:

<!-- 
<details>
<summary>
ConvNeXt-S
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_small --drop_path 0.4 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_small --drop_path 0.4 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
</details>
<details>
<summary>
ConvNeXt-B
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_base --drop_path 0.5 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_base --drop_path 0.5 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

</details>
<details>
<summary>
ConvNeXt-L
</summary>

Multi-node
```
python run_with_submitit.py --nodes 8 --ngpus 8 \
--model convnext_large --drop_path 0.5 \
--batch_size 64 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_large --drop_path 0.5 \
--batch_size 64 --lr 4e-3 --update_freq 8 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

</details>
 -->
