_base_ = "fpn_moganet_base_40k_ade20k.py"

gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000 // gpu_multiples)
