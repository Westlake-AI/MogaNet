"""
An example to use fvcore to count MACs.

please install the following packages
`pip install timm fvcore`

Example command:
python get_flops.py --model moganet_tiny
"""
import argparse
import torch
from timm.models import create_model
from fvcore.nn import FlopCountAnalysis, flop_count_table

import models  # register_model for MogaNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='moganet_tiny',
        help='model name')
    parser.add_argument(
        '--img_size',
        type=int, default=224,
        metavar='N',
        help='Image patch size (default: None => model default)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    model = create_model(args.model)
    model.eval()
    # print(model)

    input = torch.rand(1, 3, args.img_size, args.img_size)

    # Please note that FLOP here actually means MAC.
    flop = FlopCountAnalysis(model, input)
    print(flop_count_table(flop, max_depth=4))
    print('MACs (G) of {}: {:.3f}'.format(args.model, flop.total() / 1e9))
