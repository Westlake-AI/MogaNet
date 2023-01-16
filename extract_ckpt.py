"""
Extract parameters in the trained checkpoint.

Example command:
python extract_ckpt.py /path/to/checkpoint /path/to/output.pth.tar
"""
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output', type=str, help='destination file name')
    parser.add_argument(
        '--ema_only',
        action='store_true',
        help='only ema params as state_dict')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth.tar")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author="MogaNet")
    keep_keys = ['arch', 'state_dict', 'state_dict_ema', 'metric']
    for key in keep_keys:
        if ck.get(key, None) is not None:
            output_dict[key] = ck[key]

    # copy ema params to `state_dict`
    if args.ema_only:
        output_dict['state_dict'] = output_dict.pop('state_dict_ema')

    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()
