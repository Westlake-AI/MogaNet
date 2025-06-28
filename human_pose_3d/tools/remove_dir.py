import re
import os
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description='Remove files')
    parser.add_argument('dir_root', help='input dir path')
    parser.add_argument('rm_name', type=str, help='filename to remove')
    parser.add_argument('--reg', action='store_true', default=False,
                        help='whether to use reg')
    args = parser.parse_args()
    return args


def process_remove(dir_root, rm_name=None, use_reg=False):
    assert rm_name is not None
    assert os.path.exists(dir_root)
    dir_list = os.listdir(path=dir_root)
    for dir in dir_list:
        _path = os.path.join(dir_root, dir, rm_name)
        try:
            if use_reg:
                os.system("rm {}".format(_path))  # only support file
            elif os.path.isfile(_path):
                subprocess.Popen(['rm', _path])
            else:
                subprocess.Popen(['rm', '-r', _path])
        except:
            print('fail to remove:', _path)


def main():
    args = parse_args()
    process_remove(args.dir_root, args.rm_name, args.reg)


if __name__ == '__main__':
    main()
