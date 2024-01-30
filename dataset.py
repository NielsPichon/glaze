import argparse
import pathlib
import random

import tqdm


def make_data_splits(src: pathlib.Path, eval_split = 0.1, test_split = 0.1):
    train = src / "train"
    eval = src / "eval"
    test = src / "test"
    for split in [train, eval, test]:
        split.mkdir(exist_ok=True, parents=True)

    data = list(src.glob('*.npy'))
    random.shuffle(data)

    eval_thresh = len(data) * (1 - eval_split - test_split)
    test_thresh = len(data) * (1 - test_split)
    for i, line in enumerate(tqdm.tqdm(data)):
        if i < eval_thresh:
            dest = train
        elif i < test_thresh:
            dest = eval
        else:
            dest = test
        line.rename(dest / line.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=pathlib.Path, default="data")
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    args = parser.parse_args()
    make_data_splits(args.src, args.eval_split, args.test_split)
