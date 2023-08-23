#!/usr/bin/env python

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.cuda as tcuda
import torch.utils.data.distributed
from torch.utils.data import DataLoader, SequentialSampler

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc
from idisc.utils.validation import visual
from idisc.utils import (DICT_METRICS_DEPTH, DICT_METRICS_NORMALS,
                         RunningMetric, validate)


def main(config: Dict[str, Any], args: argparse.Namespace):
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = IDisc.build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()

    save_file = args.save_file

    save_dir = os.path.join(args.base_path, config["data"]["test_data_root"])
    assert hasattr(
        custom_dataset, config["data"]["train_dataset"]
    ), f"{config['data']['train_dataset']} not a custom dataset"
    valid_dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
        test_mode=True, base_path=save_dir, crop=config["data"]["crop"],
    )
    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch,
        num_workers=args.num,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
    )

    print("Start visual...")
    with torch.no_grad():
        visual(
            model,
            valid_loader,
            save_file
        )


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")

    parser.add_argument("--config-file", default='/home/ana/Study/CVPR/idisc/configs/ycbv/ycbv.json', type=str)
    parser.add_argument("--text_file", default='splits/ycbv/ycbv_test.txt', type=str)
    parser.add_argument("--model-file", default='output/YCBVDataset-best.pt', type=str)
    parser.add_argument("--save-file", default='output/images', type=str)
    parser.add_argument("--base-path", default='/home/ana/Study/CVPR/bop_ycbv/bop_datasets/ycbv')
    parser.add_argument("--batch", default=2, type=int)
    parser.add_argument("--num", default=2, type=int)

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    main(config, args)

