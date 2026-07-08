#!/bin/bash

~/anaconda3/envs/vggt/bin/python export_to_viewer_raw.py \
    --co3d_dir /home/chuanruo/co3d_data \
    --co3d_anno_dir /home/chuanruo/co3d_data/annotations \
    --category apple \
    --num_frames 7 \
    --seed 42 \
    --output results_raw
