#!/bin/bash

cd build && ninja && cd ..

for exe in build/real_time_seg "python python/video_seg.py"; do
	for device in cpu cuda; do
		echo $exe $device
		$exe example_data/videos/d1_im.mp4 example_data/models/model_jit.pt $device
	done
done
