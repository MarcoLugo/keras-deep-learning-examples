#!/bin/bash

rm -rf input
python3 create_images.py 128 1800 2
python3 split_data.py 0.2
