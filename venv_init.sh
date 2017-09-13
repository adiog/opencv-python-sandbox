#!/usr/bin/env bash

sudo apt install python3 python3-virtualenv
virtualenv -p python3 venv
. venv/bin/activate
pip install numpy opencv-python
