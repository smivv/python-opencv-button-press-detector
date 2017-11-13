#!/usr/bin/env bash

# virtual environment installation
virtualenv -p python3 .env
source .env/bin/activate

# installation requirements
pip install numpy
pip install opencv-python
deactivate

# start script
./.env/bin/python3 main.py