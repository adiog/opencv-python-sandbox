#!/usr/bin/env bash

if [[ -z "${VIRTUAL_ENV}" ]]; then
    . venv_init.sh
fi
python main.py input.png
