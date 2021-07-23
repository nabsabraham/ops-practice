#!/usr/bin/env bash
rm -rf logs/
rm -rf models/
mkdir models

torch-model-archiver --model-name BERTQA --version 1.0 --handler handler.py -f 

mv BERTQA.mar models/

torchserve --start --model-store models --models qa=BERTQA.mar --ncs
