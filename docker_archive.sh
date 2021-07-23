#!/bin/bash
docker run --rm \
    -v $(pwd):/home  \
    pytorch/torchserve:0.1-cpu \
    torch-model-archiver --model-name resnet34 \
    --version 1.0 \
    --serialized-file resnet34.pt \
    --extra-files ./index_to_name.json \
    --handler image_classifier  \
    --export-path model-store -f 

docker run --rm  \
	-it -p 3000:8080 -p 3001:8081 \
	-v $PWD/model_store:/home/model-server/model_store \
		pytorch/torchserve:latest torchserve \
	--model-store=/home/model-server/model_store 
