torchserve --stop

# clean up
rm -r logs/

# generate model archive file 
torch-model-archiver --model-name BERTQA --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./tokenhandler.py --extra-files "Transformer_model/config.json,./setup_config.json" -f

# mv to model_store
mv BERTQA.mar model_store

# serve
torchserve --start --model-store model_store --models my_tc=BERTQA.mar --ncs