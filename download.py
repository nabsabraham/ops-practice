
from myhandler import MyHandler
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, set_seed

""" This function, save the checkpoint, config file along with tokenizer config and vocab files
    of a transformer model of your choice.
"""
print('Transformers version',transformers.__version__)
set_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def transformers_model_dowloader():
    print("Download model and tokenizer bert-tiny-5-finetuned-squadv2")

    #loading pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-5-finetuned-squadv2")
    model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/bert-tiny-5-finetuned-squadv2")

    NEW_DIR = "./model"
    try:
        os.mkdir(NEW_DIR)
    except OSError:
        print ("Creation of directory %s failed" % NEW_DIR)
    else:
        print ("Successfully created directory %s " % NEW_DIR)

    print("Save model and tokenizer/ Torchscript model based on the setting from setup_config", 'in directory', NEW_DIR)
    model.save_pretrained(NEW_DIR)
    tokenizer.save_pretrained(NEW_DIR)            

    return 

if __name__== "__main__":

    transformers_model_dowloader()

#     torch-model-archiver --model-name tinybert --version 1.0 --serialized-file model/traced_model.pt --handler ./Transformer_handler_generalized.py --extra-files "./setup_config.json,./Seq_classification_artifacts/index_to_name.json"

# torch-model-archiver --model-name tinybert --version 1.0 --serialized-file model/pytorch_model.bin --handler myhandler.py --extra-files "model/config.json,./setup_config.json,./Seq_classification_artifacts/index_to_name.json"
torchserve --start --model-store model_store --models my_tc=tinybert.mar --ncs --ts-config config.properties

torchserve --model-store=model_store 

torchserve --start --model-store model_store --models bert=tinybert.mar --ts-config config.properties