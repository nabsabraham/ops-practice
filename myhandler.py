import logging
import torch
import torch.nn.functional as F
import io
import transformers
from ts.torch_handler.base_handler import BaseHandler

from transformers import pipeline



class MyHandler(BaseHandler):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pipeline =  pipeline(
                            "question-answering",
                            model="mrm8488/bert-tiny-finetuned-squadv2",
                            tokenizer="mrm8488/bert-tiny-finetuned-squadv2"
                        )


    def inference(self, data):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        outs = self.pipeline(data)
        return outs

    def postprocess(self, data):
        return data