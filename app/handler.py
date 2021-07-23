from abc import ABC
import json
import logging
import os
import ast
import torch
import transformers
from transformers import AutoTokenizer,AutoModelForQuestionAnswering, pipeline
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s",transformers.__version__)


class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithmfor Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # download model pipeline
        self.model = pipeline(
            "question-answering",
            model="mrm8488/bert-tiny-finetuned-squadv2",
            tokenizer="mrm8488/bert-tiny-finetuned-squadv2"
        )

        self.initialized = True


    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of read responses for the 
        """
        logger.info(len(requests))
        for idx, data in enumerate(requests):
            input_text = data.get("data")

            if input_text is None:
                input_text = data.get("body")

            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')

            logger.info("Received text: '%s'", input_text)

            question_context = ast.literal_eval(input_text)

        return question_context


    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of question-context pairs read from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        outputs = self.model(input_batch)
        
        return [outputs]


    def postprocess(self, inference_output):

        logger.info(inference_output)
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input question-context pair.
        Returns:
            (list): Returns a list of the Predictions.
        """

        return (inference_output)
