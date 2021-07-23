from abc import ABC
import json
import logging
import os
import ast
import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)
from ts.torch_handler.base_handler import BaseHandler
from captum.attr import LayerIntegratedGradients

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
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
                self.setup_config["model_name"],
                do_lower_case=self.setup_config["do_lower_case"],
            )

        self.model.eval()

        logger.info(
            "Transformer model from path %s loaded successfully", model_dir
        )

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # Question answering does not need the index_to_name.json file.
        if not self.setup_config["mode"] == "question_answering":
            if os.path.isfile(mapping_file_path):
                with open(mapping_file_path) as f:
                    self.mapping = json.load(f)
            else:
                logger.warning("Missing the index_to_name.json file.")
        self.initialized = True


    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')
            if self.setup_config["captum_explanation"] and not self.setup_config["mode"] == "question_answering":
                input_text_target = ast.literal_eval(input_text)
                input_text = input_text_target["text"]
            max_length = self.setup_config["max_length"]
            logger.info("Received text: '%s'", input_text)

            question_context = ast.literal_eval(input_text)
            question = question_context["question"]
            context = question_context["context"]
            inputs = self.tokenizer.encode_plus(question, context, pad_to_max_length=False, 
            add_special_tokens=True, return_tensors="pt", return_token_type_ids=True)

            logger.info(inputs)
            logger.info(f' question: {question}')
            logger.info(f' context: {context}')

            input_ids = inputs["input_ids"].to(self.device)
            token_ids = inputs["token_type_ids"].to(self.device)

            logger.info(input_ids)
            logger.info(token_ids)

            #attention_mask = inputs["attention_mask"].to(self.device)
            # making a batch out of the recieved requests
            # attention masks are passed for cases where input tokens are padded.
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    #attention_mask_batch = attention_mask
                    token_ids_batch = token_ids
                    
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    token_ids_batch = torch.cat((token_ids_batch, token_ids), 0)
                    #attention_mask_batch = torch.cat((attention_mask_batch, attention_mask), 0)
        #return (input_ids_batch, token_ids_batch, attention_mask_batch)
        return (input_ids_batch, token_ids_batch)


    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        logger.info(input_batch)
        input_ids_batch, token_ids_batch = input_batch
        inferences = []
        logger.info(input_ids_batch)
        outputs = self.model(input_ids_batch)#, token_type_ids=token_ids_batch)

        logger.info(outputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
 
        # print("This the output size for answer start scores from the question answering model", answer_start_scores.size())
        # print("This the output for answer start scores from the question answering model", answer_start_scores)
        # print("This the output size for answer end scores from the question answering model", answer_end_scores.size())
        # print("This the output for answer end scores from the question answering model", answer_end_scores)

        num_rows, num_cols = answer_start_scores.shape
        inferences = []
        for i in range(num_rows):
            answer_start_scores_one_seq = answer_start_scores[i].unsqueeze(0)
            answer_start = torch.argmax(answer_start_scores_one_seq)
            answer_end_scores_one_seq = answer_end_scores[i].unsqueeze(0)
            answer_end = torch.argmax(answer_end_scores_one_seq) + 1
            logger.info(f' answer start: {answer_start}')
            logger.info(f' answer end: {answer_end}')
            answer_tokens = self.tokenizer.convert_ids_to_tokens(input_ids_batch[i].tolist()[answer_start:answer_end])
            prediction = self.tokenizer.convert_tokens_to_string(answer_tokens)
            inferences.append(prediction)


        # ans_tokens = input_ids_batch[torch.argmax(answer_start_scores) : torch.argmax(answer_end_scores)+1]
        # logger.info(f' \n **** answer ids: {ans_tokens}')

        # answer_tokens = self.tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
        logger.info(f' \n **** answer tokens: {answer_tokens}')

        #all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids_batch)

        #inferences = self.tokenizer.convert_tokens_to_string(answer_tokens)

        
        logger.info("Model predicted: '%s'", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        res = []
        for pred in inference_output:
            res.append({'output': pred})
        return res
