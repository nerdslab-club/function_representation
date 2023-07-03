from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import RobertaConfig
import functions_manager as fm
import torch
import torch.nn.functional as functional


def getShape(embedding):
    """Print and return the shape and length of an embedding

    :param embedding: The embedding tensor whom shape and length is to be calculated
    :return: Tuple as (shape, length)
    """
    shape = embedding.shape
    length = len(embedding)
    print(shape)
    print(length)
    return shape, length


class CodeEmbedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.config = RobertaConfig.from_pretrained('microsoft/graphcodebert-base', output_hidden_states=True)
        self.model = AutoModelForMaskedLM.from_pretrained("microsoft/graphcodebert-base", config=self.config)
        self.function_manager = fm.FunctionManager()

    def _getRawFunctionOutput(self, function_name: str):
        """Calculate the output from the function input

        :param function_name: Name of the function.
        :return: The output which include hidden states and logits
        """
        function_ref = self.function_manager.getNameToReference().get(function_name)
        func_raw_str = self.function_manager.getFunctionAsStringWithoutDocString(function_ref)
        inputs = self.tokenizer(func_raw_str, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs

    def _getRawFunctionEmbedding(self, function_name: str):
        """Calculate the hidden states embedding for the function tokens

        :param function_name: Name of the function.
        :return: hidden states embedding of size [1 * n * 768]
        """
        function_ref = self.function_manager.getNameToReference().get(function_name)
        func_raw_str = self.function_manager.getFunctionAsStringWithoutDocString(function_ref)

        inputs = self.tokenizer(func_raw_str, return_tensors="pt")

        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        return hidden_states

    def getPerfectFunctionEmbedding(self, function_name: str, max_length=300):
        """Reshape the last hidden state of the embedding into [max_length, 768] tensor

        :param function_name: Name of the function.
        :param max_length: Max length for token that is supported.
        :return: Last hidden state embedding of size [n * 768]
        """
        hidden_states_embedding = self._getRawFunctionEmbedding(function_name)

        # Pad the tensor to the desired shape [300, 768]
        padded_tensor = functional.pad(hidden_states_embedding[0],
                                       (0, 0, 0, max_length - hidden_states_embedding[0].shape[1]))
        reshaped_tensor = padded_tensor.squeeze()
        return reshaped_tensor

    def getLogits(self, function_name: str, max_length=300):
        """Calculate the logits for the given function.

        :param function_name: Name of the function.
        :param max_length: Max length for token that is supported.
        :return: Logits embedding.
        """
        output = self._getRawFunctionOutput(function_name)
        logits = output.logits
        # Pad the tensor to the desired shape [300, 768]
        padded_tensor = functional.pad(logits, (0, 0, 0, max_length - logits.shape[1]))
        reshaped_tensor = padded_tensor.squeeze()
        return reshaped_tensor