from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import RobertaConfig
import functions_manager as fm
import torch


class CodeEmbedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.config = RobertaConfig.from_pretrained('microsoft/graphcodebert-base', output_hidden_states=True)
        self.model = AutoModelForMaskedLM.from_pretrained("microsoft/graphcodebert-base", config=self.config)
        self.function_manager = fm.FunctionManager()

    def _getRawFunctionEmbedding(self, function_name: str):
        pass

    def getPerfectFunctionEmbedding(self, function_name: str, max_length=300):
        pass


# hidden state
# logits
# shape