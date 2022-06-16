from typing import Union, List
from itertools import product

import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaForSequenceClassification


class NLIModel:
    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizerFast):
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer
        self.device = self.model.device

    def predict(self, premise: Union[str, List], hypothesis: Union[str, List]):
        input_encoding = self.tokenizer(premise, hypothesis,
                                        return_tensors='pt', padding=True)
        input_encoding = {k: v.to(self.device) for k, v in input_encoding.items()}

        with torch.no_grad():
            output = self.model(
                input_ids=input_encoding["input_ids"],
                attention_mask=input_encoding["attention_mask"],
            )

        return output.logits

    def predict_from_two_world(self, world1: List[str], world2: List[str]):
        pairs = list(product(world1, world2))
        sent1, sent2 = list(zip(*pairs))

        # World1 as premise, World2 as hypothesis
        out1 = self.predict(sent1, sent2)
        out1 = out1.view(len(world1), len(world2), -1).argmax(dim=-1)

        # World2 as premise, World1 as hypothesis
        out2 = self.predict(sent2, sent1)
        out2 = out2.view(len(world1), len(world2), -1).argmax(dim=-1)

        return out1, out2


def is_in_set_T(node_name: str):
    return node_name.count("F") % 2 == 0

