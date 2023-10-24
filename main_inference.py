import pickle
from argparse import ArgumentParser, Namespace

import jsonlines
import torch
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, AutoTokenizer

from inference.inference import InferenceWrapper
from inference.verifier import NLIModel


def parse_args() -> Namespace:
    args = ArgumentParser()
    args.add_argument("--device_id", type=int)
    args.add_argument("--dataset_name", default="Com2Sense", type=str)

    args = args.parse_args()
    args.device = torch.device(f"cuda:{args.device_id}")
    args.data_filename = f"./data/{args.dataset_name}/1_gen/dev.Q.json"
    args.G_filename = f"./data/{args.dataset_name}/1_gen/dev.G.pkl"

    return args


if __name__ == "__main__":
    args = parse_args()
    model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    InferenceWrapper.nli_model = NLIModel(model, tokenizer)

    with jsonlines.open(args.data_filename, "r") as f:
        samples = list(f)

    with open(args.G_filename, "rb") as f:
        G_samples = pickle.load(f)

    acc_result = [0, 0]  # [correct, incorrect]
    for sample, G in tqdm(zip(samples, G_samples), total=len(samples)):
        if G.size() == 1:
            inferred_answer = 1 if G["Q"].data["blf"][0] >= G["Q"].data["blf"][1] else -1
        elif G.size() > 1:
            score_list, correct_E_dict, graph2sat, belief, consistency = InferenceWrapper.infer(G)
            sum_score = sum([score[1] for score in score_list])
            inferred_answer = 1 if sum_score >= 0 else -1
        else:
            inferred_answer = 1

        # Record results
        gt_answer = 1 if sample["A"] else -1
        acc_result[0 if inferred_answer == gt_answer else 1] += 1

    print(f"Acc: {acc_result}")
