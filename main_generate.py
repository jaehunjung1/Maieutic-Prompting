import os
import pickle

import jsonlines
import openai
from argparse import ArgumentParser
from tqdm import tqdm

from generation.generation import GenerationWrapper
from generation.generation_config import retrieve_prompt_prefix


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")

    args = ArgumentParser()
    args.add_argument("--dataset_name", default="Com2Sense", type=str)
    args = args.parse_args()
    args.data_filename = f"./data/{args.dataset_name}/1_gen/dev.Q.json"
    args.out_filename = f"./data/{args.dataset_name}/1_gen/dev.G.pkl"

    prompt_prefix_dict = retrieve_prompt_prefix(args.dataset_name)
    generator = GenerationWrapper(prompt_prefix_dict["abductive"], prompt_prefix_dict["belief"],
                                  prompt_prefix_dict["negation"], prompt_prefix_dict["question"], max_depth=2)

    if os.path.exists(args.out_filename):
        with open(args.out_filename, "rb") as f:
            orig_out_list = pickle.load(f)
    else:
        orig_out_list = []

    with jsonlines.open(args.data_filename, "r") as f:
        samples = list(f)[len(orig_out_list):]

    out_list = orig_out_list
    for sample_idx, sample in tqdm(enumerate(samples), total=len(samples)):
        G = generator.create_maieutic_graph(sample["Q"], sample["Q_tilde"])
        out_list.append(G)

        if sample_idx % 100 == 0:  # backup save
            with open(args.out_filename, "wb") as f:
                pickle.dump(out_list, f)

    with open(args.out_filename, "wb") as f:
        pickle.dump(out_list, f)
