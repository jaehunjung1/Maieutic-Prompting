# Maieutic Prompting

This is an official code repository for our paper **Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations**.

## Requirements
```shell
pip install -r requirements.txt
```

## Generation
In order to generate maieutic tree for a given set of questions, run `main_generate.py`. We provide the code and arguments to run generation for the 3 datasets - Com2Sense, CSQA 2.0, CREAK used in the paper. To simply run generation for the dev split of these datasets, run
```shell
python main_generate.py --dataset_name=${DATASET_NAME}
```
The code reads dataset file in `./data/{datset_name}/1_gen/dev.Q.json` as input and outputs the pickled list of trees in `./data/{dataset_name}/1_gen/dev.G.pkl`.
* We use `treelib` library to represent maieutic tree. For further documentation please refer to [Official Doc](https://treelib.readthedocs.io/en/latest/).

The generation code requires OpenAI API key as an environment variable. For easier access to our method, we provide pre-generated files of maieutic tree for each dataset in `./data/{dataset_name}/1_gen/dev.G.pkl`.

## Inference
To run inference, run
```shell
python main_inference.py --device_id=${DEVICE_ID} --dataset_name=${DATASET_NAME}
```
`device_id` denotes the id of GPU to load the verifier model. The code reads dataset file in `./data/{datset_name}/1_gen/dev.Q.json` and maieutic tree file in `./data/{dataset_name}/1_gen/dev.G.pkl`
