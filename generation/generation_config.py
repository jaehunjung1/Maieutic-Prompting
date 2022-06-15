from typing import Final


PROMPT_CONFIGS: Final[dict] = {
    "CSQA2": {
        "abductive": "./generator/prompts/CSQA2/abductive.prompt.txt",
        "belief": "./generator/prompts/CSQA2/belief.prompt.txt",
        "negation": "./generator/prompts/CSQA2/negation.prompt.txt",
        "question": "./generator/prompts/CSQA2/Q.prompt.txt",
    },
    "Com2Sense": {
        "abductive": "generator/prompts/Com2Sense/abductive.prompt.txt",
        "belief": "generator/prompts/Com2Sense/belief.prompt.txt",
        "negation": "generator/prompts/Com2Sense/negation.prompt.txt",
        "question": "./generator/prompts/Com2Sense/Q.prompt.txt",
    },
    "CREAK": {
        "abductive": "generator/prompts/CREAK/abductive.prompt.txt",
        "belief": "generator/prompts/CREAK/belief.prompt.txt",
        "negation": "./generator/prompts/CREAK/negation.prompt.txt",
        "question": "./generator/prompts/CREAK/Q.prompt.txt",
    }
}


def retrieve_prompt_prefix(dataset_name: str) -> dict:
    def open_file(data: dict):
        return {
            key: open_file(value) if isinstance(value, dict) else open(value, "r").read()
            for key, value in data.items()
        }
    filename_dict = PROMPT_CONFIGS[dataset_name]

    return open_file(filename_dict)

