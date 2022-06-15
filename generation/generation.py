import itertools

import math
import dataclasses
from dataclasses import dataclass
from typing import List, Union, Tuple

import openai
from transformers import GPT2Tokenizer
from treelib import Tree


@dataclass
class PromptConfig:
    engine: str = "text-davinci-001"
    max_tokens: int = 64
    temperature: float = 1
    top_p: float = 1
    logprobs: int = 5
    n: int = 3
    echo: bool = False


class GenerationWrapper:
    """Given question, generate maieutic tree"""

    def __init__(self, generation_prefix: Union[str, dict], belief_prefix: Union[str, dict],
                 negation_prefix: str, question_prefix: str, max_depth: int = 2):
        self.generation_prefix = generation_prefix
        self.belief_prefix = belief_prefix
        self.negation_prefix = negation_prefix
        self.question_prefix = question_prefix

        self.generation_config = PromptConfig()
        self.generation_config2 = PromptConfig(
            temperature=0.5,
            top_p=1,
            n=1
        )
        self.negation_config = PromptConfig(
            temperature=0,
            top_p=1,
            n=1
        )
        self.belief_config = PromptConfig(
            temperature=0,
            top_p=1,
            n=1,
        )
        self.question_config = PromptConfig(
            temperature=0,
            top_p=1,
            n=1
        )
        self.max_depth = max_depth

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")

    def create_maieutic_graph(self, Q: str, Q_tilde: str = None):
        # Initialize G
        G = Tree()
        if Q_tilde is None:
            Q, Q_tilde = self.prompt_Q_tilde(Q)
        G_blf, G_int = self.prompt_belief(Q, Q_tilde)
        G.create_node(Q, "Q", data={
            "E": Q,
            "E_tilde": Q_tilde,
            "blf": G_blf,
            "int": G_int
        })

        for depth in range(1, self.max_depth + 1):
            generation_config = self.generation_config if depth == 1 else self.generation_config2

            parents_to_generate_from = list(filter(lambda node: not node.data["int"], G.leaves()))
            for parent_node in parents_to_generate_from:
                new_E_T_list = self.prompt_E_T(parent_node.data["E"], dataclasses.replace(generation_config))
                new_E_T_tilde_list = [self.prompt_tilde(E_T) for E_T in new_E_T_list]

                for idx, (E_T, E_T_tilde) in enumerate(zip(new_E_T_list, new_E_T_tilde_list)):
                    node_identifier = f"{parent_node.identifier}T{idx}"
                    E_blf, E_int = self.prompt_belief(E_T, E_T_tilde)
                    G.create_node(E_T, node_identifier, parent=parent_node.identifier, data={
                        "E": E_T,
                        "E_tilde": E_T_tilde,
                        "blf": E_blf,
                        "int": E_int,
                    })

                new_E_F_list = self.prompt_E_F(parent_node.data["E"], dataclasses.replace(generation_config))
                new_E_F_tilde_list = [self.prompt_tilde(E_F) for E_F in new_E_F_list]
                for idx, (E_F, E_F_tilde) in enumerate(zip(new_E_F_list, new_E_F_tilde_list)):
                    node_identifier = f"{parent_node.identifier}F{idx}"
                    E_blf, E_int = self.prompt_belief(E_F, E_F_tilde)
                    G.create_node(E_F, node_identifier, parent=parent_node.identifier, data={
                        "E": E_F,
                        "E_tilde": E_F_tilde,
                        "blf": E_blf,
                        "int": E_int,
                    })

        # Remove branches that lead to logically not integral leaf nodes
        integral_leaf_nodes = [node.identifier for node in G.leaves() if node.data["int"]]
        paths_to_integral_leaves = [path for path in G.paths_to_leaves() if path[-1] in integral_leaf_nodes]
        nodes_not_to_remove = set(itertools.chain.from_iterable(paths_to_integral_leaves))

        nodes_before_removal = list(G.nodes.keys())
        for node in nodes_before_removal:
            if node in G and node not in nodes_not_to_remove:
                G.remove_node(node)

        return G

    def prompt_E_T(self, Q: str, generation_config: PromptConfig) -> List[str]:
        """Generate E_T from Q, using generation_config"""
        prompt_str = self.create_E_T_prompt(Q)
        num_Es_to_generate = generation_config.n
        E_T_list = []

        while len(E_T_list) < num_Es_to_generate:
            generation_config.n = num_Es_to_generate - len(E_T_list)
            response = openai.Completion.create(
                prompt=prompt_str,
                **generation_config.__dict__,
            )
            E_T_list.extend(GenerationWrapper.filter_generated_explanations(response.choices))

        return E_T_list[:num_Es_to_generate]

    def prompt_E_F(self, Q: str, generation_config: PromptConfig) -> List[str]:
        """Generate E_F from Q, using generation_config"""
        prompt_str = self.create_E_F_prompt(Q)
        num_Es_to_generate = generation_config.n
        E_F_list = []

        while len(E_F_list) < num_Es_to_generate:
            generation_config.n = num_Es_to_generate - len(E_F_list)
            response = openai.Completion.create(
                prompt=prompt_str,
                **generation_config.__dict__,
            )
            E_F_list.extend(GenerationWrapper.filter_generated_explanations(response.choices))

        return E_F_list[:num_Es_to_generate]

    def prompt_tilde(self, E: str):
        """
        Generate E_tilde given E
        """
        prompt_str = self.create_negation_prompt(E)
        response = openai.Completion.create(
            prompt=prompt_str,
            **self.negation_config.__dict__,
        )
        E_tilde = GenerationWrapper.filter_generated_explanations(response.choices)[0]

        return E_tilde

    def prompt_belief(self, Q: str, Q_tilde: str):
        """
        Compute belief by comparing p(True|E) and p(True|not E).
        return p(True|E), p(True|not E), logical_integrity
        """
        Q_E_Q_prob = self.prompt_true_given_Q(Q)
        Q_tilde_E_Q_tilde_prob = self.prompt_true_given_Q(Q_tilde)
        probs = (Q_E_Q_prob, Q_tilde_E_Q_tilde_prob)

        if None not in probs:
            integrity = GenerationWrapper.logical_integrity(Q_E_Q_prob, Q_tilde_E_Q_tilde_prob)
        else:
            integrity = False

        return probs, integrity

    def prompt_true_given_Q(self, Q: str):
        """
        Compute likelihood p(True|Q), and check whether max answer is True or False
        """
        prompt_str = self.create_belief_prompt(Q)

        response = openai.Completion.create(
            prompt=prompt_str,
            **self.belief_config.__dict__,
        )
        true_given_Q = self.retrieve_true_prob(response.choices[0])

        return true_given_Q

    def retrieve_true_prob(self, choice):
        """
        Given OpenAI choice, return likelihood of "true" in the generated text.
        Return None otherwise.
        """
        generated_text = choice.text
        if ". Therefore, the statement is" in generated_text:
            token_index_list = self.tokenizer.batch_decode(self.tokenizer(generated_text).input_ids)
            true_or_false_index = len(token_index_list) - 2
            true_logprob = choice.logprobs.top_logprobs[true_or_false_index].get(" true", -math.inf)

            if not math.isinf(true_logprob):
                return math.exp(true_logprob)

        return None

    def prompt_Q_tilde(self, question: str, refine: bool = True) -> Tuple[str, str]:
        if refine and self.question_prefix is not None:
            prompt_str = self.create_Q_prompt(question)

            response = openai.Completion.create(
                prompt=prompt_str,
                **self.generation_config.__dict__
            )
            refined_Q = GenerationWrapper.filter_generated_question(response.choices)[0]
        else:
            refined_Q = question

        prompt_str = self.create_Q_tilde_prompt(refined_Q)

        response = openai.Completion.create(
            prompt=prompt_str,
            **self.generation_config.__dict__
        )
        Q_tilde = GenerationWrapper.filter_generated_question(response.choices)[0]

        return refined_Q, Q_tilde

    @staticmethod
    def filter_generated_explanations(explanations: List):
        # Extract string explanations
        filtered_explanations = [explanation.text.strip() for explanation in explanations]

        # Filter out empty string / those not ending with "."
        filtered_explanations = list(filter(lambda exp: len(exp) > 0 and exp.endswith("."), filtered_explanations))

        # Upper case the first letter
        filtered_explanations = [explanation[0].upper() + explanation[1:] for explanation in filtered_explanations]

        # If there's none left, just add the first one
        if len(filtered_explanations) == 0:
            filtered_explanations.append(explanations[0].text.strip())

        # Remove duplicates
        filtered_explanations = list(dict.fromkeys(filtered_explanations))

        return filtered_explanations

    @staticmethod
    def filter_generated_question(questions: List):
        # Extract string questions
        filtered_questions = [question.text.strip() for question in questions]
        return filtered_questions

    @staticmethod
    def logical_integrity(prob1: float, prob2: float):
        return abs(prob1 - prob2) > 0.45

    def create_negation_prompt(self, proposition: str):
        return f"{self.negation_prefix}\n" \
               f"A: {proposition}\n" \
               f"B: The statement is false."

    def create_belief_prompt(self, question: str):
        belief_prefix = self.belief_prefix
        question = question[:-1] + "?"
        return f"{belief_prefix}\n" \
               f"Q: {question}\n" \
               f"A:"

    def create_E_T_prompt(self, question: str):
        generation_prefix = self.generation_prefix
        question = question[:-1] + "?"
        return f"{generation_prefix}\n" \
               f"Q: {question}\n" \
               f"A: This statement is true, because"

    def create_E_F_prompt(self, question: str):
        generation_prefix = self.generation_prefix
        question = question[:-1] + "?"
        return f"{generation_prefix}\n" \
               f"Q: {question}\n" \
               f"A: This statement is false, because"

    def create_Q_prompt(self, question: str):
        question = question[:-1] + "."
        return f"{self.question_prefix}\n" \
               f"Q: {question}\n" \
               f"A: The statement is true."

    def create_Q_tilde_prompt(self, question: str):
        question = question[:-1] + "."
        return f"{self.negation_prefix}\n" \
               f"Q: {question}\n" \
               f"A: The statement is false."
