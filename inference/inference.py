from dataclasses import dataclass
from itertools import permutations
from typing import Tuple, Dict, List
from collections import OrderedDict

import torch.nn.functional as F
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from treelib import Tree

from inference.nli import NLIModel


@dataclass
class Node:
    identifier: str
    E: str
    blf: Tuple[float, float]
    integrity: bool


class InferenceWrapper:
    nli_model: NLIModel = None

    def __init__(self, nli_model: NLIModel):
        InferenceWrapper.nli_model = nli_model

    @staticmethod
    def infer(G: Tree):
        node_dict = InferenceWrapper.prepare_node_list(G)
        belief = InferenceWrapper.compute_belief(node_dict)
        consistency = InferenceWrapper.compute_consistency(node_dict)
        graph2sat, wcnf = InferenceWrapper.convert_graph_to_sat(node_dict, belief, consistency)

        with RC2(wcnf.copy()) as rc2:
            solution = rc2.compute()

        # Retrieve only the correct nodes from Max-SAT
        correct_E_dict = {}
        for name, node in node_dict.items():
            if solution[graph2sat[name] - 1] > 0:
                correct_E_dict[name] = node.E

        # Infer answer to Q
        if len(correct_E_dict) > 0:
            score_list, Q_consistency = InferenceWrapper.nli_with_Q(G["Q"].data["E"], G["Q"].data["E_tilde"],
                                                                    correct_E_dict)
        else:
            score_list = []
            Q_consistency = {}

        consistency = {pair: 1 if score > 0 else -1 for pair, score in consistency.items()} | Q_consistency

        return score_list, correct_E_dict, graph2sat, belief, consistency

    @staticmethod
    def prepare_node_list(G: Tree) -> OrderedDict:
        node_dict = OrderedDict()

        for node in G.all_nodes_itr():
            if node.identifier != "Q":
                if None in node.data["blf"] or node.data["blf"][0] > node.data["blf"][1]:
                    identifier = node.identifier
                    E = node.data["E"]
                    blf = node.data["blf"]
                    integrity = node.data["int"]
                else:
                    identifier = "not " + node.identifier
                    E = node.data["E_tilde"]
                    blf = (node.data["blf"][1], node.data["blf"][0])
                    integrity = node.data["int"]

                node_dict[identifier] = Node(
                    identifier=identifier,
                    E=E,
                    blf=blf,
                    integrity=integrity
                )

        return node_dict

    @staticmethod
    def compute_belief(node_dict: OrderedDict) -> Dict:
        belief = {}

        nodes_to_compute_blf = [node for node in node_dict.values() if node.integrity]

        for node in nodes_to_compute_blf:
            likelihood_feature = node.blf
            likelihood_belief = (likelihood_feature[0] - likelihood_feature[1]) / sum(likelihood_feature)
            belief[node.identifier] = likelihood_belief

        return belief

    @staticmethod
    def compute_consistency(node_dict: OrderedDict) -> Dict:
        consistency = {}

        if len(node_dict) > 1:
            nodes_to_compute_con = node_dict.keys()

            all_pair_list = list(permutations(nodes_to_compute_con, 2))
            all_pair_E_list = [(node_dict[name1].E, node_dict[name2].E) for name1, name2 in all_pair_list]

            all_pair_E_probs = F.softmax(InferenceWrapper.nli_model.predict(*zip(*all_pair_E_list)), dim=-1)

            for pair, probs in zip(all_pair_list, all_pair_E_probs):
                if probs.argmax() != 1:
                    consistency[pair] = (probs[2] - probs[0]).item()

        return consistency

    @staticmethod
    def nli_with_Q(Q: str, Q_tilde: str, correct_E_dict: Dict):
        name_list = list(correct_E_dict.keys())
        E_list = list(correct_E_dict.values())

        E_Q_labels = InferenceWrapper.nli_model.predict(
            E_list, [Q] * len(E_list)).argsort(dim=-1, descending=True).tolist()
        E_Q_tilde_labels = InferenceWrapper.nli_model.predict(
            E_list, [Q_tilde] * len(E_list)).argsort(dim=-1, descending=True).tolist()

        score_list = []
        for node_name, E_Q_label, E_Q_tilde_label in zip(name_list, E_Q_labels, E_Q_tilde_labels):
            score = 0
            if E_Q_label[0] == 2:
                score += 1
            elif E_Q_label[0] == 0:
                score += -1

            if E_Q_tilde_label[0] == 2:
                score += -1
            elif E_Q_tilde_label[0] == 0:
                score += 1
            score_list.append((node_name, score))

        Q_consistency = {(node_name, "Q"): score if abs(score) <= 1 else score / abs(score) for node_name, score in
                         score_list if score != 0}

        return score_list, Q_consistency

    @staticmethod
    def convert_graph_to_sat(node_dict: OrderedDict, belief: Dict, consistency: Dict):
        # Prepare mapping between G <=> SAT
        graph2sat = {node_name: idx + 1 for idx, node_name in enumerate(node_dict.keys())}

        # Construct Weighted Conjunctive Normal Form (WCNF)
        wcnf = WCNF()

        # Add belief clauses
        for name, weight in belief.items():
            clause = [graph2sat[name]] if weight > 0 else [-graph2sat[name]]
            wcnf.append(clause, weight=abs(weight))

        # Add edge consistency clauses
        for (name1, name2), weight in consistency.items():
            clause1 = [-graph2sat[name1], graph2sat[name2]] if weight > 0 else [-graph2sat[name1], -graph2sat[name2]]
            wcnf.append(clause1, weight=1)

        return graph2sat, wcnf

    @staticmethod
    def find_unsatisfied_clause(wcnf: WCNF, interpretation: List):
        unsatisfied_clauses = []
        for clause, weight in zip(wcnf.soft, wcnf.wght):
            satisfied_atoms = [atom in interpretation for atom in clause]

            if True not in satisfied_atoms:  # unsatisfied clause
                unsatisfied_clauses.append((clause, weight))

        return unsatisfied_clauses
