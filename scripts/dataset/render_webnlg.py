#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load the GEM/web_nlg dataset from Hugging Face and visualize RDF graphs using pydot.
Only saves graph images with a progress bar.
"""

from datasets import load_dataset
import pydot
import os
from tqdm import tqdm  # progress bar

def render_graph(triples, save_path=None):
    graph = pydot.Dot(graph_type="digraph", rankdir="LR")

    for triple in triples:
        subj, pred, obj = [s.strip() for s in triple.split("|")]

        graph.add_node(pydot.Node(subj, shape="box", style="filled", fillcolor="lightblue", fontsize="10"))
        graph.add_node(pydot.Node(obj, shape="box", style="filled", fillcolor="lightblue", fontsize="10"))

        graph.add_edge(pydot.Edge(subj, obj, label=pred, fontsize="8", fontcolor="red"))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        graph.write_png(save_path)

def main():
    dataset = load_dataset("GEM/web_nlg", "en", split="validation")
    print(f"Loaded {len(dataset)} training examples.")

    # # Shuffle and maybe limit if you don’t want all (remove .select to save all)
    # sampled = dataset.shuffle(seed=42)

    for idx in tqdm(range(len(dataset)), desc="Rendering graphs"):
        example = dataset[idx]
        save_file = f"webnlg/graphs_val/webnlg_pydot_{idx}.png"
        render_graph(example['input'], save_path=save_file)

    # render_graph(dataset[0]['input'], save_path="webnlg/graphs/webnlg_pydot_example.png")

    

if __name__ == "__main__":
    main()
