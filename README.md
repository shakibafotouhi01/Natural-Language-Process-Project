
# Narrative Graph Analysis for Cross-Cultural Storytelling

This project explores how Large Language Models (LLMs) understand and reproduce narratives across different cultures. It focuses on comparing human-written stories with LLM-generated retellings using graph-based and semantic analysis.

## Overview

Narratives are transformed into event graphs, where:

Nodes represent events (subject–verb–object triples)
Edges represent temporal relationships

These graphs allow structural comparison between original and generated stories.

## Methodology
Event Extraction: Convert stories into structured event representations
Graph Construction: Build directed narrative graphs
Similarity Analysis:
Graph metrics (nodes, edges, density, path length)
Graph edit distance
Embedding-based semantic similarity (Sentence-BERT + cosine similarity)
## Key Findings
* LLM retellings show a compression effect, preserving ~88.5% of structure
* Core narrative flow is mostly maintained, but some events are omitted
* Semantic similarity remains relatively high despite structural reduction
## Repository Structure
data/ – Original and generated narratives
src/ – Core modules (event extraction, graph building, evaluation)
notebooks/ – Experiments and visualizations
outputs/ – Built human and LLm Graphs, Tables, plots, and analysis outputs
## Goal

To evaluate whether LLMs can act as cultural narrative translators, preserving both structure and meaning across storytelling traditions.

## Context

This project was developed as part of a Natural Language Processing course
