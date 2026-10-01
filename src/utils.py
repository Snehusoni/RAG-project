import json
from typing import List, Dict, Any


def format_source_badge(source: str, page: Any, confidence: float) -> str:
    """Format a clean markdown badge for a source citation."""
    conf_pct = int(confidence * 100)
    color = "green" if conf_pct >= 70 else "orange" if conf_pct >= 50 else "red"
    return f"📄 **{source}** (Page {page}) • Relevance: `{conf_pct}%`"


def export_chat_history_markdown(messages: List[Dict[str, Any]]) -> str:
    """Export the chat dialogue to formatted Markdown."""
    lines = ["# DocuMind RAG — Chat Session Transcript\n"]
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"### {role}\n{content}\n")
        if "sources" in msg and msg["sources"]:
            lines.append("**Sources Cited:**")
            for s in msg["sources"]:
                lines.append(f"- {s['source']} (Page {s.get('page', 1)}) [Score: {int(s.get('confidence_score', 0)*100)}%]")
            lines.append("\n")
    return "\n".join(lines)


def get_sample_document_text() -> str:
    """Return a rich sample document on Transformer Architecture for immediate testing."""
    return """# Title: Attention Is All You Need — Executive Summary & Architecture Overview
Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
Published: 2017 (Google Brain & Google Research)

## Abstract & Introduction
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.

## Architecture: Encoder and Decoder
The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.
1. Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism, and a simple, position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)).
2. Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking ensures that predictions for position i can depend only on the known outputs at positions less than i.

## Attention Mechanism
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.
The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
We call our particular attention 'Scaled Dot-Product Attention'. The input consists of queries and keys of dimension d_k, and values of dimension d_v. We compute the dot products of the query with all keys, divide each by sqrt(d_k), and apply a softmax function to obtain the weights on the values.
Equation: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V.

## Why Multi-Head Attention?
Instead of performing a single attention function with d_model-dimensional keys, queries and values, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to d_k, d_k and d_v dimensions, respectively.
In our work we employ h = 8 parallel attention layers, or heads. For each of these we use d_k = d_v = d_model / h = 64.

## Positional Encoding
Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add 'positional encodings' to the input embeddings at the bottoms of the encoder and decoder stacks. We use sine and cosine functions of different frequencies.

## Computational Complexity & Results
On the WMT 2014 English-to-German translation task, the big transformer model achieves a BLEU score of 28.4, outperforming the existing best results by over 2.0 BLEU. On the English-to-French translation task, our model establishes a state-of-the-art BLEU score of 41.8 after training for 3.5 days on 8 P100 GPUs.
"""
