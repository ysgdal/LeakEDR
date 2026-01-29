#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import random
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def load_tokens_jsonl(path: str):
    """
    Load token sequences from a JSONL file.
    Each line is expected to be a JSON list of tokens.
    """
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq = json.loads(line)  # list[str]
            seqs.append(seq)
    return seqs

def load_vocab(path: str):
    """
    Load vocabulary from JSON file. Returns token -> id mapping.
    """
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab

def ensure_special_tokens_in_vocab(vocab: dict):
    """
    Ensure that all required special tokens are present in the vocabulary.
    """
    for t in SPECIAL_TOKENS:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab

def save_corpus(seqs, out_train: str, out_valid: str, valid_ratio: float = 0.05, seed: int = 42):
    """
    Shuffle and split sequences into training and validation sets, then save to files.
    """
    random.seed(seed)
    random.shuffle(seqs)
    n_valid = max(1, int(len(seqs) * valid_ratio)) if len(seqs) > 1 else 0

    valid = seqs[:n_valid]
    train = seqs[n_valid:] if n_valid > 0 else seqs

    def write_txt(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for seq in data:
                f.write(" ".join(seq) + "\n")

    write_txt(out_train, train)
    if n_valid > 0:
        write_txt(out_valid, valid)
    else:
        write_txt(out_valid, train)

def build_wordlevel_tokenizer(vocab: dict, tokenizer_json_path: str):
    """
    Build and save a WordLevel tokenizer using provided vocabulary.
    """
    model = WordLevel(vocab=vocab, unk_token="<UNK>")
    tok = Tokenizer(model)
    tok.pre_tokenizer = Whitespace()

    tok.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[("<BOS>", vocab["<BOS>"]), ("<EOS>", vocab["<EOS>"])]
    )

    tok.save(tokenizer_json_path)



def main():
    # Paths
    tokens_path = "../data/baseline/tokens.jsonl"
    vocab_path  = "../data/baseline/vocab.json"


    out_dir = Path("lm_data")
    out_dir.mkdir(exist_ok=True)


    seqs = load_tokens_jsonl(tokens_path)
    vocab = ensure_special_tokens_in_vocab(load_vocab(vocab_path))


    train_txt = str(out_dir / "train.txt")
    valid_txt = str(out_dir / "valid.txt")
    save_corpus(seqs, train_txt, valid_txt, valid_ratio=0.05)


    tokenizer_json = str(out_dir / "tokenizer.json")
    build_wordlevel_tokenizer(vocab, tokenizer_json)


    tokenizer_cfg = {
        "model_max_length": 1024,
        "unk_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>"
    }
    with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_cfg, f, ensure_ascii=False, indent=2)

    # Summary
    print("[OK] Written files:")
    print(" -", train_txt)
    print(" -", valid_txt)
    print(" -", tokenizer_json)
    print(" -", str(out_dir / "tokenizer_config.json"))
    print("[INFO] Number of sequences =", len(seqs), "Vocabulary size =", len(vocab))

if __name__ == "__main__":
    main()
