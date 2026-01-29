#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

# =========================
# Target processes (normalized to lowercase)
# =========================
# TARGET_PROCS = {
#     "nissrv.exe",
#     "msmpeng.exe",
#     "securityhealthservice.exe",
#     "securityhealthsystray.exe",
# }

TARGET_PROCS = {
    "csfalconservice.exe",
    "cssystemtray_7.15.18514.0.exe",
    "csfalconcontainer.exe",
}

# TARGET_PROCS = {
#     "avp.exe",
#     "avpui.exe"
# }

# Special tokens
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
PAD = "<PAD>"


PROC_PREFIX = "<PROC:"
PROC_SUFFIX = ">"

@dataclass
class Event:
    ts: float
    proc: str    
    method: str


def normalize_proc_name(process_name: Optional[str]) -> str:
    """
    Input:
        \\Device\\HarddiskVolume2\\Windows\\System32\\SecurityHealthService.exe
    Output:
        securityhealthservice.exe
    """
    if not process_name:
        return "unknown.exe"
    # Support both '\' and '/'
    name = process_name.replace("\\", "/").split("/")[-1]
    return name.strip().lower()


def normalize_method(method: Optional[str]) -> str:
    """
    Normalize Method token with minimal cleaning.
    """
    if not method:
        return "UnknownMethod"
    m = method.strip()
    m = re.sub(r"\s+", "", m)
    return m


def make_proc_token(proc: str) -> str:
    return f"{PROC_PREFIX}{proc}{PROC_SUFFIX}"


def read_events(jsonl_path: str) -> List[Event]:
    """
    Read JSONL file, filter by target processes,
    and extract (TimeStamp, ProcessName, Method).
    """
    events: List[Event] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts = float(obj.get("TimeStamp"))
                proc = normalize_proc_name(obj.get("ProcessName"))
                if proc not in TARGET_PROCS:
                    continue
                method = normalize_method(obj.get("Method"))
                events.append(Event(ts=ts, proc=proc, method=method))
            except Exception:
                # Skip malformed lines silently (print line_no for debugging if needed)
                continue


    events.sort(key=lambda e: e.ts)
    return events


def build_token_sequence(events: List[Event]) -> List[str]:
    """
    Strategy 2: <PROC_*> + Method sequence
    - Start with BOS, end with EOS
    - Insert <PROC:proc.exe> whenever the process changes
    """
    tokens: List[str] = [BOS]
    last_proc: Optional[str] = None

    for ev in events:
        if ev.proc != last_proc:
            tokens.append(make_proc_token(ev.proc))
            last_proc = ev.proc
        tokens.append(ev.method)

    tokens.append(EOS)
    return tokens


def build_vocab(token_seqs: List[List[str]]) -> Dict[str, int]:
    """
    Build vocabulary: token -> id
    """
    vocab: Dict[str, int] = {PAD: 0, UNK: 1}
    for seq in token_seqs:
        for t in seq:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def main():

    input_path = "../data/baseline/faclon_edr_only.jsonl"
    # input_path = "../data/baseline/kaspersky_edr_only.jsonl"

    out_tokens_path = "../data/baseline/tokens.jsonl"
    out_vocab_path = "../data/baseline/vocab.json"


    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_tokens_path = sys.argv[2]
    if len(sys.argv) >= 4:
        out_vocab_path = sys.argv[3]

    print(f"[INFO] Input file:  {input_path}")
    print(f"[INFO] Output files: {out_tokens_path}, {out_vocab_path}")
    print(f"[INFO] Target processes: {', '.join(TARGET_PROCS)}")
    print("-" * 60)

    events = read_events(input_path)
    if not events:
        print("No events found for target processes. Check input file or TARGET_PROCS.")
        sys.exit(0)


    token_seq = build_token_sequence(events)
    token_seqs = [token_seq]

    vocab = build_vocab(token_seqs)

    with open(out_tokens_path, "w", encoding="utf-8") as f:
        for seq in token_seqs:
            f.write(json.dumps(seq, ensure_ascii=False) + "\n")


    with open(out_vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"[OK] events={len(events)} tokens={len(token_seq)} vocab_size={len(vocab)}")
    print(f"[OK] Written files: {out_tokens_path}, {out_vocab_path}")
    print("[SAMPLE TOKENS]", token_seq[:40])


if __name__ == "__main__":
    main()
