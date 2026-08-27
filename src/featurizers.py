"""SELFIES featurization: SMILES -> SELFIES -> token ids.

Vocab is built from the training split only (standard practice — the model
shouldn't have privileged knowledge of valid/test-only tokens); unseen
tokens at eval time map to <UNK>.
"""
from __future__ import annotations

PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]


def smiles_to_selfies(smiles: str) -> str | None:
    import selfies as sf

    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def selfies_to_tokens(selfies_str: str) -> list[str]:
    import selfies as sf

    return list(sf.split_selfies(selfies_str))


def build_vocab(selfies_list: list[str]) -> dict[str, int]:
    token_set = set()
    for s in selfies_list:
        token_set.update(selfies_to_tokens(s))
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for tok in sorted(token_set):  # sorted for reproducibility across runs
        vocab[tok] = len(vocab)
    return vocab


def encode(selfies_str: str, vocab: dict[str, int], max_len: int) -> tuple[list[int], int]:
    """Returns (token_ids padded/truncated to max_len, true_length_including_BOS_EOS)."""
    tokens = selfies_to_tokens(selfies_str)
    ids = [vocab[BOS]] + [vocab.get(t, vocab[UNK]) for t in tokens] + [vocab[EOS]]
    true_len = min(len(ids), max_len)
    if len(ids) < max_len:
        ids = ids + [vocab[PAD]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = vocab[EOS]  # keep EOS even when truncated
    return ids, true_len
