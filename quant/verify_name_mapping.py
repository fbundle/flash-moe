"""Verify every tensor in the HF safetensors index has a name_mapping entry,
and every pattern in the mapping covers at least one real tensor."""

import json

INDEX = "quant/model.safetensors.index.json"
MAPPING = "quant/name_mapping.json"

def mapping_keys(MAPPING: str) -> set[str]:
    with open(MAPPING) as f:
        mapping = json.load(f)

    NUM_LAYERS = 40        # layers 0..39
    NUM_VISION_BLOCKS = 27  # blocks 0..26

    names = set()
    for hf_pat in mapping.keys():
        if "{L}" in hf_pat and "{B}" in hf_pat:
            assert False
        elif "{L}" in hf_pat:
            for l in range(NUM_LAYERS):
                names.add(hf_pat.format(L=l))
        elif "{B}" in hf_pat:
            for b in range(NUM_VISION_BLOCKS):
                names.add(hf_pat.format(B=b))
        else:
            names.add(hf_pat)

    return names

def hf_keys(INDEX: str) -> set[str]:
    with open(INDEX) as f:
        hf = json.load(f)
    hf_set = set(hf["weight_map"].keys())
    return hf_set

HF_SET = hf_keys(INDEX)
MAPPING_KEYS = mapping_keys(MAPPING)

assert len(HF_SET - MAPPING_KEYS) == 0