"""
STAND-IN classification map.

Bridges a generic pretrained image classifier (ImageNet MobileNetV2) to the
three categories the trap actually cares about: Harmful, Beneficial, Harmless.

Matching order matters: specific harmful keywords are checked first (e.g.
"leaf beetle"), then beneficial keywords (e.g. "ladybug"), so a label like
"ladybug, lady beetle, ladybird" is correctly Beneficial even though it
contains the word "beetle".

This is a placeholder for demo purposes. Real pest identification should use
a model trained on your own labelled insect photos — then you can output the
category directly and delete this mapping entirely.
"""

CATEGORY_HARMFUL = "Harmful"
CATEGORY_BENEFICIAL = "Beneficial"
CATEGORY_HARMLESS = "Harmless"

# Specific pest labels -> Harmful. Checked first.
_HARMFUL_KEYWORDS = [
    "weevil", "leaf beetle", "chrysomelid", "long-horned beetle",
    "rhinoceros beetle", "cabbage butterfly", "borer", "aphid",
    "locust", "grasshopper", "hopper", "cricket", "cockroach", "roach",
    "cicada", "leafhopper", "termite", "armyworm", "cutworm", "caterpillar",
]

# Pollinators / predators -> Beneficial. Never trapped.
_BENEFICIAL_KEYWORDS = [
    "ladybug", "ladybird", "lady beetle", "bee", "mantis", "mantid",
    "dragonfly", "damselfly", "lacewing", "monarch", "butterfly", "admiral",
    "ground beetle", "dung beetle", "hover fly", "hoverfly", "wasp",
]

# Everything else falls back to "Harmless" (no action taken).

import re


def _matches(text: str, keyword: str) -> bool:
    # Word-boundary match so e.g. "bee" never matches inside "beetle".
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def label_to_category(label: str) -> str:
    """Map a raw classifier label to Harmful / Beneficial / Harmless."""
    text = label.lower()

    for kw in _HARMFUL_KEYWORDS:
        if _matches(text, kw):
            return CATEGORY_HARMFUL

    for kw in _BENEFICIAL_KEYWORDS:
        if _matches(text, kw):
            return CATEGORY_BENEFICIAL

    return CATEGORY_HARMLESS


# Broad insect / arthropod terms used to decide whether a captured image is
# actually relevant to pest content. Word-boundary matched so e.g. "ant" never
# matches inside "mantis" or "plant".
_INSECT_KEYWORDS = [
    "beetle", "weevil", "bee", "wasp", "hornet", "ant", "fly", "mosquito",
    "grasshopper", "locust", "cricket", "cockroach", "roach", "mantis",
    "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "butterfly",
    "monarch", "moth", "caterpillar", "aphid", "spider", "scorpion",
    "centipede", "millipede", "tick", "mite", "insect", "bug", "ladybug",
    "ladybird", "hopper",
]


def is_insect_label(label: str) -> bool:
    """True if the raw classifier label refers to an insect/arthropod."""
    text = (label or "").lower()
    return any(_matches(text, kw) for kw in _INSECT_KEYWORDS)
