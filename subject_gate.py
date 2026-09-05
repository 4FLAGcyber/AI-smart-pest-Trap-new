"""Safety-first subject validation for the generic ImageNet classifier."""

from dataclasses import dataclass

import config

INSECT_CLASS_IDS = range(300, 327)
ANIMAL_CLASS_IDS = range(0, 398)
PRIMATE_CLASS_IDS = range(365, 385)
_HUMAN_LABEL_TERMS = (
    "ballplayer", "groom", "bridegroom", "scuba diver", "academic gown",
    "maillot", "bikini", "jersey", "sweatshirt", "suit", "lab coat",
    "military uniform", "miniskirt", "neck brace", "sunglasses",
)


@dataclass(frozen=True)
class SubjectVerdict:
    subject: str
    accepted: bool
    reason: str

    def detail(self) -> dict:
        return {
            "error": "unsupported_subject",
            "subject": self.subject,
            "reason": self.reason,
            "message_key": f"subject_{self.subject}",
        }


def _is_human_proxy(label: str) -> bool:
    text = (label or "").lower()
    return any(term in text for term in _HUMAN_LABEL_TERMS)


def evaluate_subject(prediction) -> SubjectVerdict:
    """Accept only sufficiently supported insect or non-human animal results."""
    candidates = list(getattr(prediction, "subject_top", ()) or ())
    if not candidates:
        return SubjectVerdict("unknown", False, "missing_subject_evidence")

    top_score = candidates[0][2]
    if top_score < config.SUBJECT_MIN_TOP_CONFIDENCE:
        return SubjectVerdict("unknown", False, "low_confidence")

    insect_mass = sum(score for index, _, score in candidates if index in INSECT_CLASS_IDS)
    human_mass = sum(score for _, label, score in candidates if _is_human_proxy(label))
    animal_mass = sum(
        score
        for index, _, score in candidates
        if index in ANIMAL_CLASS_IDS and index not in PRIMATE_CLASS_IDS
    )
    object_mass = sum(
        score
        for index, label, score in candidates
        if index not in ANIMAL_CLASS_IDS and not _is_human_proxy(label)
    )

    if human_mass >= config.SUBJECT_HUMAN_EVIDENCE:
        return SubjectVerdict("human", False, "human_evidence")
    if insect_mass >= config.SUBJECT_MIN_INSECT_MASS:
        return SubjectVerdict("insect", True, "insect_evidence")
    if animal_mass >= config.SUBJECT_MIN_ANIMAL_MASS:
        return SubjectVerdict("animal", True, "animal_evidence")
    if object_mass >= config.SUBJECT_OBJECT_EVIDENCE:
        return SubjectVerdict("object", False, "object_evidence")
    return SubjectVerdict("unknown", False, "ambiguous_subject")
