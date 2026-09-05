import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from classifier import Prediction
from main import decide_and_act
from pest_categories import CATEGORY_HARMFUL
from subject_gate import evaluate_subject


def prediction(candidates, category=CATEGORY_HARMFUL):
    index, label, score = candidates[0]
    return Prediction(
        label=label,
        confidence=score,
        category=category,
        top=[(label, score)],
        subject_top=candidates,
    )


class FakeTrap:
    def __init__(self):
        self.activations = 0

    def activate(self):
        self.activations += 1


class SubjectGateTests(unittest.TestCase):
    def test_accepts_insect_evidence(self):
        verdict = evaluate_subject(prediction([
            (301, "ladybug", 0.52),
            (302, "ground beetle", 0.08),
            (399, "abacus", 0.04),
        ]))
        self.assertTrue(verdict.accepted)
        self.assertEqual(verdict.subject, "insect")

    def test_accepts_non_human_animal_evidence(self):
        verdict = evaluate_subject(prediction([
            (207, "golden retriever", 0.34),
            (208, "Labrador retriever", 0.23),
            (399, "abacus", 0.03),
        ]))
        self.assertTrue(verdict.accepted)
        self.assertEqual(verdict.subject, "animal")

    def test_rejects_human_proxy(self):
        verdict = evaluate_subject(prediction([
            (982, "ballplayer, baseball player", 0.68),
            (399, "abacus", 0.04),
        ]))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.subject, "human")

    def test_rejects_object_evidence(self):
        verdict = evaluate_subject(prediction([
            (399, "abacus", 0.68),
            (400, "abaya", 0.04),
        ]))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.subject, "object")

    def test_rejects_ambiguous_or_low_confidence_evidence(self):
        verdict = evaluate_subject(prediction([
            (399, "abacus", 0.12),
            (400, "abaya", 0.10),
        ]))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.subject, "unknown")

    def test_rejected_prediction_cannot_activate_trap(self):
        trap = FakeTrap()
        acted = decide_and_act(prediction([
            (982, "ballplayer, baseball player", 0.72),
        ]), trap)
        self.assertFalse(acted)
        self.assertEqual(trap.activations, 0)

    def test_accepted_harmful_insect_can_activate_trap(self):
        trap = FakeTrap()
        acted = decide_and_act(prediction([
            (307, "weevil", 0.72),
        ]), trap)
        self.assertTrue(acted)
        self.assertEqual(trap.activations, 1)


class ManualAnalysisSafetyTests(unittest.TestCase):
    def test_rejected_frame_has_no_snapshot_or_database_event(self):
        import dashboard

        rejected = prediction([(399, "abacus", 0.73)])
        classifier = type("Classifier", (), {"classify": lambda _, __: rejected})()
        with patch.object(dashboard, "get_classifier", return_value=classifier), \
             patch.object(dashboard.cloud_logger, "save_snapshot") as save_snapshot, \
             patch.object(dashboard, "store_detection") as store_detection:
            with self.assertRaises(HTTPException) as error:
                dashboard._analyse_frame(np.zeros((8, 8, 3), dtype=np.uint8), "upload")

        self.assertEqual(error.exception.status_code, 422)
        self.assertEqual(error.exception.detail["subject"], "object")
        save_snapshot.assert_not_called()
        store_detection.assert_not_called()

    def test_accepted_frame_creates_one_analysis_only_event(self):
        import dashboard

        accepted = prediction([(307, "weevil", 0.73)])
        classifier = type("Classifier", (), {"classify": lambda _, __: accepted})()
        with patch.object(dashboard, "get_classifier", return_value=classifier), \
             patch.object(dashboard.cloud_logger, "save_snapshot", return_value="accepted.jpg") as save_snapshot, \
             patch.object(dashboard, "store_detection") as store_detection:
            result = dashboard._analyse_frame(np.zeros((8, 8, 3), dtype=np.uint8), "capture")

        self.assertEqual(result["subject"], "insect")
        self.assertFalse(result["action_taken"] if "action_taken" in result else False)
        save_snapshot.assert_called_once()
        store_detection.assert_called_once()
        self.assertFalse(store_detection.call_args.args[0]["action_taken"])


if __name__ == "__main__":
    unittest.main()
