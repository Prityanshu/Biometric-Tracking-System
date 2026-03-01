import numpy as np


class FusionEngine:
    def __init__(self):
        # Weights derived from cross-similarity analysis:
        #
        #   face  cross-sim ≈ 0.15  → highly discriminative, dominant weight
        #   body  cross-sim ≈ 0.83  → low discriminability, small weight
        #   gait  cross-sim ≈ 0.94  → average silhouette not discriminative
        #                              enough for these subjects, near-zero weight
        #
        # Weight = proportional to (1 - cross_sim)²  (squared to penalize
        # high cross-sim modalities more aggressively)
        #   face:  (1-0.15)² = 0.72
        #   body:  (1-0.83)² = 0.03
        #   gait:  (1-0.94)² = 0.004
        # Normalized:
        #   face  ≈ 0.95
        #   body  ≈ 0.04
        #   gait  ≈ 0.01
        self.default_weights = {
            "face": 0.95,
            "body": 0.04,
            "gait": 0.01
        }

    def compute_final_score(
        self,
        face_score=None,
        body_score=None,
        gait_score=None,
        attr_score=None,
        verbose=False
    ):
        scores  = []
        weights = []
        labels  = []

        if face_score is not None:
            scores.append(face_score)
            weights.append(self.default_weights["face"])
            labels.append(f"face={face_score:.3f}")

        if body_score is not None:
            scores.append(body_score)
            weights.append(self.default_weights["body"])
            labels.append(f"body={body_score:.3f}")

        if gait_score is not None:
            scores.append(gait_score)
            weights.append(self.default_weights["gait"])
            labels.append(f"gait={gait_score:.3f}")

        if len(scores) == 0:
            return 0.0, False

        scores  = np.array(scores)
        weights = np.array(weights)

        # Normalize so available modalities always sum to 1
        weights = weights / np.sum(weights)

        final_score = float(np.sum(scores * weights))

        # Only trust the result if face was available
        # Body + gait alone cannot distinguish these subjects reliably
        trusted = face_score is not None

        if verbose:
            trust_str = "trusted" if trusted else "UNTRUSTED - no face"
            print(f"  [Fusion] {' | '.join(labels)} → final={final_score:.3f} ({trust_str})")

        return final_score, trusted