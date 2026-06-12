"""Reference-frame retrieval for online visual localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gps_ex1.preprocess.reference_index import ReferenceIndex, compute_global_descriptor


@dataclass(frozen=True)
class RetrievalCandidate:
    index: int
    distance: float
    similarity: float
    query_scale: float = 1.0


def retrieve_candidates_from_descriptor(
    query_descriptor: np.ndarray,
    reference_index: ReferenceIndex,
    top_k: int = 10,
    query_scale: float = 1.0,
) -> list[RetrievalCandidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    q = query_descriptor.astype(np.float32)
    q = q / (float(np.linalg.norm(q)) + 1e-6)
    refs = reference_index.descriptors.astype(np.float32)
    refs = refs / (np.linalg.norm(refs, axis=1, keepdims=True) + 1e-6)

    similarities = refs @ q
    distances = 1.0 - similarities
    k = min(top_k, len(distances))
    candidate_indices = np.argpartition(distances, k - 1)[:k]
    candidate_indices = candidate_indices[np.argsort(distances[candidate_indices])]

    return [
        RetrievalCandidate(
            index=int(i),
            distance=float(distances[i]),
            similarity=float(similarities[i]),
            query_scale=float(query_scale),
        )
        for i in candidate_indices
    ]


def retrieve_candidates(image_path: str | Path, reference_index: ReferenceIndex, top_k: int = 10) -> list[RetrievalCandidate]:
    descriptor = compute_global_descriptor(image_path)
    return retrieve_candidates_from_descriptor(descriptor, reference_index, top_k=top_k)
