#!/usr/bin/env python3
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

def embed(model: SentenceTransformer, text: str, role: str, normalize: bool = True) -> np.ndarray:
    # E5-style prefixes: query:/passage: for best performance
    # (For pure "text-text similarity", you can use passage/passsage)
    text = (text or "").strip()
    if role:
        text = f"{role}: {text}"
    vec = model.encode([text], normalize_embeddings=normalize, convert_to_numpy=True)[0]
    return vec.astype(np.float32)

def compare_vectors(a: np.ndarray, b: np.ndarray, normalized: bool) -> dict:
    dot = float(np.dot(a, b))

    if normalized:
        cosine = dot  # because ||a||=||b||=1
    else:
        cosine = float(dot / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

    # Distances
    l2 = float(np.linalg.norm(a - b))                 # Euclidean
    l1 = float(np.sum(np.abs(a - b)))                 # Manhattan

    # Angular distance in [0, 1] where 0 = identical direction
    c = float(np.clip(cosine, -1.0, 1.0))
    angular = float(np.arccos(c) / np.pi)

    # Optional: convert distances to "similarities" (monotonic transforms)
    sim_l2 = float(1.0 / (1.0 + l2))
    sim_l1 = float(1.0 / (1.0 + l1))

    return {
        "cosine": cosine,
        "dot": dot,
        "l2_distance": l2,
        "l1_distance": l1,
        "angular_distance_0_to_1": angular,
        "l2_similarity_1_over_1_plus_d": sim_l2,
        "l1_similarity_1_over_1_plus_d": sim_l1,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_a")
    ap.add_argument("text_b")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--role_a", default="passage", choices=["query", "passage", ""])
    ap.add_argument("--role_b", default="passage", choices=["query", "passage", ""])
    ap.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu (default: auto)")
    ap.add_argument("--no_normalize", action="store_true")
    args = ap.parse_args()

    normalize = not args.no_normalize

    model = SentenceTransformer(args.model, device=args.device)  # auto device if None

    a = embed(model, args.text_a, args.role_a, normalize=normalize)
    b = embed(model, args.text_b, args.role_b, normalize=normalize)

    out = compare_vectors(a, b, normalized=normalize)

    # Pretty print
    for k, v in out.items():
        print(f"{k:28s} = {v:.6f}")

if __name__ == "__main__":
    main()
