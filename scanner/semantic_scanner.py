from sentence_transformers import (
    SentenceTransformer
)

import numpy as np

semantic_model=SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def extract_semantic_features(
    prompt
):

    embedding=semantic_model.encode(
        prompt
    )

    embedding=np.array(
        embedding,
        dtype=np.float32
    )

    embedding=(
        embedding /
        (
            np.linalg.norm(
                embedding
            ) + 1e-8
        )
    )

    return embedding.tolist()