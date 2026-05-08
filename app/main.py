# app/main.py

import streamlit as st
import sys
import os
import torch
import pandas as pd
import joblib

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from scanner.token_scanner import scan_tokens
from scanner.layer_scanner import scan_layers
from scanner.feature_extractor import extract_token_features
from scanner.model_loader import load_model_and_tokenizer
from train_final import MisbehaviorMLP

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

@st.cache_resource
def load_llm():

    llm, tokenizer = load_model_and_tokenizer()

    return llm, tokenizer

@st.cache_resource
def load_detector():

    model = MisbehaviorMLP()

    model_path = "detector/misbehavior_detector.pth"

    scaler_path = "detector/scaler.pkl"

    if (
        os.path.exists(model_path)
        and os.path.exists(scaler_path)
    ):

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device
            )
        )

        model.to(device)

        model.eval()

        scaler = joblib.load(
            scaler_path
        )

        return model, scaler

    return None, None

st.set_page_config(
    page_title="LLMScan Detector",
    layout="wide"
)

st.title(
    "🛡️ LLMScan: Neural Misbehavior Detection"
)

st.markdown(
    """
    This system analyzes:

    - Token-level causal influence
    - Layer-level behavioral perturbations
    - Internal neural activation signatures

    instead of relying only on prompt keywords.
    """
)

prompt = st.text_area(
    "Enter prompt to scan:",
    placeholder="How to..."
)

if st.button("Run Causal Scan"):

    if prompt.strip():

        with st.spinner(
            "Analyzing Neural Activations..."
        ):

            llm, tokenizer = load_llm()

            token_dist = scan_tokens(
                prompt,
                llm,
                tokenizer,
                device
            )

            layer_dist = scan_layers(
                prompt,
                llm,
                tokenizer,
                device
            )

            token_feat = extract_token_features(
                token_dist
            )

            causal_map = (
                token_feat +
                layer_dist
            )

            detector, scaler = load_detector()

            if detector is not None:

                scaled_features = scaler.transform(
                    [causal_map]
                )

                feat_tensor = torch.tensor(
                    scaled_features,
                    dtype=torch.float32
                ).to(device)

                with torch.no_grad():

                    prob = detector(
                        feat_tensor
                    ).item()

                st.subheader(
                    "Detection Result"
                )

                if prob > 0.5:

                    st.error(
                        f"🚨 MISBEHAVIOR DETECTED "
                        f"({prob*100:.2f}%)"
                    )

                else:

                    st.success(
                        f"✅ SAFE PROMPT "
                        f"({(1-prob)*100:.2f}%)"
                    )

                st.divider()

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader(
                        "Layer Causal Distribution"
                    )

                    layer_df = pd.DataFrame({
                        "Layer": list(
                            range(len(layer_dist))
                        ),
                        "Causal Effect": layer_dist
                    })

                    st.line_chart(
                        layer_df,
                        x="Layer",
                        y="Causal Effect"
                    )

                with col2:

                    st.subheader(
                        "Token Influence"
                    )

                    token_df = pd.DataFrame({
                        "Token Index": list(
                            range(len(token_dist))
                        ),
                        "Effect": token_dist
                    })

                    st.bar_chart(
                        token_df,
                        x="Token Index",
                        y="Effect"
                    )

                st.divider()

                st.subheader(
                    "Raw Feature Statistics"
                )

                stats_df = pd.DataFrame({
                    "Metric": [
                        "Token Mean",
                        "Token Max",
                        "Layer Mean",
                        "Layer Max"
                    ],
                    "Value": [
                        float(
                            sum(token_dist)
                            / len(token_dist)
                        ),
                        float(max(token_dist)),
                        float(
                            sum(layer_dist)
                            / len(layer_dist)
                        ),
                        float(max(layer_dist))
                    ]
                })

                st.dataframe(
                    stats_df,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Detector model or scaler "
                    "not found."
                )

    else:

        st.warning(
            "Please enter a prompt."
        )