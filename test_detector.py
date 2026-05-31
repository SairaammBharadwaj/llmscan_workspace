import torch
import numpy as np

from benchmark_suite import (
    load_detector,
    load_model_and_tokenizer,
    predict_prompt
)

detector=load_detector()

llm,tokenizer=load_model_and_tokenizer()

print("\nLLMScan Interactive Test")
print("Type 'exit' to quit.\n")

while True:

    prompt=input("Prompt: ")

    if prompt.lower()=="exit":
        break

    prob,pred=predict_prompt(
        prompt,
        detector,
        llm,
        tokenizer
    )

    print(f"\nRisk Score: {prob:.4f}")

    if pred==1:
        print("Prediction: UNSAFE")
    else:
        print("Prediction: SAFE")

    print("-"*50)