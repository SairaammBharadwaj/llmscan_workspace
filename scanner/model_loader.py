import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "mistralai/Mistral-7B-v0.1"


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        output_attentions=True,
        output_hidden_states=True,
    )
    model.eval()
    return model, tokenizer
