import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model_and_tokenizer():
    # You can change this to your local path if the model is already downloaded
    model_id = "mistralai/Mistral-7B-v0.1" 
    
    # Configuration for aggressive VRAM saving
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        # 🧠 THE FIX: Force the model to return the attention weights
        output_attentions=True 
    )
    
    return model, tokenizer