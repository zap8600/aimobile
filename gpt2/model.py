import torch
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("./model")
model = torch.jit.load("./model/traced_gpt2.pt")

