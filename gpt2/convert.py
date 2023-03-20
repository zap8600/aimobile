import torch
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

enc = GPT2Tokenizer.from_pretrained('gpt2')

text = "I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy."
tokenized_text = enc.tokenize(text)

indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)

tokens_tensor = torch.tensor([indexed_tokens])

config = GPT2Config(torchscript=True)

model = GPT2LMHeadModel(config)
model.eval()

traced_model = torch.jit.trace(model, [tokens_tensor])

tokenizer.save_pretrained("./model")
torch.jit.save(traced_model, "./model/traced_gpt2.pt")

print("Done!")