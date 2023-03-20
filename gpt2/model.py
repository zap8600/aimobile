import torch
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("./model")
tokenizer.pad_token = tokenizer.eos_token
model = torch.jit.load("./model/traced_gpt2.pt")

sequence = (input("Enter Prompt: "))

inputs = tokenizer(sequence, padding='max_length', max_length=128)['input_ids']
outputs = model(torch.tensor([inputs]))

text = tokenizer.decode(outputs, skip_special_tokens=True)
print(text)