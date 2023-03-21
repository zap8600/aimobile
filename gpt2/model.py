import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

sequence = ([15006, 36934, 37516, 15006])

print(len(sequence))
print(sequence)

inputs = torch.tensor(sequence)
outputs = model.generate(inputs)

tokenizer.decode(outputs)