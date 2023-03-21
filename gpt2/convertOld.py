import json
import torch
import numpy as np
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

with open("./model/vocab.json", "r") as f:
    vocabulary = json.load(f)

token_to_index = {}
for i, token in enumerate(vocabulary):
    token_to_index[token] = i

# temp = 0.8

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# text = "I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy."
text = "A compound sentence is"
tokenized_text = tokenizer(text, return_tensors='pt')['input_ids']

print(tokenized_text)

config = GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12, n_inner=None, torchscript=True,)

model = GPT2LMHeadModel(config)
model.eval()

traced_model = torch.jit.trace(model, [tokenized_text])

tokenizer.save_pretrained("./model")
torch.jit.save(traced_model, "./model/traced_gpt2.pt")

loaded_model = torch.jit.load("./model/traced_gpt2.pt")
loaded_model.eval()

text = "The dog jumps over"
tokenized_text = tokenizer(text, return_tensors='pt')['input_ids']

output = loaded_model(tokenized_text)

output_indices = torch.argmax(output[0], dim=-1)

decoded_sequence = []
for i in range(output_indices.shape[1]):
    index = output_indices[0, i].item()
    token = list(vocabulary.keys())[index]
    decoded_sequence.append(token)

final_string = tokenizer.convert_tokens_to_string(decoded_sequence)
print(final_string)

"""
output_indices = torch.argmax(output[0], dim=-1)
words = [list(vocabulary.keys())[list(vocabulary.values()).index(idx.item())] for idx in output_indices.flatten()]
sentence = " ".join(words)

print(sentence)
"""

"""
with torch.no_grad():
    while not is_end_of_sequence(output_seq):
        output = loaded_model(tokenized_text)
        
        output = np.log(output[0, -1, :].detatch().cpu().numpy()) / temp
        output = np.exp(output) / np.sum(np.exp(output))
        
        next_word_idx = np.random.choice(len(output), p=output)
        next_word = token_to_index[next_word_idx]

        output_seq.append(next_word)
        input_seq.append(next_word)

output_text = ' '.join(output_seq)
print(output_text)
"""