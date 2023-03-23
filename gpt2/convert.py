import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer 
import coremltools as ct
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

sentence_fragment = "A compound sentence is"

def top_p_filtering(logits, top_p=0.1, filter_value=-float('Inf')):
    """Filter logits to only keep the top tokens with cumulative probability <= top_p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold.
    sorted_indices_to_remove = cumulative_probs > top_p
    if torch.any(sorted_indices_to_remove):
        # Keep at least the top token that falls below the threshold.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value

    return logits

model = GPT2LMHeadModel.from_pretrained("gpt2")
compiled_model = torch.compile(model)

# Custom model

context = torch.tensor(tokenizer.encode(sentence_fragment))

torch_out = compiled_model(context)
filtered_logits = top_p_filtering(torch_out[0][-1], top_p=0.9)

probs = F.softmax(filtered_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
generated_text = tokenizer.decode(torch.cat((context[0], next_token.squeeze())).tolist())
print(generated_text)

"""
# Stock model

model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True).eval()

sentence_fragment = "The Oceans are"

input_ids = tokenizer(sentence_fragment, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids, do_sample=True, max_length=20)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print("Stock model: "+gen_text)
"""
