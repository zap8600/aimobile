import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer 
import coremltools as ct
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

sentence_fragment = "The Oceans are"

def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    """ Implements nucleus (top-p) filtering """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cum_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Fill the scores of the filtered-out tokens with a very small value
    sorted_logits[sorted_indices_to_remove] = filter_value

    # Restore the original order of the logits
    indices = torch.argsort(sorted_indices)
    logits = logits[indices]
    return logits

class GPT2(torch.nn.Module):
    def __init__(self, model):
        super(GPT2, self).__init__()
        self.next_token_predictor = model
    
    def forward(self, x):
        inputs = x
        inputs = self.next_token_predictor(inputs)

token_predictor = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True).eval()

context = torch.tensor(tokenizer.encode(sentence_fragment))
random_tokens = torch.randint(10000, (512,))
traced_token_predictor = torch.jit.trace(token_predictor, random_tokens)

model = GPT2(model=traced_token_predictor)
scripted_model = torch.jit.script(model)

# Custom model

sentence_fragment = "The Oceans are"

torch_out = scripted_model(context)
filtered_logits = top_p_filtering(torch_out[0][-1], top_p=0.9)

probs = F.softmax(filtered_logits, dim=-1)
next_token = torch.multinomial(probs, 1)

generated_text = tokenizer.decode(torch.cat((input_ids[0], next_token)).tolist())
print(generated_text)

# Stock model

model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True).eval()

sentence_fragment = "The Oceans are"

input_ids = tokenizer(sentence_fragment, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids, do_sample=True, max_length=20)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print("Stock model: "+gen_text)
