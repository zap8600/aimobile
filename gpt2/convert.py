import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer 
import coremltools as ct
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

sentence_fragment = "The Oceans are"

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
print("Custom model: {}".format(sentence_fragment))

# Stock model

model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True).eval()

sentence_fragment = "The Oceans are"

input_ids = tokenizer(sentence_fragment, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids, do_sample=True, max_length=20)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print("Stock model: "+gen_text)