from packaging import version
from onnxruntime import __version__ as ort_version
import os
import torch
import onnxruntime
import numpy
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel
from typing import List, Dict
from onnxruntime import InferenceSession
from onnxruntime.transformers.io_binding_helper import TypeHelper, IOBindingHelper

cache_dir = os.path.join(".", "cache_models")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

if version.parse(ort_version) >= version.parse("1.12.0"):
    from onnxruntime.transformers.models.gpt2.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
else:
    from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel

    raise RuntimeError("Please install onnxruntime 1.12.0 or later to run this")

model_name_or_path = "gpt2"
config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = MyGPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
device = torch.device("cpu")
model.eval().to(device)

num_attention_heads = model.config.n_head
hidden_size = model.config.n_embd
num_layer = model.config.n_layer

onnx_model_path = "./model/gpt2.onnx"

session = onnxruntime.InferenceSession(onnx_model_path)

EXAMPLE_Text = ["best hotel in bay area", "here is an example of gpt2 model"]

"""
torch_model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
device = torch.device("cpu")
torch_model.eval().to(device)
"""

def get_tokenizer(model_name_or_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_example_inputs(prompt_text=EXAMPLE_Text):
    tokenizer = get_tokenizer(model_name_or_path, cache_dir)
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.int32)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)
    position_ids = position_ids.to(torch.int32)

    # Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids, attention_mask, position_ids, empty_past

def inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, past):
    output_shapes = Gpt2Helper.get_output_shapes(
        batch_size=input_ids.size(0),
        past_sequence_length=past[0].size(3),
        sequence_length=input_ids.size(1),
        config=config,
    )
    output_buffers = Gpt2Helper.get_output_buffers(output_shapes, device)

    io_binding = IOBindingHelper.prepare_io_binding(
        session, input_ids, position_ids, attention_mask, past, output_buffers, output_shapes
    )
    session.run_with_iobinding(io_binding)

    outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(session, output_buffers, output_shapes, return_numpy=False)
    return outputs

def generate_text_onnx(tokenizer, input_text, ort_session, num_tokens_to_produce=30):
    assert len(input_text) == 1 # This function requires batch_size==1
    
    print("Text generation using OnnxRuntime...")
    
    eos_token_id = tokenizer.eos_token_id
    input_ids, attention_mask, position_ids, past = get_example_inputs(input_text)
    batch_size = input_ids.size(0)
    has_eos = torch.zeros(batch_size, dtype=torch.bool)
    all_token_ids = input_ids.clone()
    
    for step in range(num_tokens_to_produce):
        outputs = inference_with_io_binding(ort_session, config, input_ids, position_ids, attention_mask, past)
        
        next_token_logits = torch.tensor(outputs[0][:, -1, :])
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        has_eos = has_eos | (next_tokens == eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        
        input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)
        position_ids = (position_ids[:, -1] + 1).reshape(batch_size, 1)
        attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(device)
        
        past = []
        for i in range(num_layer):
            past_i = ( torch.from_numpy(outputs[i + 1]) if isinstance(outputs[i + 1], numpy.ndarray) else outputs[i + 1].clone().detach() )
            past.append(past_i.to(device))
            
        output = all_token_ids[:, -1]
        print(tokenizer.decode(output, skip_special_tokens=True), end='')
        
        if torch.all(has_eos):
            break



tokenizer = get_tokenizer(model_name_or_path, cache_dir)
input_text = EXAMPLE_Text[:1]
generate_text_onnx(tokenizer, input_text, ort_session=session)