import torch
import torch.onnx
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

dummy_text = "I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy.I believe in rei plush supremacy."
dummy_input = tokenizer(dummy_text, return_tensors='pt')['input_ids']

tokenizer.save_pretrained("./model")
torch.onnx.export(model, dummy_input, "./model/gpt2.onnx", input_names=['input'], output_names=['output'], export_params=True)

print("Done!")