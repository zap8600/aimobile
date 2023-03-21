from transformers import GPT2Tokenizer
from onnxruntime import InferenceSession

tokenizer= GPT2Tokenizer.from_pretrained("./model")
model = InferenceSession("./model/gpt2.onnx")

sequence = ("A compound sentence is")
token_ids