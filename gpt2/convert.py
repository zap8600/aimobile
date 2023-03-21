from optimum.onnxruntime import ORTModelForCausalLM

model = ORTModelForCausalLM.from_pretrained('gpt2', from_transformers=True)