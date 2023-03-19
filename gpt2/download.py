import torch
import transformers

# Load the GPT-2 model
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')

# Configure the model for inference
model.eval()
torch.set_grad_enabled(False)

# Input some sample data
input_ids = torch.rand(1, 3, 224, 224).long()

# Generate a trace of the model's computation graph
traced_model = torch.jit.trace(model, (input_ids,))

# Save the TorchScript module to a file
traced_model.save("./model/gpt2.pt")
