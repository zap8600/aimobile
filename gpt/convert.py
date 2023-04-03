import tensorflow as tf
from transformers import TFOpenAIGPTLMHeadModel

model = TFOpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

input_spec = tf.TensorSpec([1, 64], tf.int32)
model._set_inputs(input_spec, training=False)

print(model.inputs)
print(model.outputs)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

open("./model/gpt.tflite", "wb").write(tflite_model)