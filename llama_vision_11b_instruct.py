import PIL
import datasets
from datasets import load_dataset
from PIL import Image
import os
import transformers
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["HUGGINGFACE_API_TOKEN"]="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"
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)