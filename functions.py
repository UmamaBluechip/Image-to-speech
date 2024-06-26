from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import torch
import soundfile as sf

def generate_image_caption(image_path):

  model_name = "microsoft/Florence-2-large"
  prompt = "<OD>"

  model = AutoModelForCausalLM.from_pretrained(model_name)
  processor = AutoProcessor.from_pretrained(model_name)

  image = Image.open(image_path)

  inputs = processor(text=prompt, images=image, return_tensors="pt")

  generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3,
      do_sample=False
  )
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
  caption = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

  return caption




def convert_text_to_speech(text, device="cpu"):

  model_name = "parler-tts/parler_tts_mini_v0.1"

  model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

  generation = model.generate(input_ids=input_ids)
  audio_arr = generation.cpu().numpy().squeeze()

  return audio_arr
