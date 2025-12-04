
import json
import logging
import torch

import warnings
warnings.filterwarnings("ignore", message=".*generation_config.*default values have been modified.*")

# import accelerate_qwen_omni
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration, GenerationConfig
from qwen_omni_utils import process_mm_info

# Configure logging
logger = logging.getLogger(__name__)

processor, thinker = None, None

def get_response(messages):
    global thinker, processor
    if thinker is None:
        thinker = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "models/Qwen2.5-Omni-7B",
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        thinker.eval()
        thinker.disable_talker()
        processor = Qwen2_5OmniProcessor.from_pretrained("models/Qwen2.5-Omni-7B")
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    generation_config = GenerationConfig.from_pretrained("models/Qwen2.5-Omni-7B")
    
    USE_AUDIO_IN_VIDEO = True
    audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(thinker.device).to(thinker.dtype)

    # Inference: Generation of the output text and audio
    with torch.no_grad():
        generation = thinker.generate(**inputs, generation_config=generation_config, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=4096, do_sample=True, temperature=1e-6)
        generate_ids = generation[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
    # Clean up
    del generation
    del generate_ids
    del inputs
    torch.cuda.empty_cache()
    
    return response


def generate_messages(video, prompt, **kwargs):

    messages = [{
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    }]
    content = [
        {"type": "video", "video": video}, 
        {"type": "text", "text": prompt},
    ]

    messages.append({"role": "user", "content": content})
    return messages
    
