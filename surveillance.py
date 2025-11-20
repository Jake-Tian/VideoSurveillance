# python -m surveillance

import time
import glob
import os
import json
import ast
from pathlib import Path
from utils.general import parse_list_string, remove_python_code
from utils.prompts import prompt_violence_detection, prompt_falling_detection
from utils.chat_gemini import generate_messages, get_response


def streaming_process_video(clip, output_path, prompt): 

    messages = generate_messages(clip, prompt)
    response = get_response(messages)
    print(response)
    response = remove_python_code(response)
    response_list = parse_list_string(response)
    # response_list = ast.literal_eval(response)
    
    clip_id = int(clip.split("/")[-1].split(".")[0])
    # Append the response as a new line to the JSON file (JSONL format)
    result_entry = {
        "clip_id": clip_id,
        "response": response_list
    }
    
    with open(output_path, 'a', encoding='utf-8') as f:
        json.dump(result_entry, f, ensure_ascii=False)
        f.write('\n')  # Add newline for JSONL format
    
    return response_list


if __name__ == "__main__":
    
    start_time = time.time()

    # print(prompt_violence_detection)
    # print("--------------------------------")

    clip_path = "data/clips/warehouse/"

    clips = glob.glob(os.path.join(clip_path, "*.mp4"))
    output_path = "data/results/warehouse.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('')

    for clip in clips:
        response = streaming_process_video(clip, output_path, prompt_falling_detection)
        for line in response:
            print(line)

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")