# python utils/chat_gpt.py

# from prompts import prompt_violence_detection
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key = "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", 
            base_url = "https://openrouter.ai/api/v1"
        )
    return _client


def get_response(messages):
    client = _get_client()
    response = client.responses.create(
        model="openai/gpt-4o-mini",
        input=messages,
    )
    return response.output_text


def generate_messages(video_path, prompt, frame_interval=50, frame_size=None, upload_mode="frames"):
    """Encode frames from the clip and build a GPT-4o request."""   

    video = cv2.VideoCapture(video_path)
    interval = max(1, int(frame_interval or 1))
    resize_dims = None
    if frame_size:
        width = frame_size.get("width")
        height = frame_size.get("height")
        if width and height:
            resize_dims = (int(width), int(height))

    content = [
        {
            "type": "input_text",
            "text": prompt
        }
    ]

    frame_index = 0
    while True:
        grabbed = video.grab()
        if not grabbed:
            break
        if frame_index % interval == 0:
            success, frame = video.retrieve()
            if not success:
                break
            if resize_dims:
                frame = cv2.resize(frame, resize_dims)
            _, buffer = cv2.imencode(".jpg", frame)
            encoded = base64.b64encode(buffer).decode("utf-8")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "low"  # Use "low" detail to reduce context length usage
                }
            )
        frame_index += 1

    video.release()

    messages = [{
        "role": "user",
        "content": content
    }]
    return messages


if __name__ == "__main__":
    start_time = time.time()

    messages = generate_messages("data/clips/gym/0.mp4", prompt_violence_detection)
    response = get_response(messages)
    print(response)
    print(type(response))
    print(len(response))

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time} seconds")
