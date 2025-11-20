# python utils/chat_gpt.py

from prompts import prompt_violence_detection
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os


def get_response(messages):
    client = OpenAI(
        api_key = "", 
        base_url = "",
    )

    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages,
    )
    return response.output_text


def generate_messages(video_path, prompt):

    video = cv2.VideoCapture(video_path)
    print(video.get(cv2.CAP_PROP_FRAME_COUNT))

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    content = [
        {
        "type": "input_text",
        "text": prompt
        }, 
        *[
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{frame}"
            }
            for frame in base64Frames[0::50]
        ]
    ]

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
