# python utils/chat_gemini.py

# from prompts import prompt_violence_detection
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from pathlib import Path
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
    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=messages,
    )
    return response.choices[0].message.content


def encode_video_to_base64(video_path: str) -> str:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def generate_messages(video_path, prompt, frame_interval=50, frame_size=None, upload_mode="frames"):
    """Build request either from sampled frames or the raw video."""

    mode = (upload_mode or "frames").lower()
    if mode == "video":
        base64_video = encode_video_to_base64(video_path)
        content = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "video_url",
                "video_url": {
                    "url": f"data:video/mp4;base64,{base64_video}"
                }
            }
        ]

        return [{
            "role": "user",
            "content": content
        }]
    else:
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
                "type": "text",
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
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}"
                        }
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
