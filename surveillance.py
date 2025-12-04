import argparse
import glob
import importlib
import json
import os
import time
from pathlib import Path

from utils.general import parse_list_string, remove_python_code
from utils.prompts import prompt_violence_detection, prompt_falling_detection

DEFAULT_CONFIG_PATH = "configs/falling_default.json"

PROMPT_REGISTRY = {
    "violence_detection": prompt_violence_detection,
    "falling_detection": prompt_falling_detection,
}

MODEL_REGISTRY = {
    "gemini": "utils.chat_gemini",
    "gpt4o": "utils.chat_gpt",
    "qwen_omni": "utils.chat_qwen_omni",
    "qwen_sft": "utils.chat_qwen",
}


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_prompt(prompt_cfg: dict) -> tuple[str, str]:
    prompt_cfg = prompt_cfg or {}
    prompt_type = prompt_cfg.get("type", "falling_detection")

    if prompt_cfg.get("text_override"):
        return prompt_type, prompt_cfg["text_override"]

    prompt_file = prompt_cfg.get("file")
    if prompt_file:
        file_path = Path(prompt_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        return prompt_type, file_path.read_text(encoding="utf-8")

    prompt = PROMPT_REGISTRY.get(prompt_type)
    if prompt is None:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return prompt_type, prompt


def load_model_functions(model_key: str):
    module_path = MODEL_REGISTRY.get(model_key)
    if not module_path:
        raise ValueError(f"Unsupported model '{model_key}'. Available: {list(MODEL_REGISTRY)}")
    module = importlib.import_module(module_path)
    return module.generate_messages, module.get_response


def build_message_kwargs(video_cfg: dict) -> dict:
    video_cfg = video_cfg or {}
    frame_interval = video_cfg.get("frame_interval", 50)
    frame_size = video_cfg.get("frame_size")
    upload_mode = video_cfg.get("upload_mode", "frames")

    if isinstance(frame_size, list) and len(frame_size) == 2:
        frame_size = {"width": frame_size[0], "height": frame_size[1]}

    message_kwargs = {
        "frame_interval": frame_interval,
        "frame_size": frame_size,
        "upload_mode": upload_mode,
    }
    return message_kwargs


def parse_response_text(response_text: str, output_cfg: dict):
    output_cfg = output_cfg or {}
    fmt = output_cfg.get("format", "python_list")
    cleaned = response_text.strip()

    if fmt == "python_list":
        cleaned = remove_python_code(cleaned)
        return parse_list_string(cleaned)

    if fmt == "json":
        cleaned = remove_python_code(cleaned)
        return json.loads(cleaned)

    if fmt == "warning_only":
        cleaned = remove_python_code(cleaned)
        return {"warning": cleaned}

    if fmt == "text":
        return cleaned

    raise ValueError(f"Unsupported output format: {fmt}")


def streaming_process_video(
    clip: str,
    output_path: str,
    prompt: str,
    generate_messages,
    get_response,
    output_cfg: dict,
    message_kwargs: dict,
    metadata: dict,
):
    messages = generate_messages(clip, prompt, **(message_kwargs or {}))
    response_text = get_response(messages)
    print(response_text)

    response_payload = parse_response_text(response_text, output_cfg)

    clip_stem = Path(clip).stem
    try:
        clip_id = int(clip_stem)
    except ValueError:
        clip_id = clip_stem
    result_entry = {
        "clip_id": clip_id,
        "response": response_payload,
        "meta": {
            **(metadata or {}),
            "clip_path": clip,
            "timestamp": time.time(),
        },
    }

    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(result_entry, f, ensure_ascii=False)
        f.write("\n")

    return response_payload


def main():
    parser = argparse.ArgumentParser(description="Run video surveillance experiment with a config file.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the experiment config JSON file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    prompt_type, prompt_text = resolve_prompt(config.get("prompt"))
    model_key = config.get("model", "gemini")
    generate_messages, get_response = load_model_functions(model_key)

    paths_cfg = config.get("paths", {})
    clip_path = paths_cfg.get("clip_path", "data/clips/warehouse/")
    output_path = paths_cfg.get("output_path", "data/results/warehouse.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_cfg = config.get("output", {"format": "python_list"})
    message_kwargs = build_message_kwargs(config.get("video"))
    run_cfg = config.get("run", {})
    metadata = {
        "config_name": config.get("name"),
        "model": model_key,
        "prompt_type": prompt_type,
        "output_format": output_cfg.get("format", "python_list"),
    }

    clips = sorted(glob.glob(os.path.join(clip_path, "*.mp4")))
    if not clips:
        raise FileNotFoundError(f"No clips found in {clip_path}")

    max_clips = run_cfg.get("max_clips")
    if isinstance(max_clips, int):
        clips = clips[:max_clips]

    # Reset output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("")

    start_time = time.time()
    clip_times = []
    for clip in clips:
        clip_start = time.time()
        response = streaming_process_video(
            clip,
            output_path,
            prompt_text,
            generate_messages,
            get_response,
            output_cfg,
            message_kwargs,
            metadata,
        )
        if isinstance(response, list):
            for line in response:
                print(line)
        else:
            print(response)
        clip_times.append(time.time() - clip_start)

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    if clip_times:
        avg_time = sum(clip_times) / len(clip_times)
        print(f"Processed {len(clip_times)} clips, average {avg_time:.2f} seconds per clip")


if __name__ == "__main__":
    main()
