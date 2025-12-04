# python -m gradio_app

import os
# Set Gradio temp directory to avoid permission issues with /tmp/gradio/
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), ".gradio_tmp")


import json
import time
import glob
import threading
import queue
import tempfile
import re
import html
import gradio as gr
from pathlib import Path
from typing import Tuple, Dict, List, Any

from surveillance import (
    streaming_process_video,
    load_config,
    resolve_prompt,
    load_model_functions,
    build_message_kwargs,
)
from search import response as search_response
from utils.prompts import prompt_violence_detection, prompt_falling_detection

DATA_JSONL_PATH = "data/data.jsonl"
CONFIG_DIRS = [Path("configs"), Path("gemini-configs"), Path("gpt-configs")]


def list_config_files() -> List[str]:
    """Return available config files across known config directories."""
    config_paths = []
    for cfg_dir in CONFIG_DIRS:
        if not cfg_dir.exists():
            continue
        config_paths.extend(sorted(str(p) for p in cfg_dir.glob("*.json")))
    return config_paths


def load_video_data(jsonl_path: str) -> Dict[str, List]:
    """Load video data from JSONL file."""
    video_data = {}


    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            video_data[entry['id']] = [entry['video_path'], entry['clip_path'], entry['output_path']]
    return video_data


def get_detection_prompt(detection_type: str):
    """Map user input to the appropriate prompt."""
    if not detection_type:
        return prompt_violence_detection  # Default to violence detection
    
    detection_type_lower = detection_type.lower().strip()
    
    # Check for violence-related keywords
    violence_keywords = ['violent', 'violence', 'crime', 'criminal', 'attack', 'assault', 'threat']
    if any(keyword in detection_type_lower for keyword in violence_keywords):
        return prompt_violence_detection
    
    # Check for falling-related keywords
    falling_keywords = ['fall', 'fell', 'strip', 'falling', 'trip', 'collapse']
    if any(keyword in detection_type_lower for keyword in falling_keywords):
        return prompt_falling_detection
    
    # Default to violence detection if no match
    return prompt_violence_detection


def normalize_response_lines(response: Any) -> List[str]:
    """Convert any response payload into a list of strings for display."""
    if response is None:
        return []
    if isinstance(response, list):
        if all(isinstance(item, dict) for item in response):
            return [json.dumps(item, ensure_ascii=False) for item in response]
        return [json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item) for item in response]
    if isinstance(response, dict):
        return [json.dumps(response, ensure_ascii=False)]
    return [str(response)]


def load_config_bundle(config_path: str, detection_type: str = "") -> Dict[str, Any]:
    """Prepare config details (prompt/model/message kwargs) for processing."""
    config = load_config(config_path)
    prompt_type, prompt_text = resolve_prompt(config.get("prompt"))

    if detection_type and detection_type.strip():
        prompt_type = "custom"
        prompt_text = get_detection_prompt(detection_type)

    model_key = config.get("model", "gemini")
    generate_messages, get_response = load_model_functions(model_key)
    output_cfg = config.get("output", {"format": "python_list"})
    message_kwargs = build_message_kwargs(config.get("video"))
    paths_cfg = config.get("paths", {})
    clip_path = paths_cfg.get("clip_path")
    output_path = paths_cfg.get("output_path")
    if not clip_path or not output_path:
        raise ValueError("Config must specify paths.clip_path and paths.output_path")

    metadata = {
        "config_name": config.get("name"),
        "model": model_key,
        "prompt_type": prompt_type,
        "output_format": output_cfg.get("format", "python_list"),
        "config_path": config_path,
    }

    return {
        "config": config,
        "prompt_text": prompt_text,
        "clip_path": clip_path,
        "output_path": output_path,
        "output_cfg": output_cfg,
        "message_kwargs": message_kwargs,
        "metadata": metadata,
        "generate_messages": generate_messages,
        "get_response": get_response,
    }


def process_clip_worker(clip, output_path, prompt, result_queue, clip_idx, total_clips, config_bundle):
    """Worker function to process a clip in a separate thread."""
    try:
        response = streaming_process_video(
            clip,
            output_path,
            prompt,
            config_bundle["generate_messages"],
            config_bundle["get_response"],
            config_bundle["output_cfg"],
            config_bundle["message_kwargs"],
            config_bundle["metadata"],
        )
        result_queue.put(('result', clip_idx, response))
    except Exception as e:
        result_queue.put(('error', clip_idx, str(e)))


def run_surveillance(config_bundle: Dict[str, Any]):
    """Process clips defined by the config bundle and yield UI updates."""
    clip_path = config_bundle["clip_path"]
    output_path = config_bundle["output_path"]
    prompt = config_bundle["prompt_text"]
    start_time = time.time()
    
    # Get all video clips in the clip_path directory
    clips = glob.glob(os.path.join(clip_path, "*.mp4"))
    clips.sort()  # Sort for consistent processing order
    
    if not clips:
        yield "No clips found in the specified directory."
        return
    
    # Clear the output file if it exists to start fresh
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    if output_path_obj.exists():
        output_path_obj.unlink()
    
    total_clips = len(clips)
    
    # Initial message
    output_lines = []
    output_lines.append(f"Starting processing...")
    output_lines.append("")
    yield "\n".join(output_lines)
    
    result_queue = queue.Queue()
    pending_results: Dict[int, Tuple[str, Any]] = {}

    def fetch_result(target_idx: int) -> Tuple[str, Any]:
        """Fetch result for a specific clip index, buffering others."""
        if target_idx in pending_results:
            msg_type, result = pending_results.pop(target_idx)
            return msg_type, result
        while True:
            msg_type, result_clip_idx, result = result_queue.get()
            if result_clip_idx == target_idx:
                return msg_type, result
            pending_results[result_clip_idx] = (msg_type, result)
    
    # Process clips with overlapping: print results from clip N while processing clip N+1
    previous_result = None
    previous_clip_idx = 0
    thread = None
    
    for clip_idx, clip in enumerate(clips, 1):
        clip_name = os.path.basename(clip)
        
        # For the first clip, process it and wait for result
        if clip_idx == 1:
            thread = threading.Thread(
                target=process_clip_worker,
                args=(clip, output_path, prompt, result_queue, clip_idx, total_clips, config_bundle)
            )
            thread.start()
            thread.join()  # Wait for first clip to complete
            
            # Get first clip's result
            msg_type, result = fetch_result(1)
            if msg_type == 'result':
                previous_result = result
                previous_clip_idx = 1
            else:
                yield f"Error processing clip 1: {result}"
                return
        
        # For subsequent clips: start processing current clip, then print previous result
        else:
            # Start processing current clip in a background thread
            thread = threading.Thread(
                target=process_clip_worker,
                args=(clip, output_path, prompt, result_queue, clip_idx, total_clips, config_bundle)
            )
            thread.start()
            
            # Print previous clip's results while current clip is processing
            if previous_result is not None:
                response_lines = normalize_response_lines(previous_result)
                
                clip_block_start = f"""<div class='clip-block'>
                <div class='clip-header'>Clip {previous_clip_idx}/{total_clips}</div>
                <div class='clip-content'><ul>
                """
                clip_items = []
                
                if not response_lines:
                    clip_items.append("<li>No response generated.</li>")
                else:
                    for idx, line in enumerate(response_lines):
                        escaped_line = html.escape(line)
                        if idx == 0 and "WARNING:" in line:
                            escaped_line = "⚠️⚠️ " + escaped_line
                            clip_items.append(f"<li class='warning-line'>{escaped_line}</li>")
                        else:
                            clip_items.append(f"<li>{escaped_line}</li>")
                        
                        current_clip_html = clip_block_start + ''.join(clip_items) + "</ul></div></div>"
                        current_output = [current_clip_html] + output_lines
                        yield "\n".join(current_output)
                        time.sleep(1)
                
                clip_block_html = clip_block_start + ''.join(clip_items) + "</ul></div></div>"
                output_lines.insert(0, clip_block_html)
                yield "\n".join(output_lines)
            
            # Wait for current clip to complete
            thread.join()
            
            # Get current clip's result
            msg_type, result = fetch_result(clip_idx)
            if msg_type == 'result':
                previous_result = result
                previous_clip_idx = clip_idx
            else:
                yield f"Error processing clip {clip_idx}: {result}"
                return
    
    if previous_result is not None and previous_clip_idx == len(clips):
        response_lines = normalize_response_lines(previous_result)
        
        clip_block_start = f"""<div class='clip-block'>
        <div class='clip-header'>Clip {previous_clip_idx}/{total_clips}</div>
        <div class='clip-content'><ul>
        """
        clip_items = []
        
        if not response_lines:
            clip_items.append("<li>No response generated.</li>")
        else:
            for idx, line in enumerate(response_lines):
                escaped_line = html.escape(line)
                if idx == 0 and "WARNING:" in line:
                    escaped_line = "⚠️⚠️ " + escaped_line
                    clip_items.append(f"<li class='warning-line'>{escaped_line}</li>")
                else:
                    clip_items.append(f"<li>{escaped_line}</li>")
                
                current_clip_html = clip_block_start + ''.join(clip_items) + "</ul></div></div>"
                current_output = [current_clip_html] + output_lines
                yield "\n".join(current_output)
                time.sleep(1)
        
        clip_block_html = clip_block_start + ''.join(clip_items) + "</ul></div></div>"
        output_lines.insert(0, clip_block_html)
        yield "\n".join(output_lines)


def start_analysis(video_id: str, detection_type: str, video_data: Dict[str, List], config_path: str):
    """Run surveillance analysis with the selected config."""
    if not config_path:
        yield "Please select a config file."
        return
    
    if not video_id or video_id not in video_data:
        yield "Please select a video."
        return
    
    try:
        config_bundle = load_config_bundle(config_path, detection_type)
    except Exception as exc:
        yield f"Error loading config: {exc}"
        return
    
    _, expected_clip_path, expected_output_path = video_data[video_id]
    clip_path = config_bundle["clip_path"]
    output_path = config_bundle["output_path"]
    
    if (expected_clip_path and os.path.normpath(expected_clip_path) != os.path.normpath(clip_path)) or (
        expected_output_path and os.path.normpath(expected_output_path) != os.path.normpath(output_path)
    ):
        yield (
            "Warning: Selected video's clip/output paths differ from the config. "
            "Proceeding with paths defined in the config."
        )
    
    for output in run_surveillance(config_bundle):
        yield output


def update_video_paths(video_id: str, video_data: Dict[str, List]) -> Tuple[str, str, str]:
    """Update video_path, clip_path, and output_path based on selected video_id."""
    if not video_id or video_id not in video_data:
        return "", "", ""
    
    video_path, clip_path, output_path = video_data[video_id]
    return video_path, clip_path, output_path


def perform_search(question: str, video_id: str, video_data: Dict[str, List]) -> Tuple[str, str]:
    """Perform search on the question and return answer with relevant clip."""
    if not question or not question.strip():
        return "Please enter a question.", ""
    
    if not video_id or video_id not in video_data:
        return "Please select a video first.", ""
    
    video_path, clip_path, output_path = video_data[video_id]
    
    # Check if output file exists
    if not os.path.exists(output_path):
        return "Please process the video clips first by clicking START.", ""
    
    try:
        # Read JSONL file and format descriptions for search.py
        descriptions = []
        clip_id_to_path = {}
        
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    clip_id = entry.get('clip_id')
                    response_list = entry.get('response', [])
                    
                    # Format as "Clip [clip_id]: [description]"
                    description = f"Clip {clip_id}: " + " ".join(response_list)
                    descriptions.append(description)
                    
                    # Store clip path mapping
                    if clip_id is not None:
                        # Try to find the clip file
                        clips = glob.glob(os.path.join(clip_path, "*.mp4"))
                        for clip in clips:
                            clip_name = os.path.basename(clip)
                            try:
                                clip_num = int(clip_name.split('.')[0])
                                if clip_num == clip_id:
                                    clip_id_to_path[clip_id] = clip
                                    break
                            except ValueError:
                                continue
        
        # Create temporary file with formatted descriptions
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
            tmp_file.write('\n'.join(descriptions))
            tmp_path = tmp_file.name
        
        try:
            # Call search function with temporary file
            search_result = search_response(question, tmp_path)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # Parse the response to extract clip ID and answer
        lines = search_result.strip().split('\n')
        clip_id = None
        answer = []
        in_answer = False
        
        for line in lines:
            if line.startswith('Clip:'):
                # Extract clip ID
                clip_id_str = line.replace('Clip:', '').strip()
                try:
                    clip_id = int(clip_id_str)
                except ValueError:
                    # Try to extract number from string
                    match = re.search(r'\d+', clip_id_str)
                    if match:
                        clip_id = int(match.group())
            elif line.startswith('Answer:'):
                in_answer = True
                answer_text = line.replace('Answer:', '').strip()
                if answer_text:
                    answer.append(answer_text)
            elif in_answer:
                if line.strip():
                    answer.append(line.strip())
        
        # Format answer
        formatted_answer = '\n'.join(answer) if answer else search_result
        
        # Get clip path
        clip_file = clip_id_to_path.get(clip_id) if clip_id is not None else None
        
        return formatted_answer, clip_file if clip_file else ""
    
    except Exception as e:
        return f"Error performing search: {str(e)}", ""


def build_interface() -> gr.Blocks:
    """Construct the Gradio UI."""
    # Load video data
    video_data = load_video_data(DATA_JSONL_PATH)
    video_ids = list(video_data.keys())
    
    # Get default values
    default_id = video_ids[0] if video_ids else ""
    default_video_path, default_clip_path, default_output_path = video_data[default_id]
    
    config_choices = list_config_files()
    default_config = config_choices[0] if config_choices else ""
    
    # Custom CSS for professional light theme design
    custom_css = """
    :root {
        --bg: #ffffff;
        --bg-elevated: #f8fafc;
        --accent: #6d28d9;
        --accent-soft: rgba(109, 40, 217, 0.15);
        --text: #1e293b;
        --text-muted: #475569;
        --border-subtle: #e2e8f0;
    }
    
    .gradio-container {
        background: #ffffff !important;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif !important;
        color: var(--text) !important;
    }
    
    .main {
        max-width: 1320px !important;
        margin: 20px auto !important;
        background: #ffffff !important;
        border-radius: 30px !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.08) !important;
        padding: 20px 22px 22px !important;
    }
    
    h1 {
        font-size: 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        color: var(--text) !important;
        margin-bottom: 8px !important;
    }
    
    .markdown {
        color: var(--text-muted) !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }
    
    .markdown h2 {
        color: var(--text) !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        margin-top: 24px !important;
        margin-bottom: 12px !important;
    }
    
    .form {
        background: #f8fafc !important;
        border-radius: 24px !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.06) !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
    }
    
    .form label {
        color: #475569 !important;
        font-size: 13px !important;
        letter-spacing: 0.15em !important;
        font-weight: 500 !important;
        margin-bottom: 8px !important;
    }
    
    input[type="text"], textarea, select {
        background: #ffffff !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
        text-transform: none !important;
        text-capitalize: none !important;
    }
    
    input[type="text"]::placeholder, textarea::placeholder {
        text-transform: none !important;
        text-capitalize: none !important;
    }
    
    /* Disable auto-capitalization for text inputs - more specific selectors */
    input[type="text"],
    textarea,
    .gradio-textbox input,
    .gradio-textbox textarea,
    [data-testid*="textbox"] input,
    [data-testid*="textbox"] textarea {
        text-transform: none !important;
        text-capitalize: none !important;
    }
    
    /* Target Gradio's specific input wrappers */
    .gradio-textbox input,
    .gradio-textbox textarea {
        text-transform: none !important;
        text-capitalize: none !important;
    }
    
    input[type="text"]:focus, textarea:focus, select:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
        outline: none !important;
    }
    
    button.primary {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 14px rgba(109, 40, 217, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    
    button.primary:hover {
        background: linear-gradient(135deg, #6d28d9, #5b21b6) !important;
        box-shadow: 0 6px 20px rgba(109, 40, 217, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    
    button.secondary {
        background: #f8fafc !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
    }
    
    button.secondary:hover {
        background: #f1f5f9 !important;
        border-color: var(--accent) !important;
    }
    
    .video-container, .video-wrapper {
        border-radius: 18px !important;
        overflow: hidden !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        background: #ffffff !important;
    }
    
    .output-text {
        background: #ffffff !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        font-size: 16px !important;
        line-height: 1.8 !important;
        max-height: 600px !important;
        overflow-y: auto !important;
        padding: 16px !important;
    }
    
    .output-text::-webkit-scrollbar {
        width: 8px !important;
    }
    
    .output-text::-webkit-scrollbar-track {
        background: #f1f5f9 !important;
        border-radius: 4px !important;
    }
    
    .output-text::-webkit-scrollbar-thumb {
        background: #cbd5e1 !important;
        border-radius: 4px !important;
    }
    
    .output-text::-webkit-scrollbar-thumb:hover {
        background: #94a3b8 !important;
    }
    
    .clip-block {
        background: #f8fafc !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
    }
    
    .clip-header {
        font-weight: 600 !important;
        font-size: 16px !important;
        color: var(--accent) !important;
        margin-bottom: 12px !important;
        padding-bottom: 8px !important;
        border-bottom: 2px solid rgba(109, 40, 217, 0.2) !important;
    }
    
    .clip-content {
        padding-left: 8px !important;
    }
    
    .clip-content ul {
        margin: 0 !important;
        padding-left: 20px !important;
        list-style-type: disc !important;
    }
    
    .clip-content li {
        color: var(--text) !important;
        margin-bottom: 8px !important;
    }
    
    .warning-line {
        font-weight: 600 !important;
        color: #dc2626 !important;
    }
    
    hr {
        border: none !important;
        border-top: 1px solid rgba(226, 232, 240, 0.8) !important;
        margin: 32px 0 !important;
    }
    
    .info {
        background: #f8fafc !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 12px !important;
        color: var(--text-muted) !important;
    }
    """
    
    with gr.Blocks(title="Video Surveillance Monitor", theme=gr.themes.Soft(primary_hue="cyan"), css=custom_css) as demo:
        gr.Markdown(
            """
            # 🎥 Video Surveillance Monitor
            <div style="color: #475569; font-size: 14px; margin-top: 8px;">
            Select a video, then click <strong>START</strong> to process all clips (15 seconds) and run the surveillance model analysis.
            </div>
            """,
            elem_classes="header-section"
        )

        with gr.Row():
            video_id_dropdown = gr.Dropdown(
                choices=video_ids,
                value=default_id,
                label="Select Video",
                info="Choose a video to process"
            )
            config_dropdown = gr.Dropdown(
                choices=config_choices,
                value=default_config,
                label="Select Config",
                info="Choose an experiment config (JSON)",
                allow_custom_value=True
            )
            detection_type_input = gr.Textbox(
                label="Anomaly Detection Type",
                placeholder="e.g., violence, crime, fall, fell, strip",
                info="Detect violent behavior or falling-down",
                value=""
            )
            start_button = gr.Button("START", variant="primary", size="lg")
        
        # Create wrapper functions that capture video_data
        def start_analysis_wrapper(vid, detection_type, config_path):
            # Yield from the generator to properly stream output
            for output in start_analysis(vid, detection_type, video_data, config_path):
                yield output
        
        def update_video_player_wrapper(vid):
            video_path, _, _ = update_video_paths(vid, video_data)
            return video_path

        with gr.Row():
            # Left side: Video player
            with gr.Column(scale=3):
                video_player = gr.Video(
                    label="Video Player",
                    value=default_video_path,
                    autoplay=True,
                )
            
            # Right side: Model output
            with gr.Column(scale=2):
                output_text = gr.Markdown(
                    value="Click START to analyze the video clips...",
                    elem_classes="output-text",
                )

        start_button.click(
            fn=start_analysis_wrapper,
            inputs=[video_id_dropdown, detection_type_input, config_dropdown],
            outputs=[output_text],  # Only output to output_text, video player disconnected
        )
        
        # Update video player when selection changes
        video_id_dropdown.change(
            fn=update_video_player_wrapper,
            inputs=[video_id_dropdown],
            outputs=[video_player],
        )
        
        # Search Section
        gr.Markdown("---")
        gr.Markdown(
            """
            ## 🔍 Video Search
            <div style="color: #475569; font-size: 14px; margin-top: 8px;">
            Ask a question about the processed video clips to get an answer and find the most relevant clip.
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                search_question = gr.Textbox(
                    label="Question",
                    placeholder="e.g., Who is the perpetrator in the video?",
                    lines=2
                )
            with gr.Column(scale=1, min_width=120):
                search_button = gr.Button("Search", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=1):
                search_answer = gr.Textbox(
                    label="Answer",
                    lines=10,
                    interactive=False,
                    placeholder="Answer will appear here...",
                )
            with gr.Column(scale=1):
                search_clip_player = gr.Video(
                    label="Most Relevant Clip",
                    autoplay=False,
                )
        
        def search_wrapper(question, vid):
            answer, clip_path = perform_search(question, vid, video_data)
            return answer, clip_path
        
        search_button.click(
            fn=search_wrapper,
            inputs=[search_question, video_id_dropdown],
            outputs=[search_answer, search_clip_player],
        )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.launch(
        server_name="127.0.0.1",
        # server_port=7860,
        share=False,
        show_error=False,
        quiet=False,
        inbrowser=True,
        favicon_path=None
    )
