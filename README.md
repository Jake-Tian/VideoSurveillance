# Video Surveillance Monitor

A real-time video surveillance system that uses multimodal AI models to detect anomalies (violence and falling-down incidents) in video clips and provides an interactive web interface for analysis and search.

## Architecture

The project consists of four main components:

### 1. **gradio_app.py** - Front-end Web Interface  
Calls `surveillance.py` and `search.py`. 

### 2. **surveillance.py** - Core Processing Engine  
You can switch to a different model by changing: 
```
from utils.chat_gemini import generate_messages, get_response
```

**Key Function:**
- `streaming_process_video(clip, output_path, prompt)`: Processes a single clip and appends results to the output JSONL file. You can try different prompts here. 

### 3. **search.py** - Video Search Module  
Searches processed video descriptions and returns the relevant clip ID and answer.

### 4. **utils/** - Utility Modules

You can choose from four LLMs to process videos: 
1. `chat_qwen.py`: Uses an SFT model `M3-Memorization`.  
2. `chat_qwen_omni.py`: Uses Qwen2.5-Omni-7B.  
3. `chat_gpt.py`: Uses gpt-4o-mini (you can change it to other GPT models).  
4. `chat_gemini.py`: Uses gemini-2.5-flash.   


## Installation

Required Python packages (install via pip):
```bash
pip install gradio openai transformers torch
```
For Qwen-Omni models (refer to https://github.com/QwenLM/Qwen2.5-Omni): 
```bash
pip install -r requirements_web_demo.txt
```

## Model Setup

### Download Qwen2.5-Omni-7B Model

The system can use the Qwen2.5-Omni-7B model for video analysis. Download and set up the model from https://huggingface.co/Qwen/Qwen2.5-Omni-7B. 
The model should be located at `models/Qwen2.5-Omni-7B/` with the following structure:
   ```
   models/Qwen2.5-Omni-7B/
   ├── config.json
   ├── generation_config.json
   ├── model-*.safetensors
   └── ...
   ```

**Note**: If you're using other models (GPT, Gemini, or the SFT model), you don't need to download Qwen2.5-Omni-7B. Simply configure the API keys or model paths in the respective utility files.

## Data Setup

### 1. Download Video Data

Create the following directory structure:
```
data/
├── videos/          # Original video files
├── clips/           # 15-second video clips (organized by video ID)
│   ├── gym/
│   ├── warehouse/
│   └── ...
└── results/         # Analysis results (JSONL format)
```

### 2. Create data.jsonl

Create `data/data.jsonl` with the following format (one line per video):
```json
{"id": "gym", "video_path": "data/videos/gym.mp4", "clip_path": "data/clips/gym/", "output_path": "data/results/gym.json"}
{"id": "warehouse", "video_path": "data/videos/warehouse.mp4", "clip_path": "data/clips/warehouse/", "output_path": "data/results/warehouse.json"}
```

**Fields:**
- `id`: Unique identifier for the video
- `video_path`: Path to the full video file
- `clip_path`: Directory containing 15-second clip files (named as `0.mp4`, `1.mp4`, etc.)
- `output_path`: Path where analysis results will be saved (JSONL format)

### 3. Prepare Video Clips

Cut videos into 15-second segments. 

```bash
#!/bin/bash

video="gym"
input="data/videos/$video.mp4"
mkdir -p "data/clips/$video"
duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input")
duration_seconds=$(echo "$duration" | awk '{print int($1)}')
 
segments=$((duration_seconds / 15 + 1))
for ((i=0; i<segments; i++)); do
    start=$((i * 15))
    end=$(((i + 1) * 15))
    output="data/clips/$video/$i.mp4"
    ffmpeg -ss $start -i "$input" -t 15 -c copy "${output}"
done
```


### API Configuration

The search functionality uses an OpenAI-compatible API. Configure your API key in `search.py`:
```python
client = OpenAI(
    api_key = "your-api-key-here", 
    base_url = ""
)
```

## Usage

```bash
python -m surveillance
```

Modify the `__main__` block in `surveillance.py` to specify:
- `clip_path`: Directory containing video clips
- `output_path`: Output JSONL file path
- Detection prompt (violence or falling)

## Output Format

Results are saved in JSONL format (one JSON object per line):

```json
{"clip_id": 0, "response": ["WARNING: ...", "Sentence 1", "Sentence 2", ...]}
{"clip_id": 1, "response": ["Sentence 1", "Sentence 2", ...]}
```

- `clip_id`: Numeric ID of the clip (derived from filename)
- `response`: List of sentences describing the clip, with optional "WARNING:" prefix for anomalies

## Project Structure

```
VideoSurv/
├── gradio_app.py          # Main web interface
├── surveillance.py        # Core video processing
├── search.py              # Video search functionality
├── utils/
│   ├── chat_qwen_omni.py  # Qwen2.5-Omni integration
│   ├── prompts.py         # Detection prompts
│   ├── general.py         # Utility functions
│   └── ...                # Other utilities
├── data/
│   ├── data.jsonl         # Video metadata
│   ├── videos/            # Original videos
│   ├── clips/             # Video clips
│   └── results/           # Analysis results
└── models/
    └── Qwen2.5-Omni-7B/   # AI model files
```


