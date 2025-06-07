from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch
import os

model_checkpoint = "OpenGVLab/InternVL3-8B-hf"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, quantization_config=quantization_config)

# Use absolute path for local video file
video_path = os.path.abspath("../videos/11111.mp4")

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Video file not found: {video_path}")
    print("Available videos:")
    videos_dir = os.path.abspath("../videos/")
    if os.path.exists(videos_dir):
        for file in os.listdir(videos_dir):
            if file.endswith(('.mp4', '.avi', '.mov')):
                print(f"  - {os.path.join(videos_dir, file)}")
    exit(1)

print(f"Processing video: {video_path}")

# Use the exact format from Hugging Face documentation
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "url": video_path,  # Use "url" key as shown in documentation
            },
            {"type": "text", "text": "Please describe what happened in the video in detail. And let me konw what this action brings, and ended at, or what will happen in the next. Focusing on the actions and behaviors of the characters. After describing the video, please provide the action label in the following format on the last line:\n[Action Label]: Based on your entire structure, summarize a concise action description, focusing only on human actions, without describing scene information."},
        ],
    }
]

try:
    # Follow the exact pattern from documentation
    inputs = processor.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        num_frames=8,
    ).to(model.device, dtype=torch.float16)

    print("Generating response...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"Model response: {decoded_output}")
    
    # Extract action label from the response
    lines = decoded_output.split('\n')
    action_label = None
    for line in lines:
        if line.strip().startswith('[Action Label]'):
            action_label = line.strip()
            break
    
    if action_label:
        print(f"\nExtracted action label: {action_label}")
    else:
        print("\nNo formatted action label found, trying to extract from the last line...")
        last_line = lines[-1].strip() if lines else ""
        if last_line:
            print(f"Last line content: {last_line}")

except Exception as e:
    print(f"Error processing video: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()