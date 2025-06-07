from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch
import cv2
import os

def load_video(video_path, num_frames=16):
    """
    Load video and extract frames for VideoMAE model
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract (evenly spaced)
    if total_frames < num_frames:
        # If video has fewer frames than needed, repeat frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to 224x224
            frame = cv2.resize(frame, (224, 224))
            # Convert to float and normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            # Change from HWC to CHW format
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame {frame_idx}")
            # Use the last valid frame if available
            if frames:
                frames.append(frames[-1])
            else:
                # Create a black frame
                black_frame = np.zeros((3, 224, 224), dtype=np.float32)
                frames.append(black_frame)
    
    cap.release()
    
    # Convert to numpy array with shape (num_frames, channels, height, width)
    video_array = np.stack(frames, axis=0)
    return video_array

# Load a real video
video_path = "../videos/11111.mp4"
if os.path.exists(video_path):
    print(f"Loading video: {video_path}")
    video = load_video(video_path, num_frames=16)
    print(f"Video shape: {video.shape}")
    print(f"Video value range: [{video.min():.3f}, {video.max():.3f}]")
else:
    print(f"Video file not found: {video_path}")
    print("Using random data instead...")
    # Fallback to random data
    video = np.random.rand(16, 3, 224, 224).astype(np.float32)

# Convert to list format expected by the processor
video = list(video)

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")

inputs = processor(video, return_tensors="pt", do_rescale=False)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
