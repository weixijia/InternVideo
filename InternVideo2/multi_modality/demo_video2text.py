import numpy as np
import os
import io
import cv2
import warnings

# 抑制常见的库警告信息
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch

from utils.config import (Config,
                    eval_dict_leaf)

from demo.utils import (retrieve_text,
                  _frame_from_video,
                  setup_internvideo2)

video = cv2.VideoCapture('demo/example2.mp4')
frames = [x for x in _frame_from_video(video)]


text_candidates = ["A girl is swinging her arms.",
                   "A person is doing arm exercises.",
                   "A person is waving their hands in the air.",
                   "A person is playing tennis or badminton.",
                   "A person is driving a car.",
                   "A team is playing football.",
                   "A girl is conducting an orchestra.",
                   "A man and a dog are playing in the snow.",
                   "A man is eating a sandwich."]

config = Config.from_file('demo/internvideo2_stage2_config.py')
config = eval_dict_leaf(config)

intern_model, tokenizer = setup_internvideo2(config)

# 测试所有候选文本的概率
texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=len(text_candidates), config=config)

print("所有候选描述的匹配概率：")
for t, p in zip(texts, probs):
    print(f'  {t} → {p:.4f}')
    
print(f'\n🎯 最匹配的描述: {texts[0]} (概率: {probs[0]:.4f})')