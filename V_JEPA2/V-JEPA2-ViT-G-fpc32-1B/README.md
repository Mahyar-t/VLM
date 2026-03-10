---
license: mit
pipeline_tag: video-classification
tags:
- video
library_name: transformers
datasets:
- bkprocovid19/diving48
base_model:
- facebook/vjepa2-vitg-fpc64-384
---

# V-JEPA 2

A frontier video understanding model developed by FAIR, Meta, which extends the pretraining objectives of [VJEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/), resulting in state-of-the-art video understanding capabilities, leveraging data and model sizes at scale.
The code is released [in this repository](https://github.com/facebookresearch/vjepa2).

<div style="background-color: rgba(251, 255, 120, 0.4); padding: 10px; color: black; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    💡 This is V-JEPA 2 <a href="https://huggingface.co/facebook/vjepa2-vitg-fpc64-384">ViT-g 384</a> model with video classification head pretrained on <a href="http://www.svcl.ucsd.edu/projects/resound/dataset.html" style="color: black;">Diving 48</a> dataset.
</div>
<br></br>

<img src="https://github.com/user-attachments/assets/914942d8-6a1e-409d-86ff-ff856b7346ab">&nbsp;

## Installation

To run V-JEPA 2 model, ensure you have installed the latest transformers:

```bash
pip install -U git+https://github.com/huggingface/transformers
```

## Video classification code snippet

```python
import torch
import numpy as np

from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModelForVideoClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and video preprocessor
hf_repo = "facebook/vjepa2-vitg-fpc32-384-diving48"

model = AutoModelForVideoClassification.from_pretrained(hf_repo).to(device)
processor = AutoVideoProcessor.from_pretrained(hf_repo)

# To load a video, sample the number of frames according to the model.
video_url = "https://huggingface.co/facebook/vjepa2-vitg-fpc32-384-diving48/resolve/main/sample/diving.mp4"
vr = VideoDecoder(video_url)
frame_idx = np.arange(0, model.config.frames_per_clip, 8) # you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # frames x channels x height x width

# Preprocess and run inference
inputs = processor(video, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

print("Top 5 predicted class names:")
top5_indices = logits.topk(5).indices[0]
top5_probs = torch.softmax(logits, dim=-1).topk(5).values[0]
for idx, prob in zip(top5_indices, top5_probs):
    text_label = model.config.id2label[idx.item()]
    print(f" - {text_label}: {prob:.2f}")
```
Output:
```
Top 5 predicted class names:
 - ['Forward', '35som', 'NoTwis', 'PIKE']: 0.49
 - ['Forward', '25som', 'NoTwis', 'PIKE']: 0.13
 - ['Forward', '25som', '1Twis', 'PIKE']: 0.13
 - ['Forward', '35som', 'NoTwis', 'TUCK']: 0.10
 - ['Forward', '25som', '2Twis', 'PIKE']: 0.04
```

## Citation

```
@techreport{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and
  Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and
  Arnaud, Sergio and Gejji, Abha and Martin, Ada and Robert Hogan, Francois and Dugas, Daniel and
  Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and
  Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and
  Rabbat, Michael and Ballas, Nicolas},
  institution={FAIR at Meta},
  year={2025}
}
```