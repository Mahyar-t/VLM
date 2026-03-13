---
license: apache-2.0
pipeline_tag: video-classification
tags:
- video
library_name: transformers
---

# V-JEPA 2

A frontier video understanding model developed by FAIR, Meta, which extends the pretraining objectives of [VJEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/), resulting in state-of-the-art video understanding capabilities, leveraging data and model sizes at scale.
The code is released [in this repository](https://github.com/facebookresearch/vjepa2).

<img src="https://github.com/user-attachments/assets/914942d8-6a1e-409d-86ff-ff856b7346ab">&nbsp;

## Installation

To run V-JEPA 2 model, ensure you have installed the latest transformers:

```bash
pip install -U git+https://github.com/huggingface/transformers
```

## Intended Uses

V-JEPA 2 is intended to represent any video (and image) to perform video classification, retrieval, or as a video encoder for VLMs.

```python
from transformers import AutoVideoProcessor, AutoModel

hf_repo = "facebook/vjepa2-vitg-fpc64-384"

model = AutoModel.from_pretrained(hf_repo)
processor = AutoVideoProcessor.from_pretrained(hf_repo)
```

To load a video, sample the number of frames according to the model. For this model, we use 64.

```python
import torch
from torchcodec.decoders import VideoDecoder
import numpy as np

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"
vr = VideoDecoder(video_url)
frame_idx = np.arange(0, 64) # choosing some frames. here, you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
video = processor(video, return_tensors="pt").to(model.device)
with torch.no_grad():
    video_embeddings = model.get_vision_features(**video)

print(video_embeddings.shape)
```

To load an image, simply copy the image to the desired number of frames.

```python
from transformers.image_utils import load_image

image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
pixel_values = processor(image, return_tensors="pt").to(model.device)["pixel_values_videos"]
pixel_values = pixel_values.repeat(1, 16, 1, 1, 1) # repeating image 16 times

with torch.no_grad():
    image_embeddings = model.get_vision_features(pixel_values)    

print(image_embeddings.shape)
```

For more code examples, please refer to the V-JEPA 2 documentation.

### Citation

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