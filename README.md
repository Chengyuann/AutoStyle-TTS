# AutoStyle-TTS [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

**[📝 Paper]** | ​
**[🤖 Model Zoo]** | ​
**[📊 Dataset]** | 
**[🛠️ Issues]** 

**Official PyTorch implementation of "AutoStyle-TTS: Retrieval-Augmented Generation based Automatic Style Matching Text-to-Speech Synthesis"**  
[[Project Page]](https://chengyuann.github.io/AutoStyle-TTS-DEMO)  • [[Colab]](https://colab.research.google.com/github/Chengyuann/AutoStyle-TTS)

<img width="1842" height="1044" alt="image" src="https://github.com/user-attachments/assets/88e81c3a-c933-4136-84d7-c8e45f14f6df" />


## ✨ Key Features
- ​**Dynamic Style Matching**  
  🎯 Multi-modal retrieval with Llama/PER-LLM-Embedder/Moka embeddings  
  🔍 Context-aware style selection from 1000+ curated speech samples

- ​**High-Quality Synthesis**  
  🎧 24kHz neural vocoder with prosody transfer  
  🌐 Support English/Chinese/Japanese multilingual synthesis

- ​**Developer Friendly**  
  🚀 <5s inference time on single GPU  
  📦 Pre-trained models & style database available

## 🛠️ Installation
```bash
# Clone with voice samples (2.5GB)
git clone https://github.com/Chengyuann/AutoStyle-TTS.git
cd AutoStyle-TTS

# Install core dependencies
conda create -n asttts python=3.9
conda activate asttts
pip install -r requirements.txt

#predata process

utils/
# train
python scripts/train.sh

```
🏋️ Pretrained Weights
Model	Download Link(https://cloud.tsinghua.edu.cn/d/a926bbefddfe40fbb041/) 

⚠️ License
This implementation is for research purposes only. Commercial use requires written permission. See LICENSE for details.

References
LLaMA: GitHub 
Qwen2.5: Technical Report
