# Multimodal Phi-3 Finetuning

[![LinkedIn][linkedin-shield]][linkedin-url]

## 🛠️ Technology Stack
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.4](https://img.shields.io/badge/torch-v2.4-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![trl 0.10.1](https://img.shields.io/badge/trl-v0.10.1-violet)](https://huggingface.co/docs/trl/index)
[![Transformers 4.44.2](https://img.shields.io/badge/transformers-v4.44.2-red)](https://huggingface.co/docs/transformers/index)
[![PEFT 0.12.0](https://img.shields.io/badge/peft-v0.12.0-lightblue)](https://huggingface.co/docs/peft/index)
[![datasets 3.0.0](https://img.shields.io/badge/datasets-v2.15.0-orange)](https://huggingface.co/docs/datasets/index)
[![bitsandbytes 0.43.3](https://img.shields.io/badge/bitsandbytes-v0.43.3-green)](https://huggingface.co/blog/hf-bitsandbytes-integration)

## 📝 Overview
This project implements a multimodal Large Language Model (LLM) based on Phi-3, capable of processing images, audio, and text inputs to generate text outputs. The model is finetuned on a 150K instruction dataset using QLoRA techniques.

### Key Features
- 🖼️ **Image Processing**: Utilizes CLIP for image embeddings with a custom projection layer
- 🎵 **Audio Processing**: Implements Whisper ASR for audio-to-text conversion
- 📝 **Text Processing**: Leverages Phi-3's native tokenization and embeddings
- 🚀 **Efficient Training**: Uses QLoRA for parameter-efficient finetuning

## 📂 Project Structure
- [**generate_embeddings.py**](generate_embeddings.py)
    - Script to generate and store image embeddings using CLIP model.
- [**pretraining**](image_funetuning/pretraining/main.py)
    - Contains architecture of projection layer.
    - Class to load and process data.
    - Script to train projection layer.
- [**finetuning**](image_funetuning/finetuning/finetune.py)
    - Script to finetune Phi model on instruct 150K dataset

## :chart_with_upwards_trend: Projection layer training

    Loss: 6.3225 Batch_id=6571: 100%|██████████| 6572/6572 [48:00<00:00,  2.28it/s]
    Epoch 2/3
    Loss: 6.1490 Batch_id=6571: 100%|██████████| 6572/6572 [47:59<00:00,  2.28it/s]
    Epoch 3/3
    Loss: 6.1416 Batch_id=6571: 100%|██████████| 6572/6572 [47:59<00:00,  2.28it/s]

## :chart_with_upwards_trend: Finetuning using QLora

    Step	Training Loss
    100	    12.643800
    200	    12.441300
    300	    11.495200
    400	    7.042100
    500	    3.107100
    600	    2.746800
    700	    2.546100
    800	    2.320600
    900	    2.036000
    1000	1.992700
    .
    .
    .
    5100	1.321800
    5200	1.321600
    5300	1.333300
    5400	1.341200
    5500	1.306900
    5600	1.318700
    5700	1.328000
    5800	1.317500
    5900	1.311600
    6000	1.308100

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Installation
1. Clone the repository
```bash
git clone https://github.com/RInkalshah93/multi_model_phi_3.git 
cd multi_model_phi_3
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

### Training Pipeline
```bash
cd image_finetuning

# 1. Generate image embeddings
python generate_embeddings.py

# 2. Train the projection layer
python pretraining/main.py

# 3. Finetune on instruction dataset
python finetuning/finetune.py
```

## 🔄 Future Improvements
1. **Enhanced Projection Layer Training**
   - Current training loss plateaus at 6
   - Potential for improved architecture and training strategy

2. **Training Monitoring**
   - Implement periodic inference checks
   - Add validation metrics

3. **Image Processing Enhancement**
   - Implement patch-based image processing
   - Explore alternative embedding techniques

4. **Dataset Expansion**
   - Incorporate diverse data sources
   - Enhance multimodal capabilities

## 📚 References
1. [Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485)
2. [Phi-3 Technical Report](https://arxiv.org/pdf/2404.14219)
3. [Parameter-Efficient Fine-Tuning Methods](https://arxiv.org/pdf/2312.12148)
4. [LoRA: Low-Rank Adaptation](https://arxiv.org/pdf/2106.09685)
5. [QLoRA: Efficient Finetuning](https://arxiv.org/pdf/2305.14314)

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: www.linkedin.com/in/rinkalkumar4
