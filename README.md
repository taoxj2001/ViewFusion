# ViewFusion

ViewFusion is a two-stage framework for **multi-view spatial reasoning** that explicitly separates **cross-view spatial pre-alignment** from **question answering**. It is trained with (1) **SFT** on structured reasoning traces and (2) **GRPO** reinforcement learning to improve correctness while stabilizing the intended two-stage reasoning behavior. 

---



## 🤗 🔗 Links 

*  [Model Weights](https://huggingface.co/xjtao/ViewFusion-4B)
* [Training Dataset](https://huggingface.co/datasets/xjtao/ViewFusion-traindata)
* [Paper](https://arxiv.org/abs/2603.06024)


---

## ⚙️ Installation

### 1) Clone

```bash
git clone https://github.com/taoxj2001/ViewFusion.git
cd ViewFusion
```

### 2) Create environment

```bash
conda create -n vf python=3.10 -y
conda activate vf
```

### 3) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

> Make sure your PyTorch/CUDA stack matches your machine. (We assume a standard Linux CUDA environment.)

---

## 📦 Dataset Preparation

###  Download & extract

Download from the dataset link above, then extract into a directory such as:

```text
VFR-traindata/
  GRPO_data/
    images/296/296_2.png
    ...
    metadata.jsonl   (or train.jsonl / val.jsonl)
  SFT_data/
    images/1/1_1.jpg
    ...
    metadata.jsonl
```


---

## 🚀 Training

ViewFusion uses a **two-stage training pipeline**:

1. **Stage 1: SFT** — learn the structured two-stage reasoning format.
2. **Stage 2: GRPO** — improve answer correctness while maintaining the reasoning structure. 

### 🧪 Stage 1: Supervised Fine-Tuning (SFT)

```bash
bash sft.sh
```

### 🏋️ Stage 2: GRPO Reinforcement Learning

```bash
bash grpo.sh
```

---


## 📊 Evaluation

### 🔍 MMSI-Bench

Before evaluation, download the MMSI images and place them under `MMSI/images/` using the expected naming convention, e.g.:

```text
MMSI/images/0_0.jpg
MMSI/images/0_1.jpg
...
```

Then run:

```bash
python eval_mmsi.py
```



---

## 🧩 Prompt / Output Format

ViewFusion enforces a strict 3-part output in order:

1. `<spatial_thinking>`: infer viewpoint changes / cross-view correspondences
2. `<thinking>`: solve the question conditioned on the spatial workspace
3. `<answer>`: output only the option letter (A/B/C/D)

(See the prompt template described in the paper appendix.) 

---

## 📝 Citation

```bibtex
@misc{tao2026viewfusionstructuredspatialthinking,
      title={ViewFusion: Structured Spatial Thinking Chains for Multi-View Reasoning}, 
      author={Xingjian Tao and Yiwei Wang and Yujun Cai and Yifan Song and Jing Tang},
      year={2026},
      eprint={2603.06024},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.06024}, 
}
```


