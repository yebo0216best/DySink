## **DySink: Dynamic Frame Sinks for Autoregressive Long Video Generation**  
Bo Ye, Xinyu Cui, Jian Zhao, Tong Wei, Min-Ling Zhang  
*arXiv 2026* [[Paper](https://arxiv.org/pdf/2605.21028)] [[Code](https://github.com/yebo0216best/DySink)] [[Weight](https://huggingface.co/yebo0216best/DySink-weights)] 

## 📌 Overview
**DySink** is a retrieval-based framework for autoregressive long video generation that replaces **static frame sinks** (i.e., early frames cached as permanent anchors) with **dynamic frame sinks** retrieved from a compact memory bank. By adaptively selecting visually relevant historical frames and suppressing collapse‑prone attention patterns via a lightweight **sink anomaly gate**.

**Note**: **The code and weights have been released**. Please feel free to contact me directly by email if you have any related questions. My email address is [yeb@seu.edu.cn](mailto:yeb@seu.edu.cn).


## 📦 Installation
DySink shares its environment with LongLive [**LongLive installation guide**](https://nvlabs.github.io/LongLive/LongLive2/docs/#installation).

## 📄 Citation
```bibtex
@article{ye2026dysink,
  title={DySink: Dynamic Frame Sinks for Autoregressive Long Video Generation},
  author={Ye, Bo and Cui, Xinyu and Zhao, Jian and Wei, Tong and Zhang, Min-Ling},
  journal={arXiv preprint arXiv:2605.21028},
  year={2026}
}
```

## 🙏 Acknowledgements
Dysink builds on the codebases and ideas of:
- [LongLive](https://github.com/NVlabs/LongLive): the base AR long-video generation framework.
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing): the causal AR training recipe and prompts.
- [Wan](https://github.com/Wan-Video/Wan2.1): the base video generation models.
