# FSANet: Frequency Spatial Attention Network
Pytorch Implementation of paper:

> **Hand Gesture Authentication by Discovering Fine-grained Spatiotemporal Identity Characteristics**
>
> Wenwei Song, Wenxiong Kang\*, and Liang Lin.

## Main Contribution
 Dynamic hand gesture is an emerging and promising biometric trait containing both physiological and behavioral characteristics. Possessing the two characteristics makes dynamic hand gesture have more identity information enabling more accurate and secure authentication theoretically, but also poses a challenge of fine-grained spatiotemporal feature extraction and analysis. This challenge involves a seemingly paradoxical problem that high-frame-rate videos are required for behavioral characteristic analysis, but they can introduce a large amount of redundant information that is detrimental to model learning and also impose a large amount of computation. To mitigate this issue, we propose a Frequency Spatial Attention Network (FSANet) with a focus on satisfying the high-performance and low-computation requirements of authentication systems. The FSANet is established with a two-stage behavioral characteristic analysis paradigm according to the properties of behavioral characteristics. Specifically, most behavioral characteristics are reflected in adjacent video frames (short-range motions), so we design a novel behavior-highlighted pseudo-modality termed as Temporal Difference Flow (TD-Flow) for local behavioral feature distillation in the first stage, and then present a Frequency Spatial Attention (FSA) module to obtain long-range spatial-temporal dependencies of hand gestures for global behavioral feature summarization with decent computation consumption and GPU memory occupation in the second stage. Incorporating TD-Flow and FSA module enables them to complement each other's strengths, resulting in a clear-cut improvement in terms of equal error rate and running speed. Extensive experiments on the SCUT-DHGA dataset demonstrate the efficiency and efficacy of FSANet.
 
 <div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/FSANet/main/img/TDFlow.png" />
  The overall architecture of FSANet. FSANet consists of three components: 2D CNN backbone (bottom right corner), TD-Flow Module (top half), and FSA Module (bottom left corner). 
</p>
</div>

 <div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/FSANet/main/img/FSA.png" />
  FSA Module. FSA is mainly formed by four different convolutions, which are utilized to learn the bases and filter, channel compression, attention query, as well as channel adaptation and reconstruction. "Normalize" denotes vector normalization. "Corr" represents the correlation function. "Abs" represents the absolute value function. $\boldsymbol{E}$ is the identity matrix. The frequency domain filter learning is embedded in the process of frequency domain transform basis calculation and selection. We guarantee the variance of the transform bases by adding an additional loss function to penalize the bases that are too similar to others.
</p>
</div>

## Comparisons with selected SOTAs
We conduct experiments in three dimensions, including pseudo-modality, attention module, and video understanding network architecture, to justify the superiority of our FSANet in terms of EER and resource consuming. Extensive experiments evidence that FSANet achieves SOTA results on the SCUT-DHGA dataset under the MG and UMG setting. The performance of some representative models (selected from the experiment part) under the MG setting are shown below.

 <div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/FSANet/main/img/FSA_SOTA.png" />
  Dynamic hand gesture authentication performance comparison on SCUT-DHGA in terms of equal error rate (EER), computational cost (FLOPs/Video), and model size (#Params). Our proposed FSANet achieves the best trade-off between accuracy and efficiency, compared with the excellent previous methods selected from the experiment part. These models cover 3D CNN, two-stream CNN, 2D CNN, temporal difference, and attention module based on our proposed TD-Flow (the categories of these models are not strictly mutually exclusive).
 </p>
</div>

## Dependencies
Please make sure the following libraries are installed successfully:
- [PyTorch](https://pytorch.org/) >= 1.7.0

## How to use
This repository is a demo of FSANet. Through debugging ([main.py](/main.py)), you can quickly understand the 
configuration and building method of [FSANet](/model/FSANet.py), including TD-Flow and FSA module.

If you want to explore the entire dynamic hand gesture authentication framework, please refer to our pervious work [SCUT-DHGA](https://github.com/SCUT-BIP-Lab/SCUT-DHGA) 
or send an email to Prof. Kang (auwxkang@scut.edu.cn).