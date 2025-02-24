# CFFormer
This repo holds code for [CFFormer: A Cross-Fusion Transformer Framework for the Semantic Segmentation of Multi-Source Remote Sensing Images](https://ieeexplore.ieee.org/document/10786275)
# The overall architecture
We propose a novel network framework based on a transformer model, which uses the FCM and FFM to facilitate the fusion of heterogeneous data sources and achieve more accurate semantic segmentation. The algorithmic framework of this paper is shown in `Figure`. In detail, the proposed approach relies on the classical encoder-decoder architecture, where the encoder incorporates feature extraction networks without weight sharing: the FCM for filtering diverse modal noise and differences, and the FFM for enhancing the information interaction and fusion. The decoder part aggregates the multi-scale features to generate the final result. Other common methods such as ResNet can be employed as an alternative for the feature extraction network.
![Image](模型图3.tif)
