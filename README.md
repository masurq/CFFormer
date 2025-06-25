# CFFormer
This repo holds code for [CFFormer: A Cross-Fusion Transformer Framework for the Semantic Segmentation of Multi-Source Remote Sensing Images](https://ieeexplore.ieee.org/document/10786275)
# The overall architecture
We propose a novel network framework based on a transformer model, which uses the FCM and FFM to facilitate the fusion of heterogeneous data sources and achieve more accurate semantic segmentation. The algorithmic framework of this paper is shown in Figure. In detail, the proposed approach relies on the classical encoder-decoder architecture, where the encoder incorporates feature extraction networks without weight sharing: the FCM for filtering diverse modal noise and differences, and the FFM for enhancing the information interaction and fusion. The decoder part aggregates the multi-scale features to generate the final result. Other common methods such as ResNet can be employed as an alternative for the feature extraction network.
![overall architecture](dataset/figure.png)
# Citation
If you find this work useful, please consider citing:

```bibtex
@ARTICLE{10786275,
  author={Zhao, Jinqi and Zhang, Ming and Zhou, Zhonghuai and Wang, Zixuan and Lang, Fengkai and Shi, Hongtao and Zheng, Nanshan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={CFFormer: A Cross-Fusion Transformer Framework for the Semantic Segmentation of Multisource Remote Sensing Images}, 
  year={2025},
  volume={63},
  number={},
  pages={1-17},
  keywords={Feature extraction;Optical imaging;Adaptive optics;Optical sensors;Semantic segmentation;Transformers;Remote sensing;Correlation;Noise;Fuses;Feature correction module (FCM);feature fusion module (FFM);multisource remote sensing images (RSIs);semantic segmentation;vision transformer},
  doi={10.1109/TGRS.2024.3507274}}
