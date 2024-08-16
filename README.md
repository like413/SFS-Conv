# Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection
## This repository is the official implementation of CVPR 2024 "Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection" at: [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Unleashing_Channel_Potential_Space-Frequency_Selection_Convolution_for_SAR_Object_Detection_CVPR_2024_paper.html). 

## Abstract
Deep Convolutional Neural Networks (DCNNs) have achieved remarkable performance in synthetic aperture radar (SAR) object detection, but this comes at the cost of tremendous computational resources, partly due to extracting redundant features within a single convolutional layer. Recent works either delve into model compression methods or focus on the carefully-designed lightweight models, both of which result in performance degradation. In this paper, we propose an efficient convolution module for SAR object detection, called SFS-Conv, which increases feature diversity within each convolutional layer through a shuntperceive-select strategy. Specifically, we shunt input feature maps into space and frequency aspects. The former perceives the context of various objects by dynamically adjusting receptive field, while the latter captures abundant frequency variations and textural features via fractional Gabor transformer. To adaptively fuse features from space and frequency aspects, a parameter-free feature selection module is proposed to ensure that the most representative and distinctive information are preserved. With SFS-Conv, we build a lightweight SAR object detection network, called SFS-CNet. Experimental results show that SFS-CNet outperforms state-of-the-art (SoTA) models on a series of SAR object detection benchmarks, while simultaneously reducing both the model size and computational cost.

## SFS-Conv Framework
![SFS-Conv](https://github.com/like413/SFS-Conv/blob/main/fig/SFS-Conv.png)
The proposed SFS-Conv module is specially designed for SAR object detection, which consists of three units, a spatial perception unit (SPU), a frequency perception unit (FPU), and a channel selection unit (CSU).
### FPU for Frequency Analysis
<img src="https://github.com/like413/SFS-Conv/blob/main/fig/FPU.png" alt="FPU" width="500">
In SAR scenes, objects typically exhibit characteristics across multiple scales and orientations, making traditional convolution kernels potentially inflexible in feature extraction and leading to feature redundancy. For example, in a scenario where an object undergoes rotation, features extracted by traditional convolution kernels may struggle to adapt to the new direction, necessitating more kernels to cover features in different directions. This transformation guides convolution kernels in extracting high-frequency texture features at multiple scales and orientations, effectively suppressing speckle noise in SAR images.‘

### SPU for Spatial Context Representation
<img src="https://github.com/like413/SFS-Conv/blob/main/fig/SPU.png" alt="SPU" width="500">
Since SAR images are typically captured at high resolutions from an overhead perspective, which is challenging to identify objects based on appearance alone. Instead, surrounding environment of objects can offer valuable cues for recognition, such as object shape, orientation, and other characteristics. Specifically, we first partition the feature map channels, and then adopt multiple kernels of different sizes to obtain multi-scale features. In addition, we construct hierarchical residual connections between kernels to further expand the receptive field for each convolution layer.

## Performance Comparison
### Comparison with the SOTA methods for SFS-CNet on HRSID
![HRSID](https://github.com/like413/SFS-Conv/blob/main/fig/HRSID.png)
Comparison of SFS-CNet and SoTA methods on **HRSID** data set. The best and second best performance are highlighted in **bold** and _underline_. `$\dagger$` represents using the OGL strategy.

### Comparison with the SOTA methods for SFS-CNet on SAR-AIRcraft-1.0
![SAR-AIRcraft-1.0](https://github.com/like413/SFS-Conv/blob/main/fig/SAR-AIRcraft-1.0.png)
Comparison of SFS-CNet and SoTA methods on **SAR-AIRcraft-1.0** data set. The best and second best performance are highlighted in **bold** and _underline_. `$\dagger$` represents using the OGL strategy.

## Citation
If you found this code useful, please cite the paper. Welcome 👍Fork and Star👍, then I will let you know when we update.

```
@InProceedings{Li_2024_CVPR,
    author    = {Li, Ke and Wang, Di and Hu, Zhangyuan and Zhu, Wenxuan and Li, Shaofeng and Wang, Quan},
    title     = {Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {17323-17332}
}
```
