# Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection
This repository is the official implementation of CVPR 2024 "Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection" at: [CVPR Open Access]([https://drive.google.com/drive/folders/1e_wOtkruWAB2JXR7aqaMZMrM75IkjqCA?usp=drive_link](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Unleashing_Channel_Potential_Space-Frequency_Selection_Convolution_for_SAR_Object_Detection_CVPR_2024_paper.html)). 

## Abstract
Deep Convolutional Neural Networks (DCNNs) have achieved remarkable performance in synthetic aperture radar (SAR) object detection, but this comes at the cost of tremendous computational resources, partly due to extracting redundant features within a single convolutional layer. Recent works either delve into model compression methods or focus on the carefully-designed lightweight models, both of which result in performance degradation. In this paper, we propose an efficient convolution module for SAR object detection, called SFS-Conv, which increases feature diversity within each convolutional layer through a shuntperceive-select strategy. Specifically, we shunt input feature maps into space and frequency aspects. The former perceives the context of various objects by dynamically adjusting receptive field, while the latter captures abundant frequency variations and textural features via fractional Gabor transformer. To adaptively fuse features from space and frequency aspects, a parameter-free feature selection module is proposed to ensure that the most representative and distinctive information are preserved. With SFS-Conv, we build a lightweight SAR object detection network, called SFS-CNet. Experimental results show that SFS-CNet outperforms state-of-the-art (SoTA) models on a series of SAR object detection benchmarks, while simultaneously reducing both the model size and computational cost.

## SFS-Conv Framework
![SFS-Conv](https://github.com/like413/SFS-Conv/blob/main/fig/SFS-Conv.png)
The proposed SFS-Conv module is specially designed for SAR object detection, which consists of three units, a spatial perception unit (SPU), a frequency perception unit (FPU), and a channel selection unit (CSU).


## Performance Comparison
### Comparison with the SOTA methods for LPVA on HRSID

| Methods                                      | Params. (M) | FLOPs (G) | Infer. Time (ms/img) | $R$  | $P$  | $AP_{50}$ | $AP_{75}$ | $F1$ | $R$  | $P$  | $AP_{50}$ | $AP_{75}$ | $F1$ | $R$  | $P$  | $AP_{50}$ | $AP_{75}$ | $F1$ |
|----------------------------------------------|-------------|-----------|----------------------|------|------|-----------|-----------|------|------|------|-----------|-----------|------|------|------|-----------|-----------|------|
| Faster R-CNN \cite{ren2015faster}            | 41.30       | 138.5     | 18.0                 | 81.5 | 86.1 | 86.5      | 73.6      | 83.7 | 95.4 | 96.8 | 97.7      | 94.4      | 96.1 | 71.1 | 77.4 | 78.4      | 58.4      | 74.1 |
| Mask R-CNN \cite{he2017mask}                 | 43.99       | 188.4     | 24.9                 | 82.3 | 87.1 | 87.9      | 75.0      | 84.6 | 96.2 | 97.3 | 98.3      | 94.1      | 96.7 | 71.7 | 78.2 | 79.2      | 60.8      | 74.8 |
| Cascade R-CNN \cite{cai2018cascade}          | 69.15       | 159.2     | 23.1                 | 81.8 | 86.2 | 86.6      | 76.8      | 83.9 | 95.6 | 96.8 | 98.1      | 95.3      | 96.2 | 72.4 | 78.8 | 79.5      | 69.3      | 75.5 |
| Cascade Mask R-CNN \cite{cai2018cascade}     | 77.05       | -         | -                    | 82.6 | 88.0 | 88.3      | 77.2      | 85.2 | 96.3 | 97.4 | 98.5      | 95.3      | 96.8 | 73.2 | 79.3 | 80.0      | 64.7      | 76.1 |
| HTC \cite{chen2019hybrid}                    | 80.02       | -         | -                    | 82.4 | 86.9 | 87.6      | 78.5      | 84.6 | _96.9_ | 97.4 | 98.7      | **96.1** | 97.1 | 74.8 | 81.0 | 82.1      | **71.2** | 77.8 |
| HRSDNet \cite{wei2020hrsid}                  | 91.03       | -         | -                    | -    | -    | 89.3      | 79.8      | -    | -    | -    | 98.6      | _96.0_    | -    | -    | -    | 81.3      | 68.3      | -    |
| RetinaNet \cite{lin2017focal}                | 36.25       | 139.2     | 16.4                 | 77.2 | 82.9 | 83.7      | 66.5      | 79.9 | 95.6 | 96.2 | 97.6      | 92.5      | 95.9 | 61.5 | 69.6 | 69.3      | 42.7      | 65.3 |
| FCOS \cite{tian2019fcos}                     | 32.11       | 123.3     | 17.4                 | 76.8 | 80.1 | 80.7      | 57.3      | 78.4 | 95.8 | 96.7 | 97.8      | 87.5      | 96.2 | 57.3 | 64.3 | 64.5      | 33.7      | 60.6 |
| CenterNet \cite{duan2019centernet}           | 32.13       | 123.0     | 18.4                 | 82.0 | 86.4 | 87.0      | 64.9      | 84.1 | 95.9 | 96.7 | 97.9      | 90.6      | 96.3 | 71.3 | 76.7 | 77.6      | 46.0      | 73.9 |
| YOLO-FA \cite{zhang2023frequency}            | 6.86        | -         | 14.7                 | 87.6 | 93.1 | 93.5      | -         | 90.3 | -    | -    | -         | -         | -    | -    | -    | -         | -         | -    |
| YOLOv3 \cite{redmon2018yolov3}               | 61.50       | 77.4      | 13.2                 | 84.6 | 91.0 | 91.9      | 74.0      | 87.7 | 96.1 | 97.3 | 98.2      | 75.6      | 96.7 | 79.0 | 68.5 | 75.7      | 21.0      | 73.4 |
| YOLOX-Nano \cite{ge2021yolox}                | 0.90        | 0.5       | 15.9                 | 72.8 | 79.4 | 80.1      | 54.8      | 76.0 | 94.3 | 95.7 | 96.9      | 87.4      | 95.0 | 72.5 | 79.9 | 80.1      | 26.7      | 76.0 |
| YOLOv5n \cite{Jocher_YOLOv5_2020}            | 1.92        | 4.5       | 6.3                  | 84.3 | 90.7 | 91.4      | 70.4      | 87.4 | 96.0 | 97.5 | 98.2      | 80.5      | 96.7 | 77.9 | 68.6 | 75.7      | 47.1      | 73.9 |
| YOLOv5s \cite{Jocher_YOLOv5_2020}            | 7.21        | 16.5      | 6.4                  | 89.3 | 94.2 | _95.4_    | 83.3      | 91.7 | **97.2** | 98.1 | 98.9      | 93.3      | **97.6** | **79.9** | 85.5 | _86.9_    | 65.7      | _82.6_ |
| YOLOv8n \cite{YOLO_by_Ultralytics_2023}      | 3.01        | 8.9       | 12.2                 | 86.9 | 93.0 | 93.7      | 80.2      | 89.8 | 96.2 | 97.8 | 98.8      | 94.2      | 97.0 | 72.6 | 83.6 | 80.3      | 57.1      | 77.7 |
| YOLOv8s \cite{YOLO_by_Ultralytics_2023}      | 10.65       | 28.4      | 14.1                 | _90.8_ | _95.0_ | **96.2** | **87.2** | _92.9_ | 96.6 | **98.6** | **99.2** | _96.0_    | **97.6** | _79.1_ | _88.1_ | 87.3      | _70.2_    | **83.3** |
| ROI Transformer \cite{ding2019learning}      | 55.03       | -         | 50.5                 | 84.0 | -    | 79.7      | 49.4      | -    | 94.7 | 97.4 | 90.7      | -         | 96.0 | 

### Comparison with the SOTA methods for LPVA on SAR-AIRcraft-1.0

| Methods                                      | mAP  | $AP_{50}$ A330 | $AP_{75}$ A330 | $AP_{50}$ A320/321 | $AP_{75}$ A320/321 | $AP_{50}$ A220 | $AP_{75}$ A220 | $AP_{50}$ ARJ21 | $AP_{75}$ ARJ21 | $AP_{50}$ Boeing737 | $AP_{75}$ Boeing737 | $AP_{50}$ Boeing787 | $AP_{75}$ Boeing787 | $AP_{50}$ other | $AP_{75}$ other |
|----------------------------------------------|------|----------------|----------------|--------------------|--------------------|----------------|----------------|-----------------|-----------------|----------------------|----------------------|----------------------|----------------------|----------------|----------------|
| Faster R-CNN \cite{ren2015faster}            | 76.1 | 85.0           | 85.0           | 97.2               | 87.7               | 78.5           | 58.7           | 74.0            | 55.2            | 55.1                 | 42.8                 | 72.9                 | 60.5                 | 70.1           | 45.4           |
| Cascade R-CNN \cite{cai2018cascade}          | 75.7 | 87.4           | 87.4           | 97.5               | 73.9               | 74.0           | 49.1           | 78.0            | 59.0            | 54.5                 | 39.1                 | 68.3                 | 57.6                 | 69.1           | 46.1           |
| RepPoints \cite{yang2019reppoints}           | 72.6 | 89.8           | 66.4           | 97.9               | 84.9               | 71.4           | 49.4           | 73.0            | 50.9            | 55.7                 | 36.6                 | 51.8                 | 41.8                 | 68.4           | 43.1           |
| SkG-Net \cite{fu2021scattering}              | 70.7 | 79.3           | 66.4           | 78.2               | 49.6               | 66.4           | 29.8           | 65.0            | 37.7            | 65.1                 | 48.7                 | 69.6                 | 51.6                 | 71.4           | 41.4           |
| SA-Net \cite{wzr2023sar}                     | 77.7 | 88.6           | 88.6           | 94.3               | 86.6               | _90.3_         | 55.0           | 78.6            | 59.7            | 59.7                 | 41.8                 | 70.8                 | 60.4                 | 71.3           | 47.7           |
| RetinaNet \cite{lin2017focal}                | 72.3 | 92.0           | 70.1           | 92.6               | 58.4               | 73.0           | 41.7           | 63.2            | 47.1            | 47.8                 | 25.3                 | 65.4                 | 50.0                 | 67.0           | 42.3           |
| FCOS \cite{tian2019fcos}                     | 55.2 | 30.8           | 29.5           | 65.6               | 64.5               | 60.2           | 33.0           | 57.6            | 35.5            | 41.9                 | 20.2                 | 46.8                 | 34.3                 | 62.6           | 33.0           |
| CenterNet \cite{duan2019centernet}           | 71.1 | 91.4           | 69.3           | 92.3               | 64.4               | 70.5           | 44.0           | 64.6            | 45.6            | 47.3                 | 26.4                 | 65.9                 | 49.7                 | 66.1           | 41.0           |
| YOLOv3 \cite{redmon2018yolov3}               | 83.9 | 91.8           | 90.8           | 96.9               | **97.0**           | 86.5           | **69.6**       | 77.5            | 61.4            | 77.0                 | 52.6                 | 76.4                 | 65.8                 | 82.4           | 57.2           |
| YOLOX-Nano \cite{ge2021yolox}                | 81.3 | _95.6_         | 74.7           | 96.9               | 74.8               | 79.7           | 45.3           | 78.7            | 39.5            | 66.6                 | 39.7                 | 78.2                 | 51.1                 | 73.8           | 37.7           |
| YOLOv5n \cite{Jocher_YOLOv5_2020}            | 88.2 | 88.2           | 83.3           | _98.9_             | 68.2               | 84.6           | 52.5           | 86.6            | 56.1            | 75.0                 | 69.3                 | 95.2                 | 77.6                 | 84.7           | 54.4           |
| YOLOv5s \cite{Jocher_YOLOv5_2020}            | 89.0 | 92.1           | 92.1           | _98.9_             | 73.1               | 87.4           | _60.7_         | 86.4            | 56.9            | 76.3                 | 70.2                 | **96.2**             | **86.7**             | 85.1           | 59.0           |
| YOLOv8n \cite{YOLO_by_Ultralytics_2023}      | 88.4 | 93.1           | 92.0           | 97.2               | 73.1               | 85.6           | 56.3           | 86.1            | **66.1**        | 74.7                 | 70.5                 | 91.1                 | 82.6                 | 83.1           | 58.1           |
| YOLOv8s \cite{YOLO_by_Ultralytics_2023}      | _89.6_ | 95.0           | _95.2_         | 97.7               | _88.5_             | **95.8**       | 60.2           | 86.6            | _65.0_          | **78.9**             | **74.2**             | 90.9                 | 81.8                 | 84.4           | _59.6_         |
| SFS-CNet **(ours)**                          | 88.7 | 91.4           | 86.2           | 97.6               | 73.9               | 87.6           | 58.8           | **87.7**        | 60.9            | 77.8                 | _71.6_               | 92.4                 | 83.6                 | **86.6**       | **60.8**       |
| SFS-CNet $\dagger $ **(ours)**               | **89.7** | **95.9**     | **95.9**       | **99.3**           | 74.0               | 87.9           | 59.8           | _86.7_          | 61.3            | _77.9_               | 69.3                 | _92.9_               | _86.3_               | _85.6_         | _59.6_         |

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
