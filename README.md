HASIC-Net: Hybrid Attentional Convolutional Neural Network With Structure Information Consistency for Spectral Super-Resolution, TGRS, 2022.
==
[Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Songcheng Du](https://github.com/dusongcheng), [Rui song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN), [Chaoxiong Wu](https://scholar.google.com.hk/citations?user=PIsTkkEAAAAJ&hl=zh-CN&oi=ao), [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html), and [Qian Du](https://scholar.google.com/citations?user=0OdKQoQAAAAJ&hl=zh-CN).
***
Code for the paper: [HASIC-Net: Hybrid Attentional Convolutional Neural Network With Structure Information Consistency for Spectral Super-Resolution](https://ieeexplore.ieee.org/abstract/document/9678983).


<div align=center><img src="/Image/network.png" width="80%" height="80%"></div>
Fig. 1: Network architecture of our accurate hybrid attentional CNN with SIC.

Training and Test Process
--
1) Please prepare the training and test data as operated in the paper. 
2) Run "train_data_preprocess.py" and  "valid_data_preprocess.py" to prepare the data.
3) Run "train.py" to train the hasic-net.
4) Run "eval.py" to test.
5) Download the pretrained model ([Google Drive](https://drive.google.com/file/d/1shC3vEm2Xh7lnwrMj6zjTEHVdtXJVXC-/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1SrSXCuJJ-8aiijpiQglRvw?pwd=abcd ), code: `abcd`)).

References
--
If you find this code helpful, please kindly cite:

[1] J. Li, S. Du, R. Song, C. Wu, Y. Li and Q. Du, "HASIC-Net: Hybrid Attentional Convolutional Neural Network With Structure Information Consistency for Spectral Super-Resolution of RGB Images," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-15, 2022, Art no. 5522515, doi: 10.1109/TGRS.2022.3142258.
[2] Li J, Du S, Wu C, et al. DRCR Net: Dense Residual Channel Re-Calibration Network With Non-Local Purification for Spectral Super Resolution[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 1259-1268.

Citation Details
--
BibTeX entry:
```
@ARTICLE{9678983,
  author={Li, Jiaojiao and Du, Songcheng and Song, Rui and Wu, Chaoxiong and Li, Yunsong and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={HASIC-Net: Hybrid Attentional Convolutional Neural Network With Structure Information Consistency for Spectral Super-Resolution of RGB Images}, 
  year={2022},
  volume={60},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2022.3142258}}

@InProceedings{Li_2022_CVPR,
    author    = {Li, Jiaojiao and Du, Songcheng and Wu, Chaoxiong and Leng, Yihong and Song, Rui and Li, Yunsong},
    title     = {DRCR Net: Dense Residual Channel Re-Calibration Network With Non-Local Purification for Spectral Super Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1259-1268}
}
```

Licensing
--
Copyright (C) 2022 Songcheng Du

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
