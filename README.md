# AttFC: Attention Fully-Connected Layer for Large-Scale Face Recognition with One GPU

Nowadays, with the advancement of deep neural networks (DNNs) and the availability of large-scale datasets, the face recognition (FR) model has achieved exceptional performance. However, since the parameter magnitude of the fully connected (FC) layer directly depends on the number of identities in the dataset. If training the FR model on large-scale datasets, the size of the model parameter will be excessively huge, leading to substantial demand for computational resources, such as time and memory. This paper proposes the attention fully connected (AttFC) layer, which could significantly reduce computational resources. AttFC employs an attention loader to generate the generative class center (GCC), and dynamically store the class center with Dynamic Class Container (DCC). DCC only stores a small subset of all class centers in FC, thus its parameter count is substantially less than the FC layer. Also, training face recognition models on large-scale datasets with one GPU often encounter out-of-memory (OOM) issues.

## Requirements

The code of AttFC is based on Pytorch V2.2.0, please use the following link to install the appropriate version.

- Install [PyTorch](https://pytorch.org/get-started/previous-versions/) V2.2.0.

## Datasets
  - [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface)
  - [WebFace42M](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/docs/prepare_webface42m.md)
  - [IJB-B,IJB-C](https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view?usp=sharing)

  
## Training
To train a model, execute the `train.py` script with the path to the configuration files. The sample commands provided below demonstrate the process of conducting distributed training.
```shell
python train.py
```

Note:   
It is not recommended to use a single GPU for training, as this may result in longer training times and suboptimal performance. For best results, we suggest using multiple GPUs or a GPU cluster.  





## Model Zoo

- The models are available for non-commercial research purposes only.  
- All models can be found in here.  
- [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g): e8pw  
- [OneDrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)

### Performance on IJB-C and [**ICCV2021-MFR**](https://github.com/deepinsight/insightface/blob/master/challenges/mfr/README.md)

ICCV2021-MFR testset consists of non-celebrities so we can ensure that it has very few overlap with public available face 
recognition training set, such as MS1M and CASIA as they mostly collected from online celebrities. 
As the result, we can evaluate the FAIR performance for different algorithms.  

For **ICCV2021-MFR-ALL** set, TAR is measured on all-to-all 1:1 protocal, with FAR less than 0.000001(e-6). The 
globalised multi-racial testset contains 242,143 identities and 1,624,305 images. 


#### 1. Training on Single-Host GPU

| Datasets       | Backbone            | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | log                                                                                                                                 |
|:---------------|:--------------------|:------------|:------------|:------------|:------------------------------------------------------------------------------------------------------------------------------------|
| MS1MV2         | mobilefacenet-0.45G | 62.07       | 93.61       | 90.28       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_mbf/training.log)                     |
| MS1MV2         | r50                 | 75.13       | 95.97       | 94.07       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_r50/training.log)                     |
| MS1MV2         | r100                | 78.12       | 96.37       | 94.27       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_r100/training.log)                    |
| MS1MV3         | mobilefacenet-0.45G | 63.78       | 94.23       | 91.33       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_mbf/training.log)                     |
| MS1MV3         | r50                 | 79.14       | 96.37       | 94.47       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r50/training.log)                     |
| MS1MV3         | r100                | 81.97       | 96.85       | 95.02       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r100/training.log)                    |
| Glint360K      | mobilefacenet-0.45G | 70.18       | 95.04       | 92.62       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_mbf/training.log)                  |
| Glint360K      | r50                 | 86.34       | 97.16       | 95.81       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r50/training.log)                  |
| Glint360k      | r100                | 89.52       | 97.55       | 96.38       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r100/training.log)                 |
| WF4M           | r100                | 89.87       | 97.19       | 95.48       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf4m_r100/training.log)                      |
| WF12M-PFC-0.2  | r100                | 94.75       | 97.60       | 95.90       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_pfc02_r100/training.log)               |
| WF12M-PFC-0.3  | r100                | 94.71       | 97.64       | 96.01       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_pfc03_r100/training.log)               |
| WF12M          | r100                | 94.69       | 97.59       | 95.97       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_r100/training.log)                     |
| WF42M-PFC-0.2  | r100                | 96.27       | 97.70       | 96.31       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf42m_pfc02_r100/training.log)               |
| WF42M-PFC-0.2  | ViT-T-1.5G          | 92.04       | 97.27       | 95.68       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf42m_pfc02_40epoch_8gpu_vit_t/training.log) |
| WF42M-PFC-0.3  | ViT-B-11G           | 97.16       | 97.91       | 97.05       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_b_8gpu/training.log)         |

#### 2. Training on Multi-Host GPU

| Datasets         | Backbone(bs*gpus) | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | Throughout | log                                                                                                                                        |
|:-----------------|:------------------|:------------|:------------|:------------|:-----------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| WF42M-PFC-0.2    | r50(512*8)        | 93.83       | 97.53       | 96.16       | ~5900      | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_bs4k_pfc02/training.log)             |
| WF42M-PFC-0.2    | r50(512*16)       | 93.96       | 97.46       | 96.12       | ~11000     | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_lr01_pfc02_bs8k_16gpus/training.log) |
| WF42M-PFC-0.2    | r50(128*32)       | 94.04       | 97.48       | 95.94       | ~17000     | click me                                                                                                                                   |
| WF42M-PFC-0.2    | r100(128*16)      | 96.28       | 97.80       | 96.57       | ~5200      | click me                                                                                                                                   |
| WF42M-PFC-0.2    | r100(256*16)      | 96.69       | 97.85       | 96.63       | ~5200      | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r100_bs4k_pfc02/training.log)            |
| WF42M-PFC-0.0018 | r100(512*32)      | 93.08       | 97.51       | 95.88       | ~10000     | click me                                                                                                                                   |
| WF42M-PFC-0.2    | r100(128*32)      | 96.57       | 97.83       | 96.50       | ~9800      | click me                                                                                                                                   |

`r100(128*32)` means backbone is r100, batchsize per gpu is 128, the number of gpus is 32.



#### 3. ViT For Face Recognition

| Datasets      | Backbone(bs)  | FLOPs | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | Throughout | log                                                                                                                          |
|:--------------|:--------------|:------|:------------|:------------|:------------|:-----------|:-----------------------------------------------------------------------------------------------------------------------------|
| WF42M-PFC-0.3 | r18(128*32)   | 2.6   | 79.13       | 95.77       | 93.36       | -          | click me                                                                                                                     |
| WF42M-PFC-0.3 | r50(128*32)   | 6.3   | 94.03       | 97.48       | 95.94       | -          | click me                                                                                                                     |
| WF42M-PFC-0.3 | r100(128*32)  | 12.1  | 96.69       | 97.82       | 96.45       | -          | click me                                                                                                                     |
| WF42M-PFC-0.3 | r200(128*32)  | 23.5  | 97.70       | 97.97       | 96.93       | -          | click me                                                                                                                     |
| WF42M-PFC-0.3 | VIT-T(384*64) | 1.5   | 92.24       | 97.31       | 95.97       | ~35000     | click me                                                                                                                     |
| WF42M-PFC-0.3 | VIT-S(384*64) | 5.7   | 95.87       | 97.73       | 96.57       | ~25000     | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_s_64gpu/training.log) |
| WF42M-PFC-0.3 | VIT-B(384*64) | 11.4  | 97.42       | 97.90       | 97.04       | ~13800     | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_b_64gpu/training.log) |
| WF42M-PFC-0.3 | VIT-L(384*64) | 25.3  | 97.85       | 98.00       | 97.23       | ~9406      | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_l_64gpu/training.log) |

`WF42M` means WebFace42M, `PFC-0.3` means negivate class centers sample rate is 0.3.

#### 4. Noisy Datasets
  
| Datasets                 | Backbone | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | log      |
|:-------------------------|:---------|:------------|:------------|:------------|:---------|
| WF12M-Flip(40%)          | r50      | 43.87       | 88.35       | 80.78       | click me |
| WF12M-Flip(40%)-PFC-0.1* | r50      | 80.20       | 96.11       | 93.79       | click me |
| WF12M-Conflict           | r50      | 79.93       | 95.30       | 91.56       | click me |
| WF12M-Conflict-PFC-0.3*  | r50      | 91.68       | 97.28       | 95.75       | click me |

`WF12M` means WebFace12M, `+PFC-0.1*` denotes additional abnormal inter-class filtering.



## Speed Benchmark
<div><img src="https://github.com/anxiangsir/insightface_arcface_log/blob/master/pfc_exp.png" width = "90%" /></div>


**Arcface-Torch** is an efficient tool for training large-scale face recognition training sets. When the number of classes in the training sets exceeds one million, the partial FC sampling strategy maintains the same accuracy while providing several times faster training performance and lower GPU memory utilization. The partial FC is a sparse variant of the model parallel architecture for large-scale face recognition, utilizing a sparse softmax that dynamically samples a subset of class centers for each training batch. During each iteration, only a sparse portion of the parameters are updated, leading to a significant reduction in GPU memory requirements and computational demands. With the partial FC approach, it is possible to train sets with up to 29 million identities, the largest to date. Furthermore, the partial FC method supports multi-machine distributed training and mixed precision training.



More details see 
[speed_benchmark.md](docs/speed_benchmark.md) in docs.

> 1. Training Speed of Various Parallel Techniques (Samples per Second) on a Tesla V100 32GB x 8 System (Higher is Optimal)

`-` means training failed because of gpu memory limitations.

| Number of Identities in Dataset | Data Parallel | Model Parallel | Partial FC 0.1 |
|:--------------------------------|:--------------|:---------------|:---------------|
| 125000                          | 4681          | 4824           | 5004           |
| 1400000                         | **1672**      | 3043           | 4738           |
| 5500000                         | **-**         | **1389**       | 3975           |
| 8000000                         | **-**         | **-**          | 3565           |
| 16000000                        | **-**         | **-**          | 2679           |
| 29000000                        | **-**         | **-**          | **1855**       |

> 2. GPU Memory Utilization of Various Parallel Techniques (MB per GPU) on a Tesla V100 32GB x 8 System (Lower is Optimal)

| Number of Identities in Dataset | Data Parallel | Model Parallel | Partial FC 0.1 |
|:--------------------------------|:--------------|:---------------|:---------------|
| 125000                          | 7358          | 5306           | 4868           |
| 1400000                         | 32252         | 11178          | 6056           |
| 5500000                         | **-**         | 32188          | 9854           |
| 8000000                         | **-**         | **-**          | 12310          |
| 16000000                        | **-**         | **-**          | 19950          |
| 29000000                        | **-**         | **-**          | 32324          |


## Citations

```
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}
@inproceedings{An_2022_CVPR,
    author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
    title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2022},
    pages={4042-4051}
}
@inproceedings{zhu2021webface260m,
  title={Webface260m: A benchmark unveiling the power of million-scale deep face recognition},
  author={Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10492--10502},
  year={2021}
}
```


## Welcome!  
<a href='https://mapmyvisitors.com/web/1bw5e'  title='Visit tracker'><img src='https://mapmyvisitors.com/map.png?cl=ffffff&w=1024&t=n&d=0mqj5JJrL2-BR6EVSskbTRFBlGgSbqZK9ZJg6g_vh74&co=2d78ad&ct=ffffff'/></a>