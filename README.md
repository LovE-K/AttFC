# AttFC: Attention Fully-Connected Layer for Large-Scale Face Recognition with One GPU

Nowadays, with the advancement of deep neural networks (DNNs) and the availability of large-scale datasets, the face recognition (FR) model has achieved exceptional performance. However, since the parameter magnitude of the fully connected (FC) layer directly depends on the number of identities in the dataset. If training the FR model on large-scale datasets, the size of the model parameter will be excessively huge, leading to substantial demand for computational resources, such as time and memory. We propose the attention fully connected (AttFC) layer, which could significantly reduce computational resources. AttFC employs an attention loader to generate the generative class center (GCC), and dynamically store the class center with Dynamic Class Container (DCC). DCC only stores a small subset of all class centers in FC, thus its parameter count is substantially less than the FC layer. Also, training face recognition models on large-scale datasets with one GPU often encounter out-of-memory (OOM) issues.

## Requirements

The code of AttFC is based on Pytorch V2.2.0, please use the following link to install the appropriate version.

- Install [PyTorch](https://pytorch.org/get-started/previous-versions/) V2.2.0.

## Datasets
  - [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface)
  - [WebFace42M](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/docs/prepare_webface42m.md)
  - [IJB-B,IJB-C](https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view?usp=sharing)

  
## Training
Modifying the configuration file `configs/att_cfg.py`, then executing the `train.py`.
```shell
python train.py
```

## Evaluation
Executing the evaluation script in `eval/` to test the model.
```shell
python eval/eval_veri.py
python eval/eval_ijbb.py
python eval/eval_ijbc.py
```
