# TSMF-Net

Language: [English](README.md)，[简体中文](README_zh.md)

![](extra/TSMF.webp)



# 引用

```
@article{liao2022tsmf,
  title={A Two-Stage Mutual Fusion Network for Multispectral and Panchromatic Image Classification},
  author={Liao, Yinuo and Zhu, Hao and Jiao, Licheng and Li, Xiaotong and Li, Na and Sun, Kenan and Tang, Xu and Hou, Biao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--18},
  year={2022},
  publisher={IEEE}
}
```



# 环境

| Env/Package | Version  | Env/Package | Version |
| :---------: | :------: | :---------: | :-----: |
|   python    |  3.6.10  |   libtiff   |  0.4.2  |
|    cuda     |   10.1   |    numpy    | 1.19.2  |
|    torch    |  1.3.1   |   pillow    |  8.0.1  |
| torchvision |  0.4.2   |    scipy    |  1.5.4  |
|   opencv    | 4.4.0.46 | hdf5storage | 0.1.18  |
|    gdal     |  3.0.2   |    h5py     |  3.1.0  |

通过`requirements.txt`或`jianchao.yaml`创建环境，两者都在`extra`文件夹中



# 预处理

## 1) get_vec.py

**输入:** `msf.tif`和`pan.tif`

**说明:** `get_vec.py`对`msf`进行2倍上采样，形状$(H,W,4)\to(2H,2W,4)$；对`pan`进行了`2-split`操作，形状$(4H,4W,1)\to(2H,2W,4)$，此时两者形状相同

接着调用`to_tensor()`函数对两者进行归一化，数据类型`float32`，数据范围`[0,1]`

最后将两者展平，形状$(2H,2W,4)\to(2H\times2W,4)$

**输出:** `msf.mat`和`pan.mat`



## 2) get_para.m

**输入:** `msf.mat`和`pan.mat`

**说明:** 设`msf`的权重参数为$\alpha_i\quad(i=1,2,3,4)$，`pan`的权重参数为$\beta_i\quad(i=1,2,3,4)$，求解以下凸优化问题：


$$
\mathop{\min}\limits_{\alpha_i,\beta_i} \Vert \sum_{i=1}^4 \alpha_i M_i - \sum_{i=1}^4  \beta_i P_i \Vert_2^2
$$

$$
s.t.  \alpha_i,\beta_i>0, \sum_{i=1}^4  \beta_i=1
$$

**输出:** 屏幕上打印 运行时间`sj`，权重参数`para`（即$\alpha_i$和$\beta_i$)，最小值`val`

**注意:** 此MATLAB脚本依赖于函数`icanfast.m`，请不要删除或随意移动



## 3) img_fusion.py

**输入:** `msf.tif`，`pan.tif`，$\alpha_i$和$\beta_i$

**说明:** 详见原文ATIHS阐述部分

**输出:** `msf_f.npy`和`pan_f.npy`

**注意:** $\alpha_i$和$\beta_i$需要在代码`112`行左右手动修改



# 训练&测试

### train.py

**输入:** `msf_f.npy`、`pan_f.npy`和`label.mat`

**说明:** 手动选CUDA索引，四张卡分别对应0，1，2，3；训练测试一体

**输出:** 以`AA` 命名的`.pkl`模型



# 可视化

### draw.py

**输入:** `msf_f.npy`、`pan_f.npy`和`label.mat`

**说明:** 手动选CUDA索引、半图全图，半图0全图1

**输出:** 半图`xx_half.png`，全图`xx_full.png`

<img src="viz/10_half.webp" width="360"><img src="viz/10_full.webp" width="360">

