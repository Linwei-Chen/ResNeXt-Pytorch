本文结构：
1. **ResNeXt 论文原文 CIFAR 部分解读**
2. **模型具体参数**
3. **训练过程**
4. **训练结果**

**项目地址：**[GitHberChen/ResNeXt-Pytorch](https://link.zhihu.com/?target=https%3A//github.com/GitHberChen/ResNeXt-Pytorch)

## 一、ResNeXt 论文原文 CIFAR 部分解读

首先贴 上ResNeXt 论文中关于CIFAR 上实验的相关段落：

**5.3. Experiments on CIFAR**

We conduct more experiments on CIFAR-10 and 100 datasets [23]. We use the architectures as in [14] and replace the basic residual block by the bottleneck template of 1×1, 64、3×3, 64、1×1, 256. Our networks start with a single 3×3 conv layer, followed by 3 stages each having 3 residual blocks, and end with average pooling and a fully-connected classifier (total 29-layer deep), following [14]. We adopt the same translation and flipping data augmentation as [14]. Implementation details are in the appendix.

We compare two cases of increasing complexity based on the above baseline: (i) increase cardinality and fix all widths, or (ii) increase width of the bottleneck and fix cardinality = 1. We train and evaluate a series of networks under these changes. Fig. 7 shows the comparisons of test error rates vs. model sizes. We find that increasing cardinality is more effective than increasing width, consistent to what we have observed on ImageNet-1K. Table 7 shows the results and model sizes, comparing with the Wide ResNet [43] which is the best published record. Our model with a similar model size (34.4M) shows results better than Wide ResNet. Our larger method achieves 3.58% test error (average of 10 runs) on CIFAR-10 and 17.31% on CIFAR-100. To the best of our knowledge, these are the state-of-the-art results (with similar data augmentation) in the literature including unpublished technical reports.

![img](https://pic3.zhimg.com/80/v2-c137bebe130600fc164de7bf68280952_hd.jpg)

**A. Implementation Details: CIFAR**

We train the models on the 50k training set and evaluate on the 10k test set. The input image is 32×32 randomly cropped from a zero-padded 40×40 image or its flipping, following [14]. No other data augmentation is used. The first layer is 3×3 conv with 64 filters. There are 3 stages each having 3 residual blocks, and the output map size is 32, 16, and 8 for each stage [14]. The network ends with a global average pooling and a fully-connected layer. Width is increased by 2× when the stage changes (downsampling), as in Sec. 3.1. The models are trained on 8 GPUs with a mini-batch size of 128, with a weight decay of 0.0005 and a momentum of 0.9. We start with a learning rate of 0.1 and train the models for 300 epochs, reducing the learning rate at the 150-th and 225-th epoch. Other implementation details are as in [11].



主要意思有以下几点：

1. 模型采用 1×1, 64、3×3, 64、1×1, 256 形式的bottleneck模板，一个3×3初始卷积层、3 个 stage，每个 stage 3个bottleneck，加上最后一个全连接层，一共29层，经过每个 stage 后的下采样层数特征层数翻倍。
2. 最佳结果为16x64d（d 表示分组卷积中单个分组中的卷积数量，16表示共有16组）模型，十次测试平均成绩为3.58% test error on CIFAR-10，17.31% on CIFAR-100。
3. 训练集为50k，测试集为10k。
4. 数据增强采用了将 32x32 的图片通过 0-padding 的方式变成 40x40然后随机裁剪的方式，另外还使用了随机左右镜像翻转，测试时使用原图进行测试。
5. 使用SGD，decay = 0.0005，momentum = 0.9，batch_size = 128
6. 初始学习率为 0.1，第150、225个epoch时以gamma = 0.1 调整学习率（即调整为0.01、0.001）。



## 二、模型具体参数：

根据原论文的意思，设计出 ResNeXt-29， C=16，D=64，即在第一个stage中每个 Residual Block 中的 3x3 卷积分组为 16，每组 64 层，每 max-pooling 一次，stage + 1，卷积层数翻倍：

```python3
CifarResNeXt(
  (conv_1_3x3): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn_1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (stage_1): Sequential(
    (stage_1_bottleneck_0): ResNeXtBottleneck(
      (conv_reduce): Conv2d(64, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_reduce): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_expand): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_expand): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(
        (shortcut_conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (shortcut_bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stage_1_bottleneck_1): ResNeXtBottleneck(
      (conv_reduce): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_reduce): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_expand): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_expand): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
    (stage_1_bottleneck_2): ResNeXtBottleneck(
      (conv_reduce): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_reduce): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_expand): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_expand): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  (stage_2): Sequential(
    (stage_2_bottleneck_0): ResNeXtBottleneck(
      (conv_reduce): Conv2d(256, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_reduce): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_conv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
      (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_expand): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_expand): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(
        (shortcut_conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (shortcut_bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stage_2_bottleneck_1): ResNeXtBottleneck(
      (conv_reduce): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_reduce): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_conv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
      (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_expand): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_expand): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
    (stage_2_bottleneck_2): ResNeXtBottleneck(
      (conv_reduce): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_reduce): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_conv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
      (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_expand): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_expand): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  (stage_3): Sequential(
    (stage_3_bottleneck_0): ResNeXtBottleneck(
      (conv_reduce): Conv2d(512, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_reduce): BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_conv): Conv2d(4096, 4096, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
      (bn): BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_expand): Conv2d(4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_expand): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(
        (shortcut_conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (shortcut_bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stage_3_bottleneck_1): ResNeXtBottleneck(
      (conv_reduce): Conv2d(1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_reduce): BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_conv): Conv2d(4096, 4096, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
      (bn): BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_expand): Conv2d(4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_expand): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
    (stage_3_bottleneck_2): ResNeXtBottleneck(
      (conv_reduce): Conv2d(1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_reduce): BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_conv): Conv2d(4096, 4096, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
      (bn): BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_expand): Conv2d(4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn_expand): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  (classifier): Linear(in_features=1024, out_features=10, bias=True)
)
```



## 三、训练过程

![img](https://pic3.zhimg.com/80/v2-267acb7f72aa20a1f3f57abbbee34312_hd.jpg)

原论文中训练了 300 epochs，在 CIFAR-10 上达到了96.42% 的准确度，本次实现训练了 280 epochs 左右，高精度 94.64%，相对于论文的96.42% 低了 1.78%，由于需要计算资源去做其他的事情，所以中断了训练。继续训练、减小 lr 进一步训练以及采用更大的 batch size （然而我的卡放不下...）应该可以达到更高的精度。



训练过程中我采用了两种优化优化器：

1. SGD：lr = 0.1，momentum = 0.9，decay = 0.0005
2. Adam：lr = 0.001， 其它参数为 Pytorch 默认值 ( *betas=(0.9*,*0.999)*,*eps=1e-08*,*weight_decay=0*,*amsgrad=False*)



训练过程的具体调节如下：

1. 前 70个epochs 使用 batch_size =32，lr = 0.1，SGD ，训练时的测试准确度维持 70% 左右，没有明显的上升趋势。
2. 第 70~180 使用 batch_size = 48，lr = 0.1，SGD，训练时的测试准度提高到了 75%，左右，往后也没有明显的上升趋势。
3. 第 180~250 使用 batch_size = 48，lr = 0.001， Adam，精度立马提升到 90%+。
4. 第 250~280 使用 batch_size = 48，lr = 0.0001， Adam，精度进一步有 1%+ 的明显提升。



总结一句， Adam 大法好！在训练过程中尤其是初、中期，Adam 对模型精度提升可谓是立竿见影，另外用 SGD 真的挺需要耐心的，不过也有论文指出 Adam 会使模型错过最优解，所以最推荐的方式还是：前期使用 Adam 快速收敛，后期用 SGD 慢慢磨以达到最佳的精度。

参考链接：[Juliuszh：Adam那么棒，为什么还对SGD念念不忘 (2)—— Adam的两宗罪](https://zhuanlan.zhihu.com/p/32262540)





PS：

广告时间啦~

理工狗不想被人文素养拖后腿？不妨关注微信公众号：

![img](https://pic1.zhimg.com/80/c5c764858514dcb402114e619d98f6dc_hd.jpg)

欢迎扫码关注~