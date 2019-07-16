# ResNeXt-Pytorch

首先贴 上ResNeXt 论文中关于CIFAR 上实验的相关段落：

## 5.3. Experiments on CIFAR

We conduct more experiments on CIFAR-10 and 100 datasets [23]. We use the architectures as in [14] and replace the basic residual block by the bottleneck template of  1×1, 64、3×3, 64、1×1, 256。Our networks start with a single 3×3 conv layer, followed by 3 stages each having 3 residual blocks, and end with average pooling and a fully-connected classifier (total 29-layer deep), following [14]. We adopt the same translation and flipping data augmentation as [14]. Implementation details are in the appendix.

We compare two cases of increasing complexity based on the above baseline: (i) increase cardinality and fix all widths, or (ii) increase width of the bottleneck and fix cardinality = 1. We train and evaluate a series of networks under these changes. Fig. 7 shows the comparisons of test error rates vs. model sizes. We find that increasing cardinality is more effective than increasing width, consistent to what we have observed on ImageNet-1K. Table 7 shows the results and model sizes, comparing with the Wide ResNet [43] which is the best published record. Our model with a similar model size (34.4M) shows results better than Wide ResNet. Our larger method achieves 3.58% test error (average of 10 runs) on CIFAR-10 and 17.31% on CIFAR-100. To the best of our knowledge, these are the state-of-the-art results (with similar data augmentation) in the literature including unpublished technical reports.

![img](https://pic3.zhimg.com/v2-c137bebe130600fc164de7bf68280952_b.jpeg)

## A. Implementation Details: CIFAR

We train the models on the 50k training set and evaluate on the 10k test set. The input image is 32×32 randomly cropped from a zero-padded 40×40 image or its flipping, following [14]. No other data augmentation is used. The first layer is 3×3 conv with 64 filters. There are 3 stages each having 3 residual blocks, and the output map size is 32, 16, and 8 for each stage [14]. The network ends with a global average pooling and a fully-connected layer. Width is increased by 2× when the stage changes (downsampling), as in Sec. 3.1. The models are trained on 8 GPUs with a mini-batch size of 128, with a weight decay of 0.0005 and a momentum of 0.9. We start with a learning rate of 0.1 and train the models for 300 epochs, reducing the learning rate at the 150-th and 225-th epoch. Other implementation details are as in [11].



主要意思有以下几点：

1. 模型采用 1×1, 64、3×3, 64、1×1, 256 形式的bottleneck模板，一个3×3初始卷积层、3 个 stage，每个 stage 3个bottleneck，加上最后一个全连接层，一共29层，经过每个 stage 后的下采样层数特征层数翻倍。
2. 最佳结果为16x64d（d 表示分组卷积中单个分组中的卷积数量，16表示共有16组）模型，十次测试平均成绩为3.58% test error on CIFAR-10，17.31% on CIFAR-100。
3. 训练集为50k，测试集为10k。
4. 数据增强采用了将 32x32 的图片通过 0-padding 的方式变成 40x40然后随机裁剪的方式，另外还使用了随机左右镜像翻转，测试时使用原图进行测试。
5. 使用SGD，decay = 0.0005，momentum = 0.9，batch_size = 128
6. 初始学习率为 0.1，第150、225个epoch时以gamma = 0.1 调整学习率（即0.01、0.001）。