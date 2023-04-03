![img](file:///C:\Users\14669\AppData\Local\Temp\ksohtml15848\wps1.jpg)





# <center>人工智能《人工智能导论》课程项目</center>

# <center>开题报告</center>















#### <font face="仿宋">项目题目：图像风格迁移<font face>

#### <font face="仿宋">所在学院：计算机学院<font face>

#### <font face="仿宋">项目组长：夏克宇<font face>

#### <font face="仿宋">小组成员：余庆标、林子静、李文娜<font face>

#### <font face="仿宋">开题时间：2023年4月1日<font face>















### <font face="黑体">一、选题背景<font face>

当今，计算机图像处理技术已经成为人们生活中不可或缺的一部分，尤其是数字媒体、互联网、游戏等领域。图像风格迁移（*Image Style Transfer*）技术是一种深度学习技术，能够将一张图片的风格应用到另一张图片上，从而生成一张新的图片，具有很高的实用价值。

图像风格迁移技术最早是由Gatys等人在2015年提出，该技术基于深度学习的神经网络，通过将两张图片输入网络进行训练，使神经网络能够学习到如何将两种不同的特征表示融合在一起。随着深度学习技术的不断发展，图像风格迁移技术也在不断地进行改进和优化，越来越受到人们的关注。

### <font face="黑体">二、相关研究综述<font face>

自2015年Gatys等人提出基于深度学习的图像风格迁移技术以来，这一领域的研究和应用不断扩展和深化。图像风格迁移技术已逐渐成为计算机视觉领域中备受关注的热点研究方向之一。

首先，基于神经网络的图像风格迁移技术不断得到改进和创新。一些研究者提出了基于残差神经网络的图像风格迁移技术，通过引入残差网络来加速图像风格的迁移过程。此外，一些研究者提出了基于对抗生成网络（GAN）的图像风格迁移技术，该技术利用生成器和判别器两个神经网络相互博弈的方式来实现图像风格的迁移。

#### 基于神经网络的风格转换（Neural style transfer）

##### 1.基于在线图像优化的慢速图像风格化迁移算法（Slow Neural Method Based On Online Image Optimisation）

###### 1.1**基于统计分布的参数化慢速风格化迁移算法（Parametric Slow Neural Method with Summary Statistics）**

这一类方法是基于在线图像优化的慢速图像重建方法和基于统计分布的参数化纹理建模方法结合而来的。 Gatys 在2015年提出的‘Texture Synthesis Using Convolutional Neural Networks’就是基于此方法的。如下图所示，输入是噪声底图，约束是Style Loss和Content Loss分别用来优化风格图像的风格和目标图像的内容。

![91c57b3d08a9424ead576860c156a1f9](C:\Users\14669\Desktop\91c57b3d08a9424ead576860c156a1f9.png)

简单说便是输入一张随机噪声构成的底图，通过计算Style Loss和Content Loss，迭代update底图，使其风格纹理上与Style Image相似，内容上与原照片相似。

具体说，论文用 Gram 矩阵来对图像中的风格进行建模和提取，再利用慢速图像重建方法，让重建后的图像以梯度下降的方式更新像素值，使其 Gram 矩阵接近风格图的 Gram 矩阵（即风格相似），然后，用VGG网络提取的高层feature map来表征图像的内容信息，通过使 VGG 网络对底图的提取的高层feature map接近目标图高层的feature map来达到内容相似，实际应用时候经常再加个总变分 TV 项来对结果进行平滑，最终重建出来的结果图就既拥有风格图的风格，又有内容图的内容。

###### 1.2基于MRF的非参数化慢速风格化迁移算法**（Non-parametric Slow Neural Method with MRFs）

其核心思想是提出了一个取代 Gram 损失的新的 MRF 损失。思路与传统的 MRF 非参数化纹理建模方法相似，即先将风格图和重建风格化结果图分成若干 patch，然后对于每个重建结果图中的 patch，去寻找并逼近与其最接近的风格 patch。

与传统 MRF 建模方法不同之处在于，以上操作是在 CNN 特征空间中完成的。另外还需要加一个祖师爷 Gatys 提出的内容损失来保证不丢失内容图中的高层语义信息。

这种基于 patch 的风格建模方法相比较以往基于统计分布的方法的一个明显优势在于，当风格图不是一幅艺术画作，而是和内容图内容相近的一张摄影照片（Photorealistic Style），这种基于 patch 匹配（patch matching）的方式可以很好地保留图像中的局部结构等信息。

##### 2.基于离线模型优化的快速图像风格化迁移算法（Fast Neural Method Based On Offline Model Optimisation）

核心思想就是利用基于离线模型优化的快速图像重建方法来节省时间。具体来说就是预训练一个前向网络，使得图像经过一次前向计算就可以得到图像重建结果，在依据各自约束来优化这一结果。根据一个训练好的前向网络能够学习到多少个风格作为分类依据，这里可以将这一类算法再细分为单模型单风格（PSPM）、单模型多风格（MSPM）和单模型任意风格（ASPM）的快速风格化迁移算法。

###### 2.1 **PSPM的快速风格迁移（Per-Style-Per-Model Fast Neural Method）**

主要想法是针对每一风格图，我们去特定的训练处一个前向网络，这样当测试的时候，我们只需要向模型中输入一张图，经过一次前向计算就可以得到输出结果。这种方法按照纹理建模的方式不同其实又可以分成基于统计分布的参数化纹理建模和基于MRF的非参数化纹理建模两类。

2.1.1 Perceptual Losses for Real-Time Style Transfer and Super-Resolution

![72cbc458d6da4f0c806c93904da911f9](C:\Users\14669\Desktop\72cbc458d6da4f0c806c93904da911f9.png)

与Gatys提出的模型相比，输入不再是噪声底图而是目标图像，并增加了一个autoencoder形状的前向网络来拟合风格转移的过程，其他思想相同：损失函数仍是用另一个神经网络（一般是VGG）提取的content loss+style loss，并统称perceptual loss，纹理建模使用Gram矩阵来表达。

2.1.2 基于 MRF 的快速 PSPM 风格化算法

###### 2.2**MSPM 的快速风格转移（Mutil-Style-Per-Model Fast Neural Method）**

核心思想为把风格化网络中间的几层单独拎出来（ StyleBank层），与每个风格进行绑定，对于每个新风格只去训练中间那几层，其余部分保持不变。

###### **2.3 ASPM快速风格化迁移算法（Arbitrary-Style-Per-Model Fast Neural Method）**

Zero-shot Fast Style Transfer 的单模型任意风格转换（ ASPM）的畅想最早由多伦路大学的天才博士陈天奇解决。他提出的 *Fast Patch-based Style Transfer of Arbitrary Style算法是基于 patch 的，可以归到基于 MRF 的非参数化 ASPM 算法。基本思想是在 CNN 特征空间中，找到与内容 patch 匹配的风格 patch 后，进行内容 patch 与风格 patch 的交换（Style Swap），之后用快速图像重建算法的思想对交换得到的 feature map 进行快速重建。

##### 3.**损失函数方面：**

损失函数有两部分组成：内容损失和风格损失

图片内容：图片的主体，图片中比较突出的部分的

图片风格：图片的纹理、色彩等

###### **3.1内容损失*constent lose*：原始图片的内容和生成图片的内容作欧式距离：**

$$
\mathscr{L}content(\vec{x},\vec{p},l)=\frac{1}{2}\sum_{i=0}^n(F^{l}_{ij}-P^{l}_{ij})
$$

其中，等式左侧表示在第l层中，原始图像*(P)*和生成图像*(F)*的举例，右侧是对应的最小二乘法表达式。*Fij*表示生成图像第 *i* 个*feature map*的第 *j* 个输出值。
$$
{\partial\mathscr{L}content(\vec{x},\vec{p},l)\over \partial F^{l}_{ij} }=

\begin{cases}
(F^{l}-P^{l})_{ij}\quad ifF^{l}_{ij}>0\\
0 \quad ifF^{l}_{ij}<0
\end{cases}
$$
使用最小二乘法求导得出最小值，再让改的l层上生成的图片*(F)*逼近改层的原始图片*(P)*

###### **3.2风格损失*style loss*使用类*G*矩阵代表图像的风格 ：**

$$
G^{l}_{ij}=\sum_{}^kF^{l}_{ik}F^{l}_{jk}
$$

当同一个维度上面的值相乘的时候原来越小相乘之后的值变得更小，原来越大相乘之后的中就变得越大；在不同维度上的关系也在相乘的表达当中表示出来。
$$
G=A^TA=
\left[
\begin{matrix}
a^T_1 \\
a^T_2 \\
...\\
a^T_n
\end{matrix}
\right]
[a_1 & a_2 &...& a_n]=
\left[
\begin{matrix}
a^T_1a_1&a^T_1a_2&...&a^T_1a_n\\
a^T_2a_1&a^T_2a_2&...&a^T_2a_n\\
&&...\\
a^T_na_1&a^T_na_2&...&a^T_na_n
\end{matrix}
\right]
$$


### <font face="黑体">三、拟解决的问题和研究内容<font face>

本次项目参考了多组已有的基于神经网络实现图像风格迁移的代码，其中主要的几组风格迁移代码是基于pytorch通过统计分布的参数化慢速风格化迁移算法（Parametric Slow Neural Method with Summary Statistics）实现图像风格迁移，在输入内容图片和风格图片后，先形成一张随机噪声构成的底图，或者一张与内容图片相同的输入图片，通过计算Style Loss和Content Loss，迭代update底图，使其风格纹理上与Style Image相似，内容上与原照片相似。

![](C:\Users\14669\AppData\Roaming\Typora\typora-user-images\image-20230402232245038.png)

<center>图3.1</center>

![image-20230402232331382](C:\Users\14669\AppData\Roaming\Typora\typora-user-images\image-20230402232331382.png)

<center>图3.2</center>

![image-20230402232530722](C:\Users\14669\AppData\Roaming\Typora\typora-user-images\image-20230402232530722.png)

<center>图3.3</center>

以上图片是我组目前参考实现的风格迁移的效果图，其中可以看到，不同的训练模型，不同的损失函数计算方式，不同的超参都会影响到图像风格迁移最终的拟合效果；即使使用相同的训练模型，在面对不同的内容图片和风格图片是，最终形成的效果图的拟合程度也是不尽相同的。

同时，项目目前使用的图像风格迁移的代码全部为基于神经网络的风格迁移。为呈现更好的拟合效果图，也有参考或者使用使用基于对抗生成网络的风格迁移以达到更好的拟合效果。

所以，项目主要的研究内容是，基于已有的风格迁移的项目代码。通过选用不同的训练模型，优化项目代码形成拟合效果较好的图像风格迁移项目代码。项目代码使用的深度学习框架不限。若在项目期间无法形成自己的图像风格迁移项目代码，拟通过已有的图像风格迁移代码，通过修改代码内容，改良已有代码，从而减少风格迁移算法实现过程中损失函数的值，从而达到更好的拟合效果。





### <font face="黑体">四、可行性分析<font face>

**技术可行性：**图像风格迁移技术基于深度学习的神经网络，深度学习技术的发展为图像风格迁移技术的实现提供了可行性保障。目前，深度学习技术已经在许多领域得到了广泛应用，如图像识别、自然语言处理等。

**数据可行性：**图像风格迁移技术的实现需要大量的图像数据用于训练神经网络。目前，随着数字图像技术的不断发展，图像数据的获取和处理变得越来越容易。同时，也有很多公开数据集可供使用，如COCO、ImageNet等，这些数据集的质量和数量可以满足图像风格迁移技术的需求。

**算法可行性：**图像风格迁移技术的算法已经得到了很多的改进和优化，能够适应不同的应用场景和需求。例如，基于残差神经网络的图像风格迁移技术可以加速图像风格的迁移过程；基于对抗生成网络（GAN）的图像风格迁移技术可以实现更加逼真的图像风格迁移效果。

### <font face="黑体">五、计划进程安排<font face>

1. **分析代码：**对图像风格迁移的代码进行全面的分析，包括代码结构、算法实现、变量命名等方面。分析的目的是为了找出代码中的优点和不足，为后续的优化工作提供基础。
2. **确定优化目标：**在分析代码的基础上，明确需要优化的目标，如提高代码的运行速度、减少内存占用、改善图像风格的迁移效果等。
3. **设计优化方案：**根据优化目标，设计优化方案，包括改进算法、优化代码结构、调整参数等方面。同时，要考虑到优化方案的可行性和效果，避免出现新的问题。
4. **实现优化方案：**根据设计的优化方案，对代码进行修改和调试。修改后，需要进行测试，确保代码的正确性和稳定性。
5. **性能评估：**对优化后的代码进行性能评估，包括运行速度、内存占用、图像风格迁移效果等方面。评估的目的是为了验证优化方案的有效性和优化效果。
6. **修正和改进：**根据性能评估的结果，及时修正和改进优化方案。如果评估结果不理想，需要重新设计优化方案，并进行实现和测试。
7. **文档记录：**对优化过程和结果进行文档记录，包括修改的代码、优化方案、性能评估结果等方面。文档记录的目的是为了方便后续的代码维护和优化。

### <font face="黑体">六、参考文献<font face>

[[1]StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation (arxiv.org)](https://arxiv.org/abs/1711.09020)

[[2]Neural Transfer Using PyTorch — PyTorch Tutorials 2.0.0+cu117 documentation](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

[[3]图像风格迁移_Arwin（Haowen Yu）的博客-CSDN博客](https://blog.csdn.net/qq_39297053/article/details/120453246)
