# ZeRO
**在数据并行中，介绍了DP和DDP，两者总通讯量相同，但是DP存在负载不均的情况，大部分通讯压力集中在Server上，而Server的通讯量与GPU数量呈线性关系，导致DP一般适用于单机多卡场景。而DDP通过采用Ring-AllReduce这一操作，使得通讯量均衡分布在每块GPU上，且该通讯量为一固定常量，不受GPU个数影响，因此可以实现跨机器的训练**  
**DDP解决了通讯负载不均的问题，但还遗留了显存开销问题，数据并行中，每个GPU上都复制了一份完整模型，但当模型变大时，很容易OOM，所以Zero（零冗余优化）的提出就是解决这个问题，思想就是用通讯换显存**
## 一、存储消耗
### 1.1 存储分类
大模型在训练时，需要存储如下内容：
![image](https://github.com/user-attachments/assets/9a106437-6880-4b1f-ae9f-c5c67f84e29c)
存储主要分为两大块，Model States和Residual States  
**Model States**指的是和模型本身息息相关的，必须存储的内容，具体包括：  
· optimizer states：Adam优化算法中的momentum和variance  
· gradients：模型梯度  
· parameters：模型参数W  

**Residual States**指并非模型必须的，但在训练过程中会额外产生的内容，具体包括：  
· activation：激活值，在流水线并行中，在backward过程中适用链式法则计算梯度是会用到，有了它计算梯度会更快，但它不是必须存储的，因为可以重新Forward来算它【神经网络中通常会保存激活值和激活函数前的值，用于链式法则梯度计算】  
· temporary buffers：临时存储，例如把梯度发送到某块GPU上做总聚合时产生的存储  
· unusable fragment memory：碎片化的存储空间，虽然总存储空间是够的，但是娶不到连续的存储空间，相关的请求也会被fail掉，对这类空间浪费可以通过内存整理来解决。
### 1.2 混合精度训练  
提出目的：对于模型，我们希望其参数越精准越好，也即fp32【占用4字节】来表示参数w，但是在forward和backward过程中，fp32的计算开销也算庞大的，混合精度训练的提出就是为了在计算的过程中，引入fp16或bf16【半精度浮点数，存储占用2字节】来减轻压力。如下图  
<div align=center>
  <img src="https://github.com/user-attachments/assets/3dc95ad1-432a-4d8f-9549-1fab4a5b0a09" width="800" />
</div>

流程如下：  
· 存储一份fp32的parameter，momentum和variance（统称model states）  
· 在forward开始之前，额外开辟一块存储空间，将fp32 parameter减半到fp16 parameter。  
· 正常做forward和backward，在此之间产生的activation和gradients，都用fp16进行存储。  
· 用fp16 gradients去更新fp32下的model states。  
· 当模型收敛后，fp32的parameter就是最终的参数输出。
## 二、ZeRO-DP
在整个训练过程中，很多states并不会每时每刻都用到，比如：（1）Adam优化下的optimizer states只在最终做update时才用到；（2）数据并行中，gradients只在最后做AllReduce和update时才用到；（3）参数W只在做forward和backward时才会用到。  
Zero的思路是如果数据用完即废，需要的时候再从什么地方拿回来
### 2.1 Pos：优化状态分割
首先从optimizer states开始优化，将opetimizer states分成若干份，每块GPU上各自维护一份，这样就减少了相当一部分的显存开销【Adam算法会为每个参数准备一个momentum和variance，显存会增长2倍。】，如下图：
<div align=center>
  <img src="https://github.com/user-attachments/assets/47bc2077-20f0-4558-8b9d-14e0c8be3b8d" width="800" />
</div>
图中W=fp16，代表fp16的参数矩阵；G=fp16，代表梯度。O=fp32，代表fp32的参数矩阵和momentum和variance  

流程如下：  
（1）每块GPU上存一份完整的W，将一个batch的数据分成3份，每块GPU各吃一份  
（2）对梯度做一次AllReduce【求和，广播】，得到完整的梯度G，产生单卡通讯量2Z。此时所有GPU上参数矩阵一致。  
（3）得到完整梯度G，就可以对W做更新，由于W的更新由optimier states和梯度共同决定。由于每块GPU上只保管部分optimizer states，因此只能将相应的W（蓝色部分）进行更新，如下图：
<div align=center>
  <img src="https://github.com/user-attachments/assets/fad870e3-7854-440c-b674-56a3396dfdef" width="800" />
</div>
