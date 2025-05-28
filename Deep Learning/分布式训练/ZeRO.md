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
（2）对梯度做一次AllReduce【求和/求平均后，广播】，得到完整的梯度G，产生单卡通讯量2Z。在此之前所有GPU上参数矩阵一致，此时获得了完整梯度。  
（3）得到完整梯度G，就可以对W做更新，由于W的更新由optimier states和梯度共同决定。由于每块GPU上只保管部分optimizer states，因此只能将相应的W（蓝色部分）进行更新，如下图：
<div align=center>
  <img src="https://github.com/user-attachments/assets/fad870e3-7854-440c-b674-56a3396dfdef" width="800" />
</div>

（4）此时，每块GPU上都有部分W没有完成更新。所以我们需要对W做一次All-Gather【拼接多个GPU的中间表示】，从别的GPU上把更新好的部分W取回来，产生单卡通讯量Z  
做完Pos后，设GPU个数为N，模型参数大小为Z，那么根据混合精度训练的定义，可知内存开销为2Z+2Z+KZ【2Z分别是fp16的模型参数和梯度，KZ指的是32fp的模型参数和momentum和variance，因为不一定使用Adam算法，所以这里用KZ代替，如果使用Adam算法，那么K=12】  

  那么做完Pos后，显存和通讯量的对比如下：  
  · 朴素DP：单卡显存占用2Z+2Z+KZ【转换为字节后的实际大小】，单卡通讯量为2Z【梯度+模型参数，未转换为字节】。  
  · Pos：单卡显存占用2Z+2Z+(K/N)Z【转换为字节后的实际大小】，单卡通讯量为3Z【梯度+模型参数[发送梯度，接收模型参数]，拼接梯度。未转换为字节】  
  当K=12，Z=7.5B，N=64时，可计算得到DP占用显存120GB，Pos占用显存31.4GB  
### 2.2 Pos + Pg：优化状态与梯度分割
现在，更近一步，我们把梯度也拆开。每个GPU格子维护一块梯度。
![image](https://github.com/user-attachments/assets/f9179b00-5604-4a72-90e1-6fb87e961fac)

此时，数据并行的整体流程如下：  
（1）每块GPU上存一份完整的参数W，将一个batch的数据分成3份，每块GPU各吃一份，做完一轮FWD和BWD后，算的一份完整的梯度，如上图G1,G2,G3  
（2）对梯度做一次Reduce-Scatter【规约操作，即加和操作】，保证每个GPU上维持的那块梯度时聚合梯度。例如对GPU1，它负责维护G1，因此其他的GPU只需要把G1对应位置的梯度发给GPU1做加总就可。汇总完毕后，白色块对GPU无用，可以从显存中移除。单卡通讯量Z。具体见下图表示  
![image](https://github.com/user-attachments/assets/711545bb-bc6e-4b5c-99f0-ef4c9146983d)
（3）每块GPU用自己的O和G取更新相应的W，更新完毕后，每块GPU维持了一块更新完毕的W，同理对W做一次All-Gather，将别的GPU算好的W同步到自己这来，单卡通讯量为Z【同2.1最后一步】  
显存和通讯量如下：  
· Pos+Pg：单卡显存占用2Z + (2+K)/N·Z【计算的为移除梯度后的显存】，单卡通讯量为2Z。当K=12，Z=7.5B，N=64时，可计算得到占用显存16.6GB  
### 2.3 Pos + Pg + Pp：优化状态、梯度与参数分割  
现在我们除了把 O【optimizer states】,G【gradients】切开，也可以把W【parameters】切开。如下图
![image](https://github.com/user-attachments/assets/f6ba6a4f-000a-4253-86e4-1700a999c229)

数据并行的流程如下：  
（1）每块GPU上只保存部分参数W，将一个batch的数据分成3份，每个GPU各吃一份。  
（2）做FWD时，对W做一次All-Gather，取回分布在别的GPU上的W，得到一份完整的W，单卡通讯量Z，forward做完，立刻把不是自己维护的W抛弃。  
（3）做BWD时，对W做一次All-Gather，取回完整的W，单卡通讯量Z，BWD做完后，立刻把不是自己维护的W抛弃。  
（4）做完BWD，算的一份完整的梯度，对G做一次Reduce-Scatter，从别的GPU上聚合自己维护的那部分梯度，单卡通讯量Z，聚合操作后，立刻把不是自己维护的G抛弃  
（5）用自己维护的O和G，更新W。由于只维护部分W，因此无需再对W做任何AllReduce操作。
显存和通讯量如下：  
· Pos+Pg+Pp：单卡显存占用(2+2+K)/N·Z【计算的为移除梯度/参数矩阵后的显存，实际字节占用数量】，单卡通讯量为3Z。当K=12，Z=7.5B，N=64时，可计算得到占用显存1.9GB  
### 2.4 ZeRO和模型并行的区别  
既然ZeRO都把参数W给切了，那它应该是个模型并行呀？为什么要归到数据并行里呢？其实ZeRO是模型并行的形式，数据并行的实质。
模型并行，是指在forward和backward的过程中，我只需要用自己维护的那块W来计算就行。即同样的输入X，每块GPU上各算模型的一部分，最后通过某些方式聚合结果。
但对ZeRO来说，它做forward和backward的时候，是需要把各GPU上维护的W聚合起来的，即本质上还是用完整的W进行计算。它是不同的输入X，完整的参数W，最终再做聚合。
