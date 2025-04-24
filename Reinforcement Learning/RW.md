# 奖励模型
奖励模型作为RLHF中的核心部分之一，作用是替人类“打分”，在Chatgpt中，奖励模型通过人工标注的排序序列来进行训练，具体如下图所示  
<div align=center>
  <img src="https://github.com/user-attachments/assets/fcdd1bcd-136c-4e6a-8cc3-18ea8a737f1e" width="500" />
</div>

  在标注过程中，Chatgpt并不是直接让人工去标注每一句话的真实得分是多少，尽管模型最终的输出就是每句话的得分，而是让人对几句话按照好坏程度进行排序。  
  通过这个【排序序列】，模型将会学习如何为每一个句子打分。  
# 模型结构
  在模型结构上，奖励模型的基础结构为一个预训练语言模型+一个线性头。和GPT一样的decoder流程，但是最后不需要用linear+softmax生成词的概率，而是直接接一个linear，输出一个reward标量分数  
  在输入上，只拿到最后生成的token的分数即可，因为这个token的hidden state已经看到整个回答，包含了前面句子的上下文信息，只需要取一个hidden state而不是聚合所有token，还能节省计算和参数亮
  
# 【标注排序序列】替代【直接打分】  
原因：打**绝对分数**较难统一，转换为一个**相对排序**的任务相对更加简单  
# Rank Loss --通过排序序列学会打分  
假设现在有一个排好的回答序列：A>B>C>D，我们现在需要训练一个打分模型，模型给四句话打出来的分要满足r(A)>r(B)>r(C)>r(D)  
那么可以使用如下损失函数：  
<div align=center>
  <img src="https://github.com/user-attachments/assets/93df596e-6aa3-4008-9dcc-ccccbb7c1e3b" width="500" />
</div>  
其中，yw代表排序在yl的所有句子，K在上述例子中为4，而E期望那一部分代表从数据分布中采样所有 “更好-更差” 对，在本例子中共6对。  
先构造所有成对的「更好 vs 更差」样本对（记住排序一定不要搞反，前面的更好，后面的更差），再计算每对的打分差，然后喂给 sigmoid → log → 平均，即可。loss取负数，如果差值很大，那么sigmoid趋近1，则loss趋于0；如果差值为负数，那么sigmoid趋近0，则log函数趋近于负∞，再取一个负号，则loss损失会变得很大。
