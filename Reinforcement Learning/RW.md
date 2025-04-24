## 奖励模型
奖励模型作为RLHF中的核心部分之一，作用是替人类“打分”，在Chatgpt中，奖励模型通过人工标注的排序序列来进行训练，具体如下图所示  
<div align=center>
  <img src="https://github.com/user-attachments/assets/fcdd1bcd-136c-4e6a-8cc3-18ea8a737f1e" width="500" />
</div>

  在标注过程中，Chatgpt并不是直接让人工去标注每一句话的真实得分是多少，尽管模型最终的输出就是每句话的得分，而是让人对几句话按照好坏程度进行排序。  
  通过这个【排序序列】，模型将会学习如何为每一个句子打分。  
  
