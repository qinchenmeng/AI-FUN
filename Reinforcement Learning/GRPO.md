# GRPO
## 1、概述
GRPO优化算法概述如下  
GRPO优化过程仅需要加载两个模型，一个是目标优化模型，一个是ref模型。由于有规则reward的判别，并不需要reward model。（如果判别是由奖励模型，那么仍需要，比如用GRPO做RLHF需要peference reward model）
![image](https://github.com/user-attachments/assets/55ca4e65-6777-4f37-8409-3ce5378081ef)
![image](https://github.com/user-attachments/assets/da0bb7dd-0772-48d7-8f65-5a1569320dbd)
