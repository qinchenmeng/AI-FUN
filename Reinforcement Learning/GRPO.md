# GRPO
## 1、概述
GRPO优化算法概述如下  
GRPO优化过程仅需要加载两个模型，一个是目标优化模型，一个是ref模型。由于有规则reward的判别，并不需要reward model。（如果判别是由奖励模型，那么仍需要，比如用GRPO做RLHF需要peference reward model）
![image](https://github.com/user-attachments/assets/55ca4e65-6777-4f37-8409-3ce5378081ef)
![image](https://github.com/user-attachments/assets/da0bb7dd-0772-48d7-8f65-5a1569320dbd)
## 2、流程  
（1）对于原始的模型，针对单个问题q，生成多个不同的回答oi,i=1,2...G,G为组数，每个回答oi有不同的长度|oi|  
（2）对多个不同回答，先流经Ref模型，用于计算kl散度；后流经reward模型，用于计算该组回答的reward，ri  
（3）根据该组ri，根据均值方差公式，得到组内奖励均值方差，随后得到组内各oi相对优势Ai,t，t代表一个回答的每个token优势都是一致的。  
（4）有了优势后，就可以根据优势和kl散度，优化原始模型  
## 3、奖励函数和优势  
（1）主要有两种形式
