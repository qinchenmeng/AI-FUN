# 1、强化学习概述
## 1.1 流程
在强化学习框架中，一般拥有两个实体，智能体Agent和环境Environment，强化学习中两个实体的交互中有几个概念：状态空间S：指环境中所有可能状态的集合；动作空间A：指智能体所有可能动作的集合；奖励R:指智能体在环境的某一状态下所获得的奖励   

  一般情况下，智能体与环境的交互过程如下：  
  · 在t时刻，环境的状态为St，达到这一状态所获得的奖励为Rt  
  · 智能体观测到St和Rt后，采取相应的动作At  
  · 智能体采取At后，环境状态变为St+1，得到的奖励为Rt+1  

  智能体在这个过程中学习，最终目标是：**找到一个策略，这个策略根据当前观测的环境状态和奖励反馈来选择最佳的动作**
## 1.2 价值函数  
在强化学习中，价值函数衡量的是**从某个状态（或状态-动作）出发，未来能获得的“预期总回报”有多大**  

  具体而言 t时刻状态s的总收益 = 身处状态s能带来的即时收益 + 从状态s出发后能带来的未来收益  
  写成表达式为：Vt = Rt + βVt+1  
  其中Vt代表t时刻的总收益，Rt代表t时刻即时收益，Vt+1代表t+1时刻的总收益，β代表折扣因子，决定在多大程度上考虑将“未来收益”纳入到“当下收益”  
# 2、在NLP中如何理解上述角色定义 
在NLP中，t时刻，模型根据上文产生一个token，这个token即对应着强化学习中的动作，记为At，此时模型产出token At对应着的即时收益为Rt（按1.1应为Rt+1，为方便记为Rt），总收益为Vt，此时模型的状态由St变为St+1，也就是从“上文”变成“上文+新产出的token”  
# 3、PPO中的四个模型
## 3.1 Actor Model
即要训练的目标语言模型，一般用SFT阶段产出的SFT模型来初始化。最终目的是让Actor模型产生能符合人类喜欢的response。  
策略为喂给Actor一条prompt，生成对应的response，然后将“prompt+response”送入奖励-loss体系计算loss更新Actor模型  
## 3.2 Reference Model
一般也用SFT阶段得到的SFT模型做初始化，训练过程中，参数冻结，作用是防止模型训歪，具体方法是使用KL散度，具体公式见下图：
<div align=center>
  <img src="https://github.com/user-attachments/assets/2f26803b-d508-4a6e-a1a1-67ae65e53fb9" width="500" />
</div>
KL散度衡量的是从Q走到P要花多少额外的信息代价，DKL(P||Q) != DKL(Q||P)，并且非负仅当两者相同时等号成立。  

对于Actor Model输出response时，对每个生成的token都会有对应的概率，同样的对于Ref模型，将Actor生成的prompt+response也喂给它，同样能给出每个token的prob结果  


    这里需要说明的是：在Ref生成概率的时候采取的为教师强制的方法，即Ref模型不输出新的token，因为它已经有了Actor模型给的response答案，把答案的token作为decoder的输入，去预测一个得到一个概率，但不根据这个概率产生新的输出token并作为下一次的输入，而是继续采用答案token作为输入，即通过“知道答案”去预测答案，看它自己本来的倾向有多大。  

在KL散度方面有 DKL(Actor||Ref) = Ex~actor(x)*[log(actor(x)/ref(x))]，这个值越小意味着两个分布的相似性越高
## 3.3 Critic Model  
用于预测期望总收益Vt，即1.2提到的价值函数，为强化学习的训练目标，需要同Actor一样做参数更新，在模型结构上有RW阶段的RM模型初始化而来，同RM模型去掉原本decoder的linear+softmax，重新接了一个线性层输出Vt。  

  · 到此为止，我们从RW阶段得到RM输出的分数，是一个样本集的奖励，即一条prompt+response；而我们需要每个token的具体反馈，才能完成价值函数  
  · Policy Gradient策略梯度需要优势，而这个值我们只能通过Critic来估算，即在t时刻，给不出客观存在的总收益Vt，只能通过训练一个模型来预测    
  
critic的本质目标是是在每一个状态st下预测，如果从这里继续生成，将来能得到的总奖励是多少。这个「总奖励」，在 RLHF 中 通常简化为整个 response 得到的 reward（因为我们只有句子级 reward）。  
举例：在实际训练时，我们会把response一次性生成出来，假设是一个长度为T的token序列，那么可以构造T个状态：  
- s1 = prompt + token1
- s2 = prompt + token1 + token2
- ...
然后在每个st上都用 critic输出一个值Vt，代表**从这一点往后走，能拿到多少reward**，由于我们只有整段段reward，所以我们简单把这个R当作每个st的监督信号（一种方法，最常见也是最简单的方法，把整句的reward R直接作为所有step的目标值。）。
## 3.4 Reward Model  
具体介绍见RW.md，不过这里需要注意的一点是，在RLHF的强化学习阶段，它的参数被冻结，仅用于为完整的response打分，而非提供每个token的即时收益。

# 4、Loss计算
## 4.1 Actor loss
（1）直观设计  
Actor接收上文St，产生token At，即输出概率P(At|St)，Critic模型根据St，At，产生对总收益的预测Vt。那么Actor_loss = - ∑Vt*logP(At|St)  
需要最小化这个损失，直观理解上，Vt>0，证明critic给了当前采取动作的正反馈，因此提高P(At|St)，从而减小loss；Vt<0，意味着Critic对Actor给了负反馈，所以要降低P(At｜St)，从而达到减小loss的作用。  
（2）引入优势  
对于NLP任务来说，如果Critic对At对总收益预测为Vt，但实际执行At后的总收益是Rt + βVt+1 ，那么可以定义优势为Advt = Rt + βVt+1 - Vt 【GPT认为Advt=R-Vt,而非后面介绍的动归】 
用优势来替换掉原来预测的总收益，则此刻Actor_loss = -∑Advt * logP(At|St)   
（3）重新设计Rt（非R）  【这一节属于人工设计奖励的范畴，据GPT所言，标准PPO中，认为每一个step的reward都是R】
按照上述的理解，Rt应该表示每个Actor产出token At带来的即时收益，然而并非如此，在deepspeed的RLHF实践中，对Rt做了另一种设计，具体见下图
<div align=center>
  <img src="https://github.com/user-attachments/assets/d6a12075-030f-4dc7-8ce2-f0bf27235f61" width="500" />
</div>
其中kl_ctl为一个控制比例的缩放因子，而-log(..)为KL散度计算公式，目的是防止训歪。    
所以综上，当t != T时，即非最后一个token时，Rt更加关心Actor有没有在Ref的约束下生产tokenAt，而当t = T时，不仅关心是否遵从了Ref的约束，也关心真正的奖励R（此时Rt就是上面提到的R）
（4）重新设计优势  
我们在引入优势的时候，替换了总收益Vt，而根据Vt的表达式可知，Vt在设计上考虑了未来的总收益，所以同理我们也可以这样改造优势，对于优势而言，也要考量未来的优势   

  从而可得Advt = (Rt + βVt+1 - Vt) + θ*γ*Advt+1 ，其中γ是一个常量，即权衡因子。  
  那么如何计算Advt+1,可以发现最后一个时刻，未来收益Vt+1和未来优势Advt+1都为0，从而可得AdvT = RT-VT，T代表最后一个时刻，从而可以继续从后往前，把所有时刻的优势都计算出来 
 
（5）PPO-epoch：引入新约束  
总结目前的训练流程：  
- 第一步：准备一个batch的prompts
- 第二步：将这个batch的prompts喂给Actor模型，生成对应的responses
- 第三步：将prompt+responses喂给Critic/Reward/Reference模型，生成用于计算actor/critic loss的数据，强化学习中称这些数据为经验
- 第四步：根据loss更新Actor和Critic模型

  计算Actor_loss的步骤:首先Reward模型会给出一个最终的奖励信号R，而Reference可以计算每个token的prob，从而根据这些信息，可以计算出来瞬时奖励Rt；同样的Critic模型会预测每个时刻的Vt，拿最后一个时刻T的VT用于计算最后一个时刻的优势AdvT,从而根据动态规划往前计算出来每个时刻的优势Advt，有了Advt，自然我们就可以计算一个句子的Actor_loss

然而一个batch的经验值被用于n次模型更新：在强化学习中，收集一个batch的经验非常耗费时间，对应PPO，收集一次经验，需要等四个模型都推理完才可以，而这样却仅更新一次loss  
而如果我们想让一个batch的经验值被重复使用ppo_epochs次，等价于我们想要Actor在这个过程中，模拟和环境交互ppo_epochs次。 
其实就是更新后的Actor能模仿最开始的Actor，从而达成模拟的效果  
<div align=center>
  <img src="https://github.com/user-attachments/assets/a32c7773-0774-4c13-92be-b95d5cb01fa1" width="500" />
</div>  
其中Pold代表真正吃了batch的经验的Actor，而P表示ppo_epochs实时迭代更新的Actor，这个公式也可以理解为：在Actor想通过模拟交互的方式，使用一个batch的经验值更新自己时，它需要收到真正吃到batch的那个时刻的Actor的约束，这样才能在有效利用batch，提升训练速度的基础上，保持训练的稳定。另外相比之前的Actor_loss丢弃了log，设计到了重要性采样的内容  
同时对 P(At/St)/Pold(At/St)，设置一个范围，比如(0.8，1.2)，超过这个范围就裁剪到范围边缘，从而保证参数不进行突变，而一旦超过这个范围后，认为这部分loss无关于Actor模型，停止更新，综上，此时actor_loss设计如下图：  
<div align=center>
  <img src="https://github.com/user-attachments/assets/812a6848-84f0-49bc-b0c1-65a705f3c4dc" width="500" />
</div>  
（6）小结  

- 我们已经对Rt进行来改造，使其能够衡量Actor模型是否遵从了Ref模型的约束  
- 我们已经对Advt进行改造，使其不仅考虑了当前时刻的优势，还考虑了未来的优势
- 我们重复利用了1个batch的数据，使本来只能被用来做1次模型更新的它现在能被用来做ppo_epochs次模型更新。我们使用真正吃了batch，产出经验值的那个时刻的Actor分布来约束ppo_epochs中更新的Actor分布  
- 我们考虑了剪裁机制（clip），在ppo_epochs次更新中，一旦Actor的更新幅度超过我们的控制范围，则不对它进行参数更新。
## 4.2 Critic loss  
critic模型会预测每个时间t的总收益Vt，那么他的标签是什么呢？前面有提到Vt = Rt + βVt+1 ，Vt为预测值，而Rt + βVt+1显然是比Vt更接近t时刻真值总收益多一个值  
所以第一想法为critic_loss = (Rt + βVt+1 - Vt)^2  
（1）实际收益优化  
由于我们之前引入了优势，优势更加丰富的刻画了实时收益，所以实际收益可以优化为：Advt + Vt  
（2）预估收益优化
原始的预估收益为Vt,类比于Actor模型，Critic模型在ppo_epochs也是不断更新的，所以Vt可以理解为Critic old即真正吃了经验的critci模型产生的收益预测  
预估收益，随着ppo_epochs的进行不断优化而变动。  

  总结：实际收益（Advt，Vt）都是老Critic模型产生的结果，而预估收益是随着ppo_epochs进行变动，两者进行MSE计算取loss来优化模型即可
