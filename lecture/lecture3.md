## Transformer的架构
### 原始transformer vs 现代transformer
- 归一化的位置：
1. 传统的是post-layernorm(在残差相加之后：$x_{l+1} = \text{LN}(x_l + F(x_l))$)，因为layernorm对反向传播有削弱的作用，post-layernorm需要：$$\text{到达第 1 层的梯度} = \text{初始梯度} \times J_{\text{LN}_{100}} \times J_{\text{LN}_{99}} \times \dots \times J_{\text{LN}_1}$$，
2. 但是现代的不需要：$x_{100} = x_0 + F(\text{LN}(x_0)) + F(\text{LN}(x_1)) + \dots + F(\text{LN}(x_{99}))$，$$\frac{\partial x_{100}}{\partial x_0} = \mathbf{I} + \left( \text{各个支路的导数} \right)$$，现代的是在进入前先进行归一化之后再进行子网络层：$x_{l+1} = x_l + F(\text{LN}(x_l))$
3. 未来可能会使用double Norm的做法，$x_{l+1} = x_l + \mathbf{\text{LN}_{\text{post}}}(F(\text{LN}_{\text{pre}}(x_l)))$，这样能够避免一直叠加所导致的数值过大，同时因为没有包含输入本身，在进行反向传播的时候不需要连乘(和post-norm那种)
*** 残差网络的意思是在输入的基础上进行加工，而非重头计算的一种网络
- 位置编码：传统为直接使用token的绝对位置，难以有外推能力，一旦位置改变边难以变通，现代做法为使用相对位置
- 激活函数：传统ReLU的坏处为梯度硬裁切，会导致神经元死亡的问题，缺乏平滑度，现代使用swiGLU，使用门控线性单元控制来选择性筛选信息而不是一刀砍
- 偏移项：
1. 传统的模型是使用bias来解决平面强制过原点的问题，同时也有配合ReLU等激活函数的作用(多大才能被激活)，但是现在的模型抛弃了bias，因为模型的维度极高，加不加bias都无所谓了，而且抛弃bias后计算会只剩下矩阵乘法，能够契合GPU的使用
2. 由于不需要bias，数据的搬运变快，总体时间变快
- 串行与并行行
1. 串行：$y = x + \text{MLP}(\text{LN}(x + \text{Attn}(\text{LN}(x))))$
2. 并行行：$y = x + \text{MLP}(\text{LN}(x)) + \text{Attn}(\text{LN}(x))$，可以将attention和mlp数据矩阵进行融合，训练加快

## 训练技巧
- z-loss：为了防止数值膨胀问题，避免softmax饱和(因为模型容易整体往极大的正数漂移，哪怕相对的概率没有变化，但是结果会变得巨大)，增加惩罚项，如果变得太大就要求约束在1附近，$$L_z = \alpha \cdot \log^2 Z(x)$$，使得模型需要同时学习让参数别过大
- QK-Norm：在attention中，Query和Key的向量会越学越大，在经过softmax的时候会变为0/1，导致梯度消失，因此我们在对Q和K进行点积相乘前先分别套一层layernorm，这样也不会破坏核心的语义，维度之间的比例不变，而且attention本质是计算
- logit soft-capping：logit(未归一化的得分)的软截断(传统是使用硬阶段，超出的直接归零)，在计算出相关性得分后，有些logits会暴涨到很大的地方，截断的本质就是强行给这个得分设定一个安全边界，我们使用：$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$ 进行截断，无论x多大都会在-1和1之间，layernorm之类的是限制特征x在方差之中，但是logit是限制后面的结果

## 注意力的变化
- MHA (Multi-Head Attention)，表达能力最佳，但是显存占用高，因为每次生成新的token的时候都需要加载N分的KV矩阵
- MQA (Multi-Query Attention)，强制N个Q头共享同一份key/value头，这会导致变现能力下降但是显存占用较少
- GQA (Grouped-Query Attention)，折中方案，将N个Q分为G组