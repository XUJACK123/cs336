# embedding
- 所有词向量就是直接保存在 Embedding 模块内部的
- 想要什么直接在embedding内部提取

# linear
- 所有可学习的参数都在linear中

# 概率归一化和loss：
- softmax 是一种基于指数的非线性概率归一化函数
- 把任意实数范围的未归一化得分（logits）变为和为 1 的概率分布
## loss:
- $$\text{Loss}_{\text{total}} = -\log(P_{\text{total}}) = -\log(P_1 \cdot P_2 \cdots P_N) = (-\log P_1) + (-\log P_2) + \dots + (-\log P_N)$$
- 单个loss：$$\text{Loss} = -\log \left( \frac{e^{z_y}}{\sum_j e^{z_j}} \right)$$
    - 损失函数定义为概率的负对数：$\text{Loss} = -\log P(y)$，最后能够变为loss = log_sum_exp - correct_logits(在概率分布中最大的那个 - 正确的数字的概率，如果最大==正确，那么loss趋近于0，log_sum_exp 和 correct_logits 处理的均是未经 Softmax 的原始得分 $z$（Logits），并非归一化后的概率 $P$)

## RMSNorm:
- 它是传统 LayerNorm 的轻量化变体，通过舍弃均值平移计算，仅利用均方根（RMS）对神经元激活值进行缩放
- $$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$

# lu
## silu:
- SiLU（$\text{SiLU}(x) = x \cdot \sigma(x)$）
- 抹平了 ReLU 在 $x=0$ 处的尖锐折角

## swiglu
- $$\text{SwiGLU}(x) = \left( \text{SiLU}(x W) \odot (x V) \right) W_2$$
1. 通过 $W$ 矩阵投影后经过 SiLU 激活，生成一个介于约 $-0.28$ 到正无穷之间的动态门控权重
2. 通过 $V$ 矩阵直接进行线性变换，保留原本的特征信息值
3. 用门控权重去对候选特征逐元素“打折”或“放行”
4. 将乘积结果映射回输出维度