# 资源管理+分配+最大化
## 数据精度
- FP32：4字节，在深度学习中显存消耗较大
- FP16：2字节，指数位较少(5位)，在大模型训练中容易发生上溢
- BP16：2字节，指数位和FP32一样(都是8个)，但是牺牲了部分精度，是目前LLM预训练的标准精度
- 混合使用以上三个，在前向传播和反向传播中用后两者计算，优化器有FP32

## 计算量
- FLOPS：每秒浮点运算次数
- 前向传播计算量大概为2*模型参数量*token数量
- 反向传播计算量大概为4*模型参数量*token数量

## Compute-bound和Memory-bound
- Compute-bound为算力瓶颈，指的是computation time > communication time
- Memory-bound为传输瓶颈，指的是communication time > computation time
- 一般情况下都是memory bound，可以通过增加arithmetic intensity来解决，使得computation time增加，例如使用GeLU而非ReLU