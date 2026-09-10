# GPU的硬件框架和影响的因素
## Warp Control Divergence(线程束分歧)
- 以32个线程组成的Wrap(线程束)为最小的调度单元，一个Wrap内的所有线程共享同一个指令发射器，执行SIMT(多线程)模式，意思是所有线程在同一时钟周期必须执行相同的指令
- 如果代码中有条件分支，wrap内部满足if和else的时候，硬件无法并行执行两个分支，需要挂起else先去执行if，再挂起if去执行else，使得硬件算力利用率折半

## shared memory bank conflicts(共享内存bank冲突)
- shared memory是SM内部的高速缓存，被平均划分为32个独立的存储体
- 如果一个wrap(线程束)在同一个时钟周期内访问同一个bank中的不同内存地址就会触发bank conflict，使得bank无法响应
- 例如在矩阵乘法读取shared memory的时候容易触发，因此我们常常采用swizzling(地址重排)或padding(补位)来错开bank的映射

## HBM Memory Coalescing(全局内存合并访问)
- GPU访问HBM是以128Bytes为单位批量访问事务
- 当warp访问一块连续且对齐的128字节的时候GPU变只需要一次便可以满足wrap的需求，使得带宽的使用率达到100%

## Block Occupancy and Wave Quantization
- GPU会将Grid中的thread block调度分发给不同的shared memory，block会以waves为单位分批投递执行
- 这会导致有时候大量的wave没能利用起来
- 调整Kernel的Grid Size，使总线程块数量尽可能是硬件SM数量的整数倍（如 148、296 等），从而彻底消除尾部闲置

# Triton
- Triton和CUDA本质上都算是一种编程模型，关注的是代码该怎么映射到硬件上面
- CUDA：
    - 按照线程的角度来编写计算逻辑，同时写出并行的掩码
    - 理论上来说性能最高，适合对硬件敏感的通用高性能计算
    - 程序员相当于是安排每一个线程去拿哪一个的内存地址，如何与Wrap内的其他线程一起配合使用
- Triton：
    - 按照数据块来进行编写
    - 对Tensor进行向量化的计算，专注于深度学习的算子
    - 采用分块程序模块，开发者直接定义一个固定大小的数据块，然后用python算术运算符对这个数据块进行整体的运算
