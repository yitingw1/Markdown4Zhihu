## DDP Optimizer

原文链接：[TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

**以前，torchdynamo 会严重干扰 DDP 中的 计算通信重叠 compute-communication overlap**，以至于使用 dynamo 进行 DDP 训练的速度比使用 eager 进行 DDP 训练的速度慢 25%。**我们修改了 dynamo，在检测到 DDP 时添加额外的graph break，以恢复计算通信重叠的机会。**通过这些新的更改，使用 dynamo 进行 DDP 的速度不会比使用 eager 进行 DDP 慢 1% 以上，使用 torchinductor 进行编译时，在 64 个 gpu 上比 eager 快 15%。这些结果基于 6 个 OSS 模型的基准测试。

**Torch Dynamo**

如果您是 TorchDynamo 的新手，以下链接将帮助您了解最新探索。TorchDynamo 从 Python 字节码生成 FX 图，并将各种后端与 TorchDynamo 集成以完成模型的推理/训练。未来，借助成本模型，TorchDynamo 可以自动为每个子图选择最佳后端，以实现最佳性能。

- [Update 1: An Experiment in Dynamic Python Bytecode Transformation 168](https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361)
- [Update 2: 1.48x Geomean Speedup on TorchBench CPU Inference 40](https://dev-discuss.pytorch.org/t/torchdynamo-update-1-48x-geomean-speedup-on-torchbench-cpu-inference/397)
- [Update 3: GPU Inference Edition 58](https://dev-discuss.pytorch.org/t/torchdynamo-update-3-gpu-inference-edition/460)
- [Update 4: Lazy Tensors & nvFuser Experiments 34](https://dev-discuss.pytorch.org/t/torchdynamo-update-4-lazytensor-nvfuser-experiments/496/3)
- [Update 5: Improved Capture and Bigger Graphs 26](https://dev-discuss.pytorch.org/t/torchdynamo-update-5-improved-capture-bigger-graphs/556)
- [Update 6: Training support with AOTAutograd 72](https://dev-discuss.pytorch.org/t/torchdynamo-update-6-training-support-with-aotautograd/570)
- [Update 7: Inference with FX2TRT 37](https://dev-discuss.pytorch.org/t/torchdynamo-update-7-inference-with-fx2trt/576)
- [Update 8: TorchDynamo passed correctness check on 7k+ github models 76](https://dev-discuss.pytorch.org/t/torchdynamo-update-8-torchdynamo-passed-correctness-check-on-7k-github-models/663)
- [TorchDynamo Update 10: Integrating with PyTorch/XLA for Inference and Training](https://dev-discuss.pytorch.org/t/torchdynamo-update-10-integrating-with-pytorch-xla-for-inference-and-training/935)
- [TorchDynamo Update 11: Making FSDP and Dynamo Work Together](https://dev-discuss.pytorch.org/t/torchdynamo-update-11-making-fsdp-and-dynamo-work-together/1037)

**Background**

why Dynamo doesn’t work well with DDP？

DDP (Distributed Data Parallel) 分布式训练，用于synchronously training single-gpu models in parallel

**DDP训练步骤**通常如下：

1. 每个rank都会从相同的model副本开始，一个rank是一个进程，不同的rank可以是同一台机器（可能位于不同的GPU上），或者也可能是不同的机器
2. 每个rank计算不同的input batch
3. 每个rank跑forward
4. 每个rank跑backward，跑完后不同的rank上的grad是不一样的，因为用不同的input进行计算的
5. 使用allreduce call来同步grad。一个allreduce call会在不同的ranks之间进行communicate。在allreduce之后，不同rank的梯度又会变成一样的（例如，都变成不同rank的梯度的平均值）
6. 跑optimizer

<img src="C:/Users/yitingw1/OneDrive - Intel Corporation/Desktop/Notebook/resources/pytorch training/imgs/image-20240826094004256.png" alt="image-20240826094004256" style="zoom:80%;" />

在上图中，我们可以看到<u>步骤 4 和步骤 5 是通过允许 allreduce 在反向传播完成之前启动而结合在一起的</u>。通过将部分通信与反向传递的其余计算重叠，这可以加快训练过程。这就是当今**eager DDP 训练**的工作方式。默认情况下，allreduce 被收集到”buckets" 中，这些存储桶是通过启发式方法选择的，该启发式方法试图生成 25MB 的buckets（大小可配置）。

但是 - 一旦我们**启用 dynamo**(如下图)，dynamo 就会将各个内核编译成单个graph（如果发生graph break，则编译成少量graph）。然后，直到整个反向传递完成，同步才能发生。

<img src="C:/Users/yitingw1/OneDrive - Intel Corporation/Desktop/Notebook/resources/pytorch training/imgs/image-20240826094036569.png" alt="image-20240826094036569" style="zoom:80%;" />

在上图中，我们可以看到，如果使用 dynamo，DDP allreduce 直到整个反向传递计算完成才会启动。**在许多情况下，通信/计算重叠机会的损失可能比inductor提供的加速更为显著。**



**Solution**

DDP Optimizer, 其执行以下操作：

+ 检测DDP是否active
+ 如果是，则DDP Optimizer会识别DDP bucket大小并将 dynamo 图拆分为子图，以便每个子图中参数大小的总和大致等于bucket大小。

<img src="C:/Users/yitingw1/OneDrive - Intel Corporation/Desktop/Notebook/resources/pytorch training/imgs/image-20240826100037373.png" alt="image-20240826100037373" style="zoom:80%;" />

DDPOptimizer 使用的启发式方法并不总是会产生与eager DDP 产生的bucket相同的bucket；我们假设eager DDP 策略启发式方法也不是完美的，特别是在可能出现额外graph break的情况下。

**Results**

在**没有使用DDPOptimizer**的情况下，我们能够对比 DDP+dynamo 和 DDP+eager的latency，能够发现，当rank>1时，dynamo有时候比eager还要差25%

<img src="C:/Users/yitingw1/OneDrive - Intel Corporation/Desktop/Notebook/resources/pytorch training/imgs/image-20240826100511541.png" alt="image-20240826100511541" style="zoom:67%;" />

上图显示了不使用 DDPOptimizer 时 DDP 训练中 eager 和 inductor 之间的延迟比较。例如，timm_VIT 中约 25% 的减速： 64 个 gpu 上约 1720ms 的 eager 延迟，而 64 个 gpu 上的 inductor 延迟约为 2300ms。

在使用了DDPOptimizer的情况下，我们做相同的对比，可以发现DDP和eager相比少于1%的worse。而在64 gpu的配置下，最多有15%的性能提升。

<img src="C:/Users/yitingw1/OneDrive - Intel Corporation/Desktop/Notebook/resources/pytorch training/imgs/image-20240826100838355.png" alt="image-20240826100838355" style="zoom:67%;" />

我们同样可以比较DDPOptimizer带来每个model的性能提升。下面的表格是DDP+dynamo 和DDP+dynamo+DDPOptimizer的latency对比。

<img src="C:/Users/yitingw1/OneDrive - Intel Corporation/Desktop/Notebook/resources/pytorch training/imgs/image-20240826101020753.png" alt="image-20240826101020753" style="zoom:67%;" />

我们发现，**在大多数情况下，对于 1 node（即 8 GPU）配置，DDPOptimizer 带来的益处非常小（甚至会减慢速度）**，<u>因为这种配置的通信时间更短</u>。**但对于通过网络进行通信的多节点配置，我们看到了更大的加速**，尤其是对于 hf_T5_large 或 timm_VIT 等较大的模型。



**Caveats 注意事项**

**1.DDP 需要在 static_graph=False 的情况下运行。**

<u>静态图是针对 Eager DDP 的优化。它依赖于对程序行为保持不变的假设</u> - 例如，同一组参数的梯度在每次调用时必须始终以相同的顺序提供。它允许进行一些优化：

+ 重新排序bucket以更准确地匹配实际执行顺序

+ 跳过 find_unused_parameter 步骤，该步骤通常需要运行each iteration以确定反向传递中需要哪些参数。
+ Activation checkpointing

在我们测试的 6 个 OSS 模型中，我们没有看到 static_graph=True 对性能有任何可衡量的影响（至少在 Eager 模式下）。但是，已知其他一些模型可以从这些优化中受益。

**不幸的是，dynamo + DDP 目前不适用于 static_graph=True**。 （这是因为 DDP interprets any tracing as a first step，在此期间它打算收集有关运行的数据；然后后续迭代会fail some assertions）。

我们希望可以添加一些解决方法来支持这一点 - 但目前，static_graph 需要关闭才能与 dynamo 一起使用。 目前这个限制已经可以解除，PR：[[DDP] multiple forward support for static graph](https://github.com/pytorch/pytorch/pull/103487)

**2.Cudagraphs cause OOM.**

<u>Cudagraphs 在许多情况下都表现出了性能提升，但也增加了其他情况下的内存使用量</u>。由于 DDPOptimizer 创建了额外的graph，因此会加剧这些内存问题。因此，我们预计许多用户需要关闭 cudagraphs 才能使用 DDPOptimizer 运行。



**Next Steps**

+ FSDP - @wconstab 和 @aazzolini 已经开始调查使用 FSDP 模型运行 dynamo 时出现的问题。

+ 与 DDP 更好地集成可能会提供对 static_graph=True 的支持，或提供更好的性能改进。目前，<u>DDPOptimizer 会尽最大努力匹配 DDP 的 bucket；然后 DDP 根据自己的启发式方法重新划分 bucket，这可能并不总是与 DDPOptimizer 匹配。这可能会导致延迟的 allreduce 调用。</u>如果 DDPOptimizer 可以将其 bucket 选择提供给 DDP，那么这将不是问题。

### 源码

**class DDPOptimizer**相关在 torch/_dynamo/backends/distributed.py 

class DDPOptimizer注释如下

DDPOptimizer 适用于 Dynamo 编译 DistributedDataParallel (DDP) 中wrapped的模型时，将 Dynamo graph拆分成块(chunks)以单独编译，拆分与 DDP 选择的 Gradient-Allreduce 桶的边界对齐。

**Background/Motivation**

- DDP 使用 allreduce collectives来同步在不同workers上计算的部分梯度
- DDP 将梯度 allreduce 分组到“bucket”中，以优化 all-reduce 的通信效率
- 分组到bucket中的参数在时间上是相邻的，因此它们在反向计算期间大约同时准备就绪，因此可以有效地共享相同的 allreduce
- Allreduce 必须与backward compute overlap才能获得最佳训练性能
- DDP 使用从 pytorch 中的 c++ autograd engine触发的“hooks”来调度 allreduce，当单个 grad 变为“就绪”时运行
- Dynamo+AOTAutograd 生成一个从 autograd engine的角度“atomically”运行的single fused graph，以便所有梯度同时变为“ready”。hook在整个融合的反向函数执行后触发，从而阻止计算和通信的任何重叠

**Algorithm**

- DDPOptimizer 从 dynamo 跟踪的 FX 图开始，该图代表正向。它可以反向遍历此图，以确定梯度在反向期间准备就绪的真实顺序。
- 参数大小按反向顺序计数，直到达到存储桶大小限制，此时将启动新的存储桶并引入graph break
- 每个subgraph都由用户提供给 dynamo 的编译器编译，然后重新融合在一起，形成返回给用户的外部模块

**Notes**

- 最好强制（通过向 DDP 添加 API）DDP 使用此处选择的存储桶拆分，
  并且 DDP 不需要在运行时观察执行来检测或优化bucket order，就像它在 eager 中所做的那样。
- 如果 Dynamo 无法捕获由 DDP wrapped的模型的整个图，则此算法当前将产生不一定与 DDP 使用的存储桶对齐的分割。这应该会导致性能下降接近不使用图分割的基线情况，但不会更糟。
- 如果后端编译器无法编译单个子图，它将eager执行，而其余子图将compiled
- DDP 有一个“parameters_and_buffers_to_ignore”字段，DDPOptimizer 会尝试通过读取 DDP 在各个参数上留下的标记来遵守该字段。在还使用其他转换（例如重新参数化 reparameterization）的情况下，忽略标记可能会丢失。如果 DDPOptimizer 无法忽略 DDP 忽略的参数，这不是灾难性的，但可能会通过选择次优存储桶分割来影响性能。
- DDPOptimizer 始终忽略所有缓冲区，无论它们的忽略标志如何，因为缓冲区不需要梯度，因此不会被 DDP  all-reduced。（它们在forward期间broadcast，但 DDPOptimizer 不包括这一点）

**Debugging**

- 通常，使用 pdb 在单个进程程序中调试 DDPOptimizer 最容易。
- 在许多情况下，日志消息很有用（它们显示存储桶大小分配）-
  只需将 TORCH_LOGS 环境设置为包含“dynamo”、“distributed”或“dist_ddp”中的任何一个。
- 请参阅 `benchmarks/dynamo/distributed.py`，了解将在单个进程中（或使用 torchrun 在多个进程中）运行toy model或 torchbench 模型 in a single process.

**Args：**

+ bucket_bytes_cap（int）：控制用于确定graphbreaks的存储桶的大小（以字节为单位）。应设置为与原始 DDP 模块上的等效参数匹配。

+ backend_compile_fn（可调用）：一个 dynamo 编译器函数，用于编译每个子图。

+ first_bucket_cap（int）：控制第一个 bucket 的大小。应与 DDP 的第一个 bucket 上限匹配。DDP
  特殊情况下的第一个 bucket 大小，因为有时最好尽早启动一个较小的 allreduce。

class DDPOptimizer具体代码：

+ compile_fn()

  + 实现graph splitting，首先通过按图的反向顺序计算参数大小来确定一组存储桶，然后调用用户/后端编译器来编译每个子图。最后，将编译后的图拼接到一个图模块中并返回其可调用函数。

  + 1: compute the partition map according to DDP bucket logic

    + 每次新增的Bucket()都会放在最前面 `buckets.insert(0, Bucket())`

  + 2: partition the graphmodule according to bucket capacity

    ```
    split_gm = fx.passes.split_module.split_module(
        gm, None, lambda node: partition_map[node]
    )
    ```

    + 接着split_gm会通过SubmodCompiler进行编译，走到class SubmodCompiler的run_node()函数中，编译时使用fake tensor

      ```
      submod_compiler = SubmodCompiler(split_gm, self.backend_compile_fn, fake_mode)
      submod_compiler.run(*example_inputs)
      split_gm.recompile()
      ```


这个DDPOptimizer是在`torch/_dynamo/convert_frame.py`中的class CatchErrorWrapper中`__call__`使用，里面的使用位置:

```python
if config._get_optimize_ddp_mode() == "ddp_optimizer":
    ddp_module = DistributedDataParallel._get_active_ddp_module()
    if ddp_module:
        with compile_lock:
            from torch._dynamo.backends.distributed import DDPOptimizer

            ddp_optimizer = DDPOptimizer(
                bucket_bytes_cap=ddp_module.bucket_bytes_cap,
                backend_compile_fn=self._torchdynamo_orig_callable._torchdynamo_orig_callable,
            )
            assert hasattr(
                self._torchdynamo_orig_callable, "_clone_with_backend"
            ), "DDPOptimizer only supports callback fns that know how to clone themselves."
            hijacked_callback = (
                self._torchdynamo_orig_callable._clone_with_backend(
                    ddp_optimizer.compile_fn,
                )
            )
            return hijacked_callback(
                frame, cache_entry, self.hooks, frame_state
            )
```





**Improvements for DDP Optimizer** https://github.com/pytorch/pytorch/pull/87549

+ 增加了对“first_bucket_cap”参数的支持，以便更精确地对齐bucketing。 在DDP情况下，这可能会使得第一个bucket较小
+ 重构bucket拆分逻辑以使其更清晰 
+ 为存储桶信息添加pretty-print，以及一种从test case或benchmark中的 DDPOptimizer 类访问存储桶信息的方法 
+ 将调试日志转储到标准输出