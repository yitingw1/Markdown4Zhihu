# [译]Pytorch中的Functionalization

原文链接： https://dev-discuss.pytorch.org/t/functionalization-in-pytorch-everything-you-wanted-to-know/965  

## What is functionalization？

Functionalzation是基础架构的一部分：它有助于缓解 PyTorch 中两个相互矛盾的目标之间的紧张关系：

+ PyTorch 以富有表现力和易于使用而闻名。PyTorch 拥有庞大的 API 界面，支持各种张量上的别名alias和突变mutation。

+ 我们希望能够编译 PyTorch 程序（参见 PT2 宣言）。编译器通常不喜欢mutation。

## Why do we not like mutations?

Compilers don't like mutation.

+ AOT Autograd 包含“最小割分区” “min-cut partitioning”优化。它涉及在正向和反向图之间移动节点，以减少为反向保存的张量数量，减少内存使用量，并且在许多情况下还减少运行时间（从 DRAM 读取已保存的张量有时比在反向中动态重新计算它更慢）。如果这些节点涉及副作用（mutation），那么“在正向和反向之间移动节点”是不合理的，因此这种分区逻辑需要一个功能图 functional graph。

+ AOT Autograd 运行死代码消除 dead-code-elimination(但有_copy的节点可能消除不掉），消除图中的“unused”node。如果您知道它是“functional”，那么从图中删除死节点很容易，但如果您不得不担心删除具有副作用的节点，则更困难。

+ 移动Mobile: The mobile export pipeline移动导出管道（WIP）涉及捕获图并在其上运行多个优化过程，包括死代码消除和内存规划。如果可以假设没有mutation，那么这两个过程的编写都会简单得多。

+ PyTorch/XLA：XLA 是google的编译器，它将 HLO IR 作为输入，并将其编译为多种类型的硬件（包括 TPU）。如果你盯着 HLO 中的operator set，你会发现没有mutation operations！XLA 只接受“functional programs”作为输入进行编译。

+ 其他 PT2 后端。Inductor 是 PT2 的默认后端，实际上完全能够处理mutation。随着越来越多的graph-mode后端尝试与 PyTorch 集成，他们可能更愿意获得 a graph that’s entirely functional for them to optimize。

另一种看待它的方式是：你可以想象一个没有用于mutation或别名alias的user API 的 PyTorch 版本。但 PyTorch surface API 中的view + mutation操作非常棒！它们提供了两个好处：

+ <u>它们是eager mode的一种性能优化，可以重复使用内存</u>！但是……当我们有编译器时，我们更愿意将所有优化留给编译器！

+ 它们让用户以更大的灵活性表达程序。例如，让用户仅使用函数运算符编写上述代码片段是一件麻烦事；编写mutatable 版本 (y[:, 1].add_(1)) 更直观。

## **How does it work？**
我认为 函数化 functionalization 一种程序到程序 (program-to-program)的转换。给定一个 PyTorch 运算符的程序/函数，函数化将返回一个新函数，该函数：

+ 具有与旧函数相同的语义

+ Has no mutations in it

事实上，它是 functorch 中的 API：**functionalize**

<img src="https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240806215333934.png" alt="image-20240806215333934" style="zoom:33%;" />

在 API 契约层面，这是一个非常好的思维模型。PyTorch 中有许多围绕图形捕获和图形转换的不同子系统。函数化提供的契约是，你给它一个函数 / fx.GraphModule（可能带有mutation），它会返回一个没有mutation的等效函数。

```python
import torch
from functorch import functionalize
from torch.fx.experimental.proxy_tensor import make_fx

def f(x):
    y=x.clone()
    y.add_(1)
    return y

functionalized_f=functionalize(f)
x=torch.ones(4)
print(torch.allclose(f(x),functionalized_f(x)))

# print an FX graph of the "functionalized" version of f
fx_g=make_fx(functionalized_f)(x)
print(fx_g.code)

# fx_g.code
'''
def forward(self, x_1):
    clone = torch.ops.aten.clone.default(x_1);  x_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    return add
'''
```

functionalize函数实现的伪代码如下：

```python
# Goal: loop through all nodes (op calls) in the program, eliminating mutations
# While preserving program semantics
for node in program:
    if is_mutable(node):
        # when we encounter a mutable op x.foo_(), replace it with x_updated = x.foo()
        x = node.arg
        node.op = get_functional_variant(node.op)
        # (1) replace all later usages of x with x_updated
        x_updated = node.op.output
        for use_site in x.later_usage_sites():
            use_site.arg = x_updated
        # (2) for every alias of x, regenerate it, given the mutation (given x_updated)
        for alias in node.arg.aliases():
            alias_updated = regenerate_alias(alias, node.arg)
            # (3) and replace all later usages of that alias with alias_updated
            for use_site in alias.later_usage_sites():
                use_site.arg = alias_updated  
```

## **Examples: Before & After**

So if you’re a compiler operating on a PyTorch program **post-functionalization**, what should you expect?

下面是一些functionalization转换的例子。需要注意的是， functionalization的转换是ATen API之间的转换.

### **Example 1**: simple case (1 view + mutation)

![image-20240806220458821](https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240806220458821.png)

```python
# fx_g.code
def forward(self, x_1):
    clone = torch.ops.aten.clone.default(x_1);  x_1 = None
    view = torch.ops.aten.view.default(clone, [-1]);  clone = None
    add = torch.ops.aten.add.Tensor(view, 1);  view = None
    view_1 = torch.ops.aten.view.default(add, [2, 2]);  add = None
    view_2 = torch.ops.aten.view.default(view_1, [-1])
    return view_1
```



### **Example 2: mutate a slice**

张量的高级索引advanced indexing通常会分解为 ATen 运算符，如 **aten.slice 和 aten.select**。给定张量的updated “slice”及其原始base，ATen 还有一些运算符表示生成“updated”的base tensor：**slice_scatter、select_scatter** 等

![image-20240806222012416](https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240806222012416.png)

看起来就是memory metadata操作+inplace操作(mutate)

```python
# fx_g.code
def forward(self, x_1):
    clone = torch.ops.aten.clone.default(x_1);  x_1 = None
    slice_1 = torch.ops.aten.slice.Tensor(clone, 0, 0, 9223372036854775807)
    select = torch.ops.aten.select.int(slice_1, 1, 1);  slice_1 = None
    add = torch.ops.aten.add.Tensor(select, 1);  select = None
    slice_2 = torch.ops.aten.slice.Tensor(clone, 0, 0, 9223372036854775807)
    select_scatter = torch.ops.aten.select_scatter.default(slice_2, add, 1, 1);  slice_2 = add = None
    slice_scatter = torch.ops.aten.slice_scatter.default(clone, select_scatter, 0, 0, 9223372036854775807);  clone = select_scatter = None
    slice_3 = torch.ops.aten.slice.Tensor(slice_scatter, 0, 0, 9223372036854775807)
    select_1 = torch.ops.aten.select.int(slice_3, 1, 1);  slice_3 = None
    return slice_scatter
```

### **Example 3: multiple outstanding aliases.** 

当我们mutate别名时, 我们需要弄清楚如何将mutation传播到所有未完成的别名--

<img src="https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240806222618608.png" alt="image-20240806222618608" style="zoom:50%;" />

```python
# fx_g.code
def forward(self, x_1):
    clone = torch.ops.aten.clone.default(x_1);  x_1 = None
    transpose = torch.ops.aten.transpose.int(clone, 1, 0)
    view = torch.ops.aten.view.default(clone, [2, 2]);  clone = None
    add = torch.ops.aten.add.Tensor(transpose, 1);  transpose = None
    transpose_1 = torch.ops.aten.transpose.int(add, 1, 0);  add = None
    transpose_2 = torch.ops.aten.transpose.int(transpose_1, 1, 0)
    add_1 = torch.ops.aten.add.Tensor(transpose_1, transpose_2);  transpose_2 = None
    view_1 = torch.ops.aten.view.default(transpose_1, [2, 2]);  transpose_1 = None
    add_2 = torch.ops.aten.add.Tensor(add_1, view_1);  add_1 = view_1 = None
    return add_2
```

## **History**

视图view运算符，使用它们的 *_copy() 变体替换它们。

在整个 2022 年，PT2 背后的基础设施（dynamo、aot_autograd、inductor）发展迅速，我们发现了某些带有突变的程序无法编译的漏洞。最初，dynamo 有一些逻辑可以在基本情况下删除突变，并在更复杂的情况下回退到 eager。这相当有效，但也存在一些问题：该过程无法查看和检测到我们代码库的 C++ 部分中引入的任何突变（C++ 分解、在 autograd 中调用的后向公式）。在 2022 年中期，我们增加了对函数化的支持，以直接作为 AOT Autograd 的一部分运行（在 aot_autograd 中启用 ，在 dynamo 中启用 ）

**Functionalization在 PT2 堆栈中的使用？（case study）**
AOTAutograd 需要处理的有关aliasing和mutation的一些有趣案例，展示了 PT2 中的一些设计决策
这是 PT2 堆栈 10,000 英尺视图的图片，以及功能化Functionalization在其中的位置。

![image-20240812230429359](https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240812230429359.png)

How does AOT Autograd use functionalization? There a bunch of interesting edge cases around how AOTAutograd needs to handle external mutations and aliasing in this doc 10 on AOTAutograd 2.0.

I want to cover two particularly interesting end-to-end cases.

函数化是 AOTAutograd 内部的一个组件。AOTAutograd 可以做很多事情，但在高层次上，它有一个混乱的 torch ops forward graph，其中包含突变mutation、自动求导autograd逻辑、张量子类tensor subclass以及来自 torch API 的各种内容，并将clean low-level ATen forward + backward graph 发送给compiler。

在 aot autograd 内部创建“functionalize ATen IR”的实际代码位于此处（[代码指针 1](https://github.com/pytorch/pytorch/blob/ce9963e6ba0e40b8307477f5b4113733e7a30ec2/torch/_functorch/aot_autograd.py#L1518)，[代码指针 2](https://github.com/pytorch/pytorch/blob/ce9963e6ba0e40b8307477f5b4113733e7a30ec2/torch/_functorch/aot_autograd.py#L625)）。请注意，此代码仍处于相当活跃的开发中。

需要注意的一件重要事情是，我们实际上并没有为“ATen IR”、“Functionalized ATen”和“Functionalized ATen + 分解decomps” 实现单独的graph。我们直接从“Torch IR”追溯到“Functionalized ATen + 分解decomps” 。

AOT Autograd 如何使用函数化？在 AOTAutog edge cases。

我想介绍两个特别有趣的端到端案例。

### **Case study 1: graph breaks + input mutations**

包含mutation和graph break

<img src="https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240813095011554.png" alt="image-20240813095011554" style="zoom:80%;" />

会生成两张图

![image-20240813095048244](https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240813095048244.png)

但有一个问题：在graph 2 中，“y” 是一个graph input，并且它会发生mutation。graph 2 是有状态的：运行它会导致可观察到的副作用。

这个特殊情况有点不幸：

+ 从用户的角度来看，“y” 是在函数内部创建的临时变量，并且可以轻松删除mutation。
+ 从编译器的角度来看，我们不知道“y”是什么。它可能是程序中需要变异的全局状态（例如模块参数）。在编译和执行与graph 2 对应的graph后，我们有义务确保正确保留所有副作用side effects（任何发生变异的图输入都应该变异  any graph inputs that got mutated should be mutated）。

这种情况会发生在真实的用户案例中，例如

```python
python benchmarks/dynamo/torchbench.py  --accuracy --backend aot_eager --training --only hf_Longformer
```

该模型的代码位于 HuggingFace repo中。该模型通过调用 new_zeros() 创建了一个新的零张量，“diagonal_attention_scores” 在[这里](https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py#L859)。再往下几行，它尝试在[这里](https://github.com/huggingface/transformers/blob/52c9e6af2928cbbe78a8b8664cad716aca9f32e0/src/transformers/models/longformer/modeling_longformer.py#L865)改变mutate “diagonal_attention_scores”的几个片段slice。

问题是，**对 new_zeros() 的调用会导致graph break**，因此“diagonal_attention_scores”成为下一个graph中的graph input。为什么？

+ new_zeros() 的大小参数之一是“chunk_count”
+ “chunk_count” 是通过[调用 torch.div(int, int) 创建的](https://github.com/huggingface/transformers/blob/52c9e6af2928cbbe78a8b8664cad716aca9f32e0/src/transformers/models/longformer/modeling_longformer.py#L834)
+ torch.div(int, int) 始终返回一个张量（在本例中为零维zero-dim张量）。
+ 当我们将一个维数为 0 的张量作为大小参数传递给 new_zeros() 时，<u>会在该张量上隐式调用 .item()。</u>
+ 通常，**在张量上调用 .item() 会导致graph break** 。这个想法是，我们在编译时不知道张量内部数据的值，我们需要知道具体值才能获得有关我们正在创建的张量大小的编译时信息。

请注意，在这个特定示例中，graph break是完全可以避免的。我们可以增强跟踪基础以处理(We can beef up our tracing infra to handle ) torch.div(int, int)，并将其输出视为在编译时已知。或者，有人可以更新该用户代码以执行常规 Python 整数除法，而不是使用 torch.div。

请注意，一般来说，PT2 运行在一个可能因多种原因而发生graph break的世界中 - 因此即使我们修复了这个特定模型，还有很多其他情况会导致我们陷入这种情况：graph break 可以将用户代码中的“intermediate mutation”变为子图中的任意“input mutation”。

FWIW（For what it's worth），input mutation在不久的将来会很常见的另一种情况是优化器optimizers：当我们编译优化器步骤 compile the optimizer step时，参数（通常）是我们根据梯度更新的graph input。

**How AOT Autograd handles input mutations**

如上所述，第二个graph具有input mutation，我们在编译graph时需要尊重这些input mutation。几个月前，AOT Autograd 中的input mutation support [已发布](https://github.com/pytorch/pytorch/pull/88817)。

AOT Autograd 有义务创建一个没有mutations的graph来优化，并且还有义务确保input mutations不被破坏？(respect)。它分两步完成此操作：

+ 它创建一个图，删除所有mutations（包括input mutations）。这非常简单 - 我们运行functionalization。functionalization将很乐意从图中删除所有mutations，包括input mutations。
+ 它创建一个运行时run-time“结语 epilogue”（[代码](https://github.com/pytorch/pytorch/blob/a7749ae177aa84c60815c869a20772fab7693293/torch/_functorch/aot_autograd.py#L1723)），运行编译后的图，并在返回 dynamo 之前执行任何input mutations。

在上面的例子中，AOTAutograd会用上述graph2 创建下述内容

<img src="https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240813103915516.png" alt="image-20240813103915516" style="zoom:80%;" />

**mutated input 是会被return的，return的是"updated input",** 其作为forward graph中的addtional output。input mutations会通过y.copy_(y_updated)传回到上一张图中，防止graph break使其相互作用切断

这实现了我们的目标：创建一个要编译的功能图 functional graph，并忠实地复制图中任何可观察到的副作用（input mutations），尽管你可能会认为这不是最佳选择。未来的一项工作可能是与后端编译器创建一个契约，我们同意将input mutations发送到graph中，以便编译器可以融合它。

### **Case Study 2: **TorchScript comparison

**specializing on aliasing relationships (and comparison to TorchScript)**

<img src="https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240813105337701.png" alt="image-20240813105337701" style="zoom: 33%;" />

```
tensor([4., 4.], grad_fn=<CompiledFunctionBackward>)  # out
tensor([2., 2.], grad_fn=<CompiledFunctionBackward>)  # out2
```

这是怎么回事？我们的函数改变 x，然后对 y 进行一些操作。这个程序会有不同的行为，具体取决于 x 和 y 是否为别名！

这个例子之所以有趣，是因为它展示了 PT2 决定如何编译程序与 TorchScript 之间的哲学差异。特别是**guarding + specializing**。

**TorchScript (torch.jit.script)** 和 **PT2 (torch.compile)** 在编译此程序的方式上有何不同？

+ TorchScript 将编译上述函数一次。这很有用，例如对于export，您可以获得程序的单一、忠实表示。但是，这会阻止很多优化机会。**TorchScript 无法对 x 和 y 是否别名做出任何假设**！<u>这意味着 TorchScript 将无法删除 x 上的突变，从而阻止优化机会。</u>

+ PT2 将专注于特定输入的别名属性。对于给定的一组用户输入，PT能识别别名，并可以对不同情况生成不同的graph（如是否使用了别名）

  + 缺点：<u>我们可能会编译多次</u>（如果用户碰巧多次调用他们的函数，则使用具有不同别名属性的输入）。

  + 优点：通过了解有关输入的别名信息，我们可以保证能够从我们跟踪的每个程序中删除所有mutations。<u>这意味着我们可以始终运行所有优化</u>。

两种情况下的graph如下所示：

![image-20240813110643746](https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240813110643746.png)

在右侧的图中，AOT Autograd 最终创建了一个graph，其中：

+ 只有 1 个输入 primals_1，对应于 x 和 y 都是别名的合成“base”。

+ x 和 <u>y（原始用户输入）是使用 as_strided 从 primals_1 生成的别名</u>。FWIW（For what it's worth），as_strided() 出现在graph中而不是简单的 view() 操作的原因是因为我们依赖于 autograd 的 [view-replay 逻辑](https://github.com/pytorch/pytorch/blob/5a6019033fb8121a02306034a2018d5193a97d62/torch/csrc/autograd/python_variable.cpp#L688)，该逻辑默认在 eager 模式下使用 as_strided()。我们可以考虑改变这一点，但这会对 eager 性能产生负面影响。还值得指出的是，这种特定的**alias + mutation** 情况在用户代码中很少出现

+ mul.Tensor 节点的输入参数是“as_strided_3”和“as_strided_8”；这些对应 “x_updated” 和 “y_updated”，在用户代码中“x” mutated之后。

PT2 在创建graph时有效地专注于其输入的别名alias关系，从而允许它更积极地进行优化。

FWIW，上面写的示例实际上partially broken：仍然需要教 AOT Autograd 如何将别名关系上的guards传播回 dynamo，以便它知道重新编译。相反，从 1/4/23 开始，上述代码将在第二次调用时引发断言失败。

## **Where else is functionalization used in PyTorch？**

我提到，除了 PT2 之外，函数化还用于 PyTorch 的其他领域。另外两个主要领域包括：Mobile和pytorch/xla

### **Mobile**

Mobile还有另一个要求：**导出的graph中的所有张量必须是连续的contiguous** (memory-dense)（内存密集型）。我不会在这里讲得太详细，因为移动人员之前已经写过这方面的内容（参见第 6 篇文章）。

但是函数化在这里如何提供帮助？函数化还有另一个消除view的功能。<u>如果每个输入张量都是连续的，并且图中没有视图view，那么我们可以保证每个中间和输出也是连续的</u>。简单示例：

![image-20240813121959474](https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240813121959474.png)

当使用`remove='mutations_and_views'`时, 会同时在graph中去除{view} op，将其替换为{view}_copy. 这将保证连续的output。 默认是`remove='mutations'`，前面打印的都是FX graph都是`remove='mutations'`

上述代码会打印下述内容：

![image-20240813122050493](https://raw.githubusercontent.com/yitingw1/Markdown4Zhihu/master/Data/Pytorch_Functionalization/image-20240813122050493.png)

```python
# FX graph (remove='mutations')
def forward(self, x_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    diagonal = torch.ops.aten.diagonal.default(x_1);  x_1 = None
    add = torch.ops.aten.add.Tensor(diagonal, ones);  diagonal = ones = None
    return add
```

```python
# FX graph (remove='mutations_and_views')
def forward(self, x_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(x_1);  x_1 = None
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    return add
```

torch.diagonal()会返回一个局部视图，提取对角线元素.

**diagonal_copy() 的语义是它始终返回一个连续的张量**；因此原始程序和被跟踪的程序具有相同的语义，但被跟踪的程序中的所有张量都保证是连续的。

### **PyTorch/XLA**

PyTorch/XLA 正在努力迁移以使用来自core的functionalization version。这里有一个[原型prototype正在开发中](https://github.com/pytorch/xla/pull/4158)。当我们将 LazyTensor TorchScript backend 迁移到使用函数化时，它为 XLA的集成提供了一个很好的蓝图。

现在函数化已经经过了充分的测试，更新 PyTorch/XLA 以使用它将会修复许多已知的错误：一个是 [mark_step()  API](https://pytorch.org/xla/master/#running-on-a-single-xla-device) 当前切断了别名张量之间的别名关系，这可能会导致静默正确性问题(没有报错就直接出错了)。

## **What Future Work Is There?**

+ AOT Autograd 2.0 [文档](https://docs.google.com/document/d/19UoIh_SVrMy_b2Sx5ZaeOJttm6P0Qmyss2rdBuyfoic/edit#heading=h.arzne7xas5cp) 涵盖了一系列与alias和mutation相关的eage cases；其中一些已经修复，但并非所有情况都已完成（这些错误有未解决的 github 问题，链接在文档中）。

+ 目前，<u>AOT Autograd 仅在需要跟踪backward时才真正运行functionalization</u>，而在仅编译前向图时则跳过functionalization。目前这还算可以，因为在这种情况下我们不运行最小割分区 min-cut partitioning 和 DCE (Dead Code Elimination) 之类的优化，并且inductor可以处理mutation，但其他backend无法处理mutation，更希望得到一张functional graph（请参阅 [issue](https://github.com/pytorch/pytorch/issues/90759)）。

+ **Custom dispatcher ops**。今天“functional” custom ops with functionalization（任何op内部的aliasing/mutation也是可以的），<u>但我们目前不支持具有外部可见别名/变异alias/mutation语义的自定义运算符</u>（mutate inputs, or return outputs that alias inputs）。
+ **cond op**：当我们不想专注于控制流时，cond() op 是导出用例中需要的。Tugsbayasgalan (Tugsuu) Manlaibaatar  最近为其添加了涵盖大多数情况的功能化支持：如果在真/假分支内部存在内部aliasing/mutation，它们将被功能化。最终，我们可能需要支持mutate inputs, and/or alias outputs的真/假分支true/false branches，类似于custom op story。
+ **通过张量子类tensor subclasses进行编译**。这只是间接相关的，但我们想要一个适用于张量子类的 AOTDispatch 版本，将具有张量子类的用户程序转换为仅涉及plain tensor的flat aten ops graph。作为这项工作的一部分，我们需要安排 functionalization在张量子类“下方underneath”运行。
  + 没看懂这里的tensor subclasses
+ **改进input mutation的性能**。我上面提到，当我们在 AOT Autograd 中看到input mutation时，理论上我们可以通过允许编译器“看到”input mutation并将其融合到其graph中来提高性能。我们今天不这样做，但在进行这种优化之前可能值得进行基准测试（这可能对编译优化器optimizers有用）
+ **XLA**。Pytorch/XLA 团队一直致力于从核心core迁移到使用功能化functionalization。