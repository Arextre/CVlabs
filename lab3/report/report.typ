#import "@preview/cuti:0.2.1": show-cn-fakebold
#import "@preview/mitex:0.2.5": *
#show: show-cn-fakebold
#show math.equation.where(block: false): math.display

#let 字号 = (
  初号: 42pt,
  小初: 36pt,
  一号: 26pt,
  小一: 24pt,
  二号: 22pt,
  小二: 18pt,
  三号: 16pt,
  小三: 15pt,
  四号: 14pt,
  中四: 13pt,
  小四: 12pt,
  五号: 10.5pt,
  小五: 9pt,
  六号: 7.5pt,
  小六: 6.5pt,
  七号: 5.5pt,
  小七: 5pt,
)

#let 字体 = (
  // 宋体，属于「有衬线字体」，一般可以等同于英文中的 Serif Font
  宋体: ("Times New Roman", "SimSun"),
  // 黑体，属于「无衬线字体」，一般可以等同于英文中的 Sans Serif Font
  黑体: ("Arial", "SimHei" ),
  // 等宽字体，用于代码块环境，一般可以等同于英文中的 Monospaced Font
  等宽: ("Consolas", "FangSong"),
  楷体: ("Times New Roman", "KaiTi"),
  仿宋: ("Times New Roman", "FangSong"),
)


/* ------------- Cover & Init ---------------- */

#align(center)[
  #set text(size: 42pt, font: "LiSu", weight: "semibold")
  华中科技大学 \
  计算机科学与技术学院
  #set text(size: 20pt, font: "KaiTi", weight: "semibold")
  《计算机视觉导论》实验报告 \
  #strong[基于剪枝算法的深度神经网络压缩]
  \

  #box(height: 23%)[
    #place(center, dy: -20pt)[#image("img/icon.png", width: 44%)]
  ]
]

#{
  set text(
    size: 14pt,
    font: "STSong",
    weight: "semibold",
    top-edge: 0.7em,
    bottom-edge: -0.7em,
  )
  let distr(s, w: auto) = {
    block(
      width: w,
      stack(
        dir: ltr,
        ..s.clusters().map(x => [#x]).intersperse(1fr),
      ),
    )
  }
  align(center)[
    #table(
      stroke: none,
      columns: (4em, 0.5em, 17em),
      // columns: 3,
      [#distr("专业", w: 4em)], [：], [#box(width: 15.5em, height: 1.2em, stroke: (bottom: black))[计算机科学与技术（图灵班）]],
      [#distr("班级", w: 4em)], [：], [#box(width: 15.5em, height: 1.2em, stroke: (bottom: black))[图灵2301班]],
      [#distr("学号", w: 4em)], [：], [#box(width: 15.5em, height: 1.2em, stroke: (bottom: black))[U202314607]],
      [#distr("姓名", w: 4em)], [：], [#box(width: 15.5em, height: 1.2em, stroke: (bottom: black))[向恩泽]],
      [#distr("成绩", w: 4em)], [：], [#box(width: 15.5em, height: 1.2em, stroke: (bottom: black))[]],
      [#distr("指导教师", w: 4em)], [：], [#box(width: 15.5em, height: 1.2em, stroke: (bottom: black))[#distr("刘康", w: 3em)]],
    )
  ]
}


#place(bottom + center, dy: -1em)[
  #set text(size: 16pt, font: "STSong", weight: "bold")
  完成日期：2025 年 11 月 2 日
]


#pagebreak()

#show raw.where(block: true): it => {
  set text(font: 字体.等宽)
  line(length: 100%)
  it
  line(length: 100%)
}

// set first-line-indent and par-justify
#set par(first-line-indent: (amount: 2em, all: true), justify: true, leading: 1em)
#show heading.where(level: 1): set block(below: 1.5em)

// initial setting of simple fonts and pars
#set text(lang: "zh", region: "CN", size: 12pt, top-edge: 0.7em, bottom-edge: -0.3em, font: 字体.楷体)

/* ------------- outline page ------------ */

#set page(
  paper: "a4",
  header: [
    #place(box(width: 1fr, height: 2pt, stroke: (top: gray, bottom: gray)), dy: 0.5in)
  ],
  footer: [
    #box(width: 1fr, height: 6pt, stroke: (top: gray))
  ]
)

#{
  set footnote.entry(separator: none,)
  show footnote.entry: hide
  show ref: none
  show footnote: none
  show align: none
  set text(font: ("Times New Roman", "SimSun"))
  show outline.entry.where(level: 1): it => {
    set text(size: 14pt)
    v(16pt, weak: true)
    strong(it)
  }
  show outline.entry: it => smallcaps(it)
  outline(
    title: "目录",
    indent: 2em,
    depth: 2,
  )
}
#pagebreak()

/* ---------- main matters ------------- */

#counter(page).update(1)
#set page(numbering: "1")

#set page(
  paper: "a4",
  header: [
    #place(box(width: 1fr, height: 2pt, stroke: (top: gray, bottom: gray)), dy: 0.5in)
  ],
  footer: [
    #box(width: 1fr, height: 6pt, stroke: (top: gray))[#place(center, dy: 5pt)[#context counter(page).display("1 / 1", both: true)]]
  ]
)

// set head elements
#set heading(numbering: "I.1.a:")
#show heading: set text(font: ("Times New Roman", "SimHei"), weight: "semibold")
#show heading.where(level: 1): it => {
  set align(center)
  set text(size: 16pt)
  it
}

// #set text(size: 16pt)

= 任务要求

#h(-2em)#strong[任务要求]：

对实验二构建的分类神经网络进行权重剪枝实现模型压缩。

#h(-2em)#strong[实现步骤]：

1. 可对最后一层卷积层，依据输出特征图的神经元激活的排序，进行依次剪枝。例如：若最后一层卷积层的权重大小为 $D×3×3×P$，输出特征图大小为 $M×N×P$，在测试数据集上对 $P$ 个输出特征图的神经元激活（$"test_dataset_size"×M×N$）求平均并进行排序。按激活水平由低到高，对前 $K$ 个神经元权重进行剪枝，$K=1 "to" P-1$。

2. 	剪枝后的卷积层权重大小为 $D×3×3×(P-K)$，测试此时神经网络分类准确率。


= 环境介绍

本次实验使用 Python 3.10 版本进行开发，并使用 Pytorch 2.8 + cu128 作为深度学习框架。具体环境信息如下：

#table(
  align: horizon + center,
  columns: (0.5fr, 1fr),
  table.header(
    [*环境名*], [*具体信息*],
  ),
  [CPU], [AMD Ryzen 9 9955HX3D 16-Core Processor\*],
  [GPU], [NVIDIA GeForce RTX 5070ti Laptop 12G],
  [DRAM], [32 GB(16GB x2) DDR5 5200MT/s],
  [操作系统], [Windows 11 + WSL2 Ubuntu 22.04 LTS],
  [Python 版本], [Python 3.10.12],
  [Pytorch 版本], [Pytorch 2.8 + cu128],
)

\* AMD Ryzen 9 9955HX3D 默认包含两个 CCD（CCD0 和 CCD1），每个 CCD 提供 8 核心（16 线程），总计 16 核心（32 线程）。由于环境设置，CCD1 已被禁用，仅 CCD0 在工作。CCD0 几乎等效于 Ryzen 7 9800X3D，提供 8 核心（16 线程）和 96 MB 的 3D V-Cache。

= 数据集简介与预处理

MNIST 数据集是一个经典的手写数字识别数据集，包含 60000 张训练图片和 10000 张测试图片，每张图片为 28x28 像素的灰度图像，表示手写的数字（0-9）。本次实验中，我根据实验要求在 MNIST 数据集中，分别于训练集和测试集中各随机选取其中 10% 的数据，构建了本次实验所需的训练集和测试集。共含有 6000 张训练图片（每类数字 600 张）和 1000 张测试图片（每类数字 100 张）。

当然，由于图片原始是 [0, 255] 的灰度值范围，因此我对图片进行了归一化处理，将像素值缩放到 [0, 1] 范围内，以便于神经网络的训练。具体的预处理可以参考以下代码：

```py
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

选择出原始的训练图片和测试图片之后，为了匹配本次实验任务，我选择以下逻辑构建训练集，而测试集同理：

1. 正样本：从每个类别数字中，进行#strong[完全的]两两匹配，每一对都会构成一个正样本对，标签为二维向量 $[0, 1]$ 表示不同的概率为 0，相同的概率为 1；这样构成的正样本对理论上有 $(600 times 599) / 2 times 10 = 1797000$ 对（具体实验可能因为奇怪取整稍有偏差）。

```py
# for label == 1
for i in range(10):
    buc_size = len(selected_data[i])
    for j in range(buc_size):
        for k in range(j + 1, buc_size):
            img1 = selected_data[i][j]
            img2 = selected_data[i][k]
            label = torch.tensor([0, 1], device=device, dtype=dtype)

            self.data.append((img1, img2))
            self.labels.append(label)
            label_counter_1 += 1
```

2. 负样本：如果按照正样本进行完全组合构造负样本，会导致负样本数量远大于正样本，从而导致数据集严重不平衡，影响模型训练效果。因此，我使用类似下采样的思想，在所有负样本对中随机选取与正样本数相等的负样本对。具体选择方式为：随机选择两个不同类别的数字，然后从各自类别中随机选择一张图片，构成一个负样本对，标签为二维向量 $[1, 0]$ 表示不同的概率为 1，相同的概率为 0；

```py
random.seed(seed)
for _ in range(label_counter_1):
    label_i, label_j = random.sample(range(10), 2)
    assert label_i != label_j, "label_i and label_j must be different"
    buc_size_i = len(selected_data[label_i])
    buc_size_j = len(selected_data[label_j])
    idx_i = random.randint(0, buc_size_i - 1)
    idx_j = random.randint(0, buc_size_j - 1)

    img1 = selected_data[label_i][idx_i]
    img2 = selected_data[label_j][idx_j]
    label = torch.tensor([1, 0], device=device, dtype=dtype)

    self.data.append((img1, img2))
    self.labels.append(label)
```

= 模型设计

由于本次实验任务是针对第二次实验模型进行剪枝，因此这一部分内容将只会简略介绍模型。

在实验二中，我设计了一种基于孪生网络（Siamese Network）结构的深度神经网络，用于对输入样本对之间的特征相似度进行建模与判别。模型整体由三个主要模块组成：Embedding 模块、ResidualBlock 残差模块以及Siamese 主干网络，最终由一个分类头部输出两类判别结果。

网络的具体实例化可以在 `./main.py` 中查看 `model = Net(...)`，实例化时可以指定网络的嵌入通道维度以及在孪生网络中的隐藏通道维度。由于本次实验任务是对模型最后一个卷积层进行剪枝，因此接下来将只对模型的主要部分进行介绍。

== 模型的简要说明

该模型最重要的部分即特征提取网络，其主要负责提取输入图片对的深层特征。其结构由多个 ResidualBlock 模块堆叠而成，每个模块后面跟随一个最大池化层，用于降低特征图的空间维度。通过多层残差模块的堆叠，网络能够学习到更加复杂和抽象的特征表示。最后通过 Flatten 层以及一个全连接层，将提取到的特征映射到一个固定维度的嵌入空间中。具体网络结构如下：

#figure(
  image("img/Extractor.svg"),
  caption: "Siamese 特征提取网络结构示意图"
)

而特征提取网络将作为孪生网络的一个重要部件，主干网络由两个共享权重的特征提取网络组成，分别处理输入的两张图片。通过共享特征提取器权重，确保两个分支提取到的特征具有相同的表示能力。提取到的两个嵌入向量随后通过一系列拼接、全连接层来获得二者的“距离”。最终得到一个二维向量，表示输入图片对属于不同类别和相同类别的权重。具体网络结构如下：

#figure(
  image("img/SiameseNetwork.svg"),
  caption: "Siamese 主干网络结构示意图"
)

== 模型剪枝分析

经过上述模型结构的分析，容易注意到模型的最后一个卷积层位于特征提取器（Extractor）中，具体位置是特征提取器的第二个残差块的第二个卷积层（即 `Extractor.ResidualBlock2.conv2`）。该卷积层的输入通道数和输出通道为超参数，由启动脚本 `--hidden_channels` 设置。

= 剪枝实验策略与分析

== 剪枝策略

由实验要求，我们需要以模型的激活水平为依据，进行剪枝操作。具体来说，我们需要在测试数据集上，对最后一个卷积层的每个输出特征图的神经元激活进行统计，计算每个输出特征图的平均激活水平。然后根据激活水平的排序，选择性地剪除那些激活水平较低的特征图对应的卷积核权重。

具体的代码实验，使用了 PyTorch 提供的钩子函数（hook）机制，在前向传播过程中捕获最后一个卷积层的输出特征图，并计算其激活水平。随后根据激活水平的排序，选择性地剪除对应的卷积核权重。剪枝后的模型将重新进行测试，以评估剪枝对模型性能的影响。

具体的钩子函数实现如下：

```py
def hook_fn(_module, _input, output):
    nonlocal channel_sum, means_count
    out = output.detach()
    b, c, h, w = out.shape
    cur_sum = out.float().sum(dim=sum_dim)  # (C, ) or (C, H, W)
    if channel_sum is None:
        channel_sum = cur_sum.cpu().clone()
    else:
        channel_sum += cur_sum.cpu()
    means_count["pixel_means"] += b * h * w
    means_count["batch_means"] += b
```

该函数在每次前向传播时被调用，累积最后一个卷积层输出特征图的激活值，并统计像素数量和批次数量，以便后续计算平均激活水平。该函数的实现位于 `./prune_module.py` 文件的 `collect_channel_activate_means()` 函数中，外层函数声明 `channel_sum` 与 `means_count` 变量，用于统计每个通道的激活总和以及像素和批次数量。

== 数据分析

设置卷积通道数 32，剪去 30 通道后，模型在测试集上的准确率变化如下表所示：

#figure(
  image("img/prune.png"),
  caption: "模型在测试集准确率随剪枝通道数变化图（smooth=0.4）"
) <img1>

从 @img1 中可以看出，在剪枝的早期，模型还能在一定程度上维持准确率。但随着剪枝通道数的增加，模型在测试集上的准确率逐渐下降。当剪去 30 个通道时，模型的准确率仍然保持在较高水平，表明该剪枝策略在一定程度上保留了模型的性能。然而，当剪去更多通道时，准确率开始显著下降，说明过度剪枝会损害模型的表达能力。



需要补充的是，实验代码剪枝顺序如下所示：

```txt
Pruned channels: [6, 7, 29, 10, 27, 12, 16, 24, 15, 1, 4, 5, 11, 8, 31, 9, 18, 3, 0, 2, 17, 28, 13, 22, 19, 26, 21, 23, 30, 20]
```

这与呈现的最后一层卷积层的激活强度的灰度图从肉眼上看是匹配的（颜色越浅表示激活越强）：

#figure(
  image("img/feature_maps.png"),
  caption: "最后一层卷积层输出特征图激活强度热力图（浅色更强）"
) <img2>

此处需要进行补充说明一点：由于代码每次进行剪枝的模型都是在本地进行重新训练的，因此每次剪枝的顺序会因为随机初始化和训练过程中的随机性而有所不同，导致剪枝顺序与 @img2 中的激活强度排序并不完全一致。

同时，针对训练出来的不同的模型，在这些模型上应用上述剪枝逻辑也可能呈现出不同的准确率变化结果。在完成这份报告之前，我便在本地环境中进行多次模型训练与剪枝，测试变量包括训练集大小、模型通道数量与剪去通道数量等，并观察到某些现象：

1. 在部分情况下，剪去部分通道后，模型的准确率反而有所提升；

2. 减小训练集规模后，模型准确率整体有所下降，但剪枝后准确率的下降幅度减小；

3. 增加模型通道数量后，模型准确率整体有所提升，同时剪枝后准确率的下降幅度减小。

== 实验分析与结论

在多次实验与观察后，可以得到一些大概率准确的结论：

1. 剪枝的确会降低模型的表达能力，从而导致准确率下降，不过在剪枝的早期，模型仍然能够维持较高的准确率，说明 “部分通道对于模型性能的贡献较小” 这一观点是成立的；

2. 剪枝后准确率有所提升的现象，可能是由于剪枝起到了某种正则化的作用，减少了模型的过拟合，从而在测试集上表现更好；当然，也不止这一种可能，比如剪去某些通道之后恰好“人为地”去除了一些噪声特征，从而提升了模型的泛化能力；

3. 训练集规模的减小，导致模型整体准确率下降，但剪枝后准确率下降幅度减小，可能是因为在较小的训练集上，模型本身就难以学习到复杂的特征，因此剪枝对其影响相对较小；

4. 增加模型通道数量后，模型准确率提升，同时剪枝后准确率下降幅度减小，说明更大的模型具有更强的表达能力和冗余性，因此在剪枝时能够更好地保留重要特征，从而减小对性能的影响。

5. 总体而言，剪枝作为一种模型压缩技术，能够在一定程度上减少模型的复杂度和计算资源需求，但需要谨慎选择剪枝策略和参数，以平衡模型性能和效率之间的关系。

== 补充分析以及 LoRA 理论联动（雾

从本地进行的多次实验的准确率来看，模型在剪枝早期能够保持较高的准确率，这与 LoRA 微调的核心思想有一定的相似性。LoRA 微调通过在预训练模型的基础上，针对特定任务进行轻量级的适应性调整，从而提升模型在该任务上的性能，而其核心思想是利用低秩矩阵分解来减少参数量，同时保留模型的表达能力。

LoRA 的理论基础是，预训练模型中存在大量冗余参数，这些参数并非全部对特定任务有重要贡献。通过引入低秩矩阵，可以有效地捕捉任务相关的关键信息，同时减少不必要的参数，从而实现模型的压缩和加速。而在低秩矩阵上对模型进行微调，无论是时间还是成本，都远低于对整个模型的全调参。

而本次实验中的剪枝策略，正是通过分析模型的激活水平，识别出那些对模型性能贡献较小的通道，并将其剪除，从而实现模型的压缩。这样的剪枝操作以及其得到的一系列实验结果，可以从侧面印证 LoRA 的理论基础，即模型中确实存在大量冗余参数，而通过合理的剪枝或低秩分解，可以在一定程度上保留模型的表达能力，同时减少计算资源的需求。

= 启动 / Startup

== 环境要求

由于本次实验数据量较大，如果需要在本地运行实验代码，请确保硬件环境满足以下要求：

- GPU：建议使用 NVIDIA GPU，显存至少 8GB（经过计算，如果 select_scale=0.1，那么使用 float32 的数据量大概在 5GB，使用 bfloat16 半精度则需要约 2.5GB）；如果 GPU 显存不足，可以考虑将数据集保存在 CPU 内存中，此时要求 CPU 内存充足，建议至少 32GB RAM。

== 依赖安装

代码所需依赖保存在 `./requirements.txt` 文件中，可以通过以下命令进行安装：

```bash
pip install -r ./requirements.txt
```

#h(-2em)本实验使用 python venv 虚拟环境进行环境隔离与开发，建议在虚拟环境中安装依赖。

== 启动脚本

实验代码的启动通过命令行参数进行配置，主要参数包括：

- `--select_scale` : 选择 MNIST 数据集的比例，默认为 0.1（即 10%）；
- `--embed_channels` : Embedding 模块的输出通道数，默认为 8；
- `--hidden_channels` : Siamese 主干网络中的隐藏通道数，默认为 32；
- `--learning_rate` : 学习率，默认为 0.001；
- `--batch_size` : 每个 mini-batch 的样本数量，默认为 1024；
- `--num_epoch` : 训练轮数，默认为 10；
- `--log_path` : Tensorboard 日志保存目录，默认为 `./notebook/logs`。

- `--num_prune` : 剪枝通道数，默认为 30。

本地启动脚本 `launch.sh` 示例：

```bash
LOG_DIR="./notebook/logs/SiameseNetwork"
# clear logs folder
rm -r $LOG_DIR
mkdir -p $LOG_DIR

# run main.py with specified arguments
python main.py \
    --select_scale=0.1 \
    --embed_channels=8 \
    --hidden_channels=32 \
    --learning_rate=1e-3 \
    --batch_size=1024 \
    --num_epoch=1 \
    --log_path=$LOG_DIR \
    --num_prune=30
```

然后运行 `./launch.sh` 即可一键开始训练。

== Tensorboard 可视化

训练代码使用 Tensorboard 进行训练过程的可视化监控。

如果需要使用 Tensorboard 进行可视化，可以在命令行中运行以下命令：

```bash
tensorboard --logdir=/path/to/tensorboard/logs --port=6006
```

然后在浏览器中打开 `http://localhost:6006` 即可查看训练过程中的各种指标变化情况，例如 Prune/Accuracy。

= 实验代码与数据

实验代码和实验报告均可以在以下 GitHub 仓库中找到：#link("https://github.com/Arextre/CVlabs/tree/main/lab3", text("Arextre/CVLabs/lab3", fill: blue)).

具体的实验数据通过 Tensorboard 保存在#strong[提交文件] `./notebook/logs` 目录下，如果需要验证检查，可以使用

```bash
tensorboard --logdir=./logs --port=6006
```

#strong[注意]：Github 仓库中没有 tensorboard 日志，该日志仅在提交文件中存在（缓存日志记录被设置 .gitignore 忽略了，没有上传至 Github 仓库）

#pagebreak()

#heading(
  "附录：提交文件说明",
  level: 1,
  numbering: none,
)

提交的文件结构如下：

```
向恩泽-U202314607-实验报告二
├── launch.sh           # 启动脚本实例
├── notebook            # 运行实验的缓存目录
│   ├── logs            # Tensorboard 日志文件夹
│   │   └── ......
├── main.py             # 程序入口源代码
├── report              # 实验报告源代码
│   ├── img
│   │   └── ......
│   └── report.typ
├── requirements.txt    # 依赖列表
├── utils.py            # 工具函数源代码
├── prune_module.py     # 剪枝模块源代码
├── models.py           # 模型定义源代码
└── 实验报告.pdf        # 实验报告 pdf 版本
```

实验所用代码位于 `main.py`, `utils.py` 和 `models.py` 中。实验报告源代码位于 `report/report.typ` 中，实验日志文件保存在 `./notebook/logs/` 目录下。