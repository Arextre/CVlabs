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
  《计算视觉导论机》实验报告 \
  #strong[基于前馈神经网络的分类任务设计]
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

/*
#place(bottom + center, dy: -1em)[
  #set text(size: 16pt, font: "STSong", weight: "bold")
  完成日期：2025 年 6 月 10 日
]
*/

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

设计一个前馈神经网络，对一组数据实现分类任务。

下载 `dataset.csv` 数据集，其中包含四类二维高斯数据和它们的标签。设计至少含有一层隐藏层的前馈神经网络来预测二维高斯样本 $("data"_1, "data"_2)$ 所属的分类 label. 这个数据集需要先进行随机排序，然后选取 $90%$ 用于训练，剩下的 $10%$ 用于测试。

= 数据集简介与预处理

观察 `dataset.csv` 可知，点被分为了四类（label 分别为 1,2,3,4），每类 1000 个点，总共 N = 4000。每样本为二维向量 $("data"_1, "data"_2)$.

首先，需要对数据进行分割。考虑需要从 `.csv` 文件中加载数据，为了代码的模块化和可读性，我的代码中实现了一个 `class CSVLoader` 类来完成从 `dataset.csv` 中加载数据并进行训练集、测试集分割的任务。打乱时直接使用 `numpy` 提供的 `numpy.random.permutation` 即可。

并且，为了训练方便，需要对 label 进行变形，如果直接使用原 label 的话，label 的 shape 为 (N,)，而对分类任务，表现较好的交叉熵损失函数 `nn.CrossEntropyLoss()` 需要的 label shape 为 (N, C)，其中 C 为类别数目，因此需要将 label 变形为 (N, C) 形式。该变形特别简单，只需要将 label 变为 one-hot 形式即可（将 label 进行独热编码）：

```python
# import torch.functional as F
def to_onehot(label: torch.Tensor) -> torch.Tensor:
    label = label.to(dtype=torch.int64) - 1 # [1, 4] -> [0, 3]
    label = F.one_hot(label)
    label = label.to(dtype=torch.float32)
    return label
```

#h(-2em)该函数只需要将形如 (N, ) 的 label 传入，即可返回形如 (N, C) 的 one-hot 形式 label.

最后，对数据集按比例进行 train/validate 划分即可。需要注意的是，为了避免随机化情况下导致每种 label 在训练集和测试集中的分布不均匀，我在划分数据集时，先对每个 label 的数据分别进行划分，然后再将各个 label 的训练集和测试集合并，最终得到完整的训练集和测试集。具体细节请见 `CSVLoader` 类的代码。

由于标准化处理并不是 Loader 的逻辑，因此我将其单独放在了模型训练代码中进行处理。标准化处理的代码如下，该代码只对 feature 进行标准化处理（均值为 0，标准差为 1）：

```py
    def normalize(feature: torch.Tensor):
        # used for normalize feature data
        mean = feature.mean(dim=0, keepdim=True)
        std = feature.std(dim=0, keepdim=True)
        feature = (feature - mean) / std
        return feature
```

= 实验设计

我设计的网络结构是十分经典的 Embedding + MLP Head 结构，接下来分为 Embedding, MLP 和 Head 三个部分对我的模型进行介绍。

== Embedding Layer

由于数据 feature 是十分简单的二维向量，因此我设计的 Embedding 层也十分简单，即一个全连接层 `nn.Linear()`，其中输入维度为 2，输出的嵌入维度为 8，即 `nn.Linear(2, 8)`。该层的作用是将二维向量映射到一个更高维度的空间中，便于后续的非线性变换和分类。

== Fully Connected Network (FCN)

考虑到数据集较为简单，且数据量不大，因此模型设计上不需要特别多的参数（过多的参数，一是本地计算资源有限，参数过多跑不动，二是参数过多容易过拟合#underline("，三是确实没这个必要", evade: false, offset: -0.4em)），因此，实现代码时在代码中内置了三种 FCN 模型，根据参数规模分别区分为 tiny, normal 和 huge 三种，具体参数如下：

```python
builtin_model = {
    "tiny": [4, "tanh", 6, "leakyrelu", 4, "sigmoid"],
    "normal": [4, "tanh", 8, "leakyrelu", 8, "leakyrelu", 4, "sigmoid"],
    "huge": [4, "tanh", 8, "sigmoid", 16, "leakyrelu", 20, "leakyrelu", 10, "sigmoid"]
}
```

#h(-2em)需要注意的是，这里只是整个 Network 中的 FCN 部分，整个 Network 被我分为了 Embedding + FCN + head(Softmax to probability) 三个部分。