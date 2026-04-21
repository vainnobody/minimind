# 第 1 课：带读 `trainer/train_tokenizer.py`

这篇讲义的目标不是教你重新设计一个 tokenizer，而是借 MiniMind 的 `trainer/train_tokenizer.py` 看懂三件事：

1. 一个教学用 tokenizer 训练脚本是怎样串起来的。
2. `tokenizer.json` 和 `tokenizer_config.json` 分别承担什么职责。
3. MiniMind 为什么选择 `BPE + ByteLevel + 小词表 + 对话控制 token` 这一套组合。

> 先记住结论：MiniMind 官方并不建议重新训练 tokenizer。这个脚本主要用于教学和理解，真正训练主线模型时应尽量沿用仓库现有 tokenizer。

## 一、先看脚本在做什么

`trainer/train_tokenizer.py` 的流程很短，可以先压缩成 5 步：

1. 从 `dataset/sft_t2t_mini.jsonl` 里抽取多轮对话文本。
2. 创建一个 `BPE` tokenizer，并把预切分器设成 `ByteLevel`。
3. 提前注入一批控制 token，包括对话、工具调用、思维链和 buffer token。
4. 训练并导出 `tokenizer.json`、`vocab.json`、`merges.txt`、`tokenizer_config.json`。
5. 用 `eval_tokenizer()` 做一次最小验证，检查 chat template、编码解码和压缩率。

换句话说，这个脚本不是只“训词表”，而是在同时定义：

- 词表本身
- token 切分规则
- 控制 token 集合
- Hugging Face 运行时配置
- 对话模板 `chat_template`

## 二、带读源码

### 1. 入口常量

脚本开头先定义了四个常量：

```python
DATA_PATH = '../dataset/sft_t2t_mini.jsonl'
TOKENIZER_DIR = '../model_learn_tokenizer/'
VOCAB_SIZE = 6400
SPECIAL_TOKENS_NUM = 36
```

这里最值得讲的是两点：

- 数据源选的是 `sft_t2t_mini.jsonl`，不是纯 pretrain 语料。这说明这个教学脚本希望 tokenizer 直接见到多轮对话、角色标记和工具调用场景。
- `VOCAB_SIZE=6400` 很小，这和 MiniMind 的“小模型定位”一致。词表越大，embedding 层和 lm head 占用的参数越多；对 64M 级别模型来说，这个取舍很重要。

### 2. `get_texts()`：把对话数据变成训练语料

`get_texts(data_path)` 是一个生成器。它逐行读取 `.jsonl`，抽取每条样本里 `conversations` 的 `content`，再用换行符拼起来：

```python
contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
yield "\n".join(contents)
```

这里可以重点讲三个设计：

- 它只取前 `10000` 行，明显是为了教学时更快看到结果，而不是追求极限词表质量。
- 它只保留 `content`，不保留额外字段，说明 tokenizer 训练关注的是文本分布，而不是监督标签。
- 它对坏样本比较宽容：`errors='ignore'`，并跳过 JSON 解析失败的行。这是一种很常见的工程化容错。

### 3. `train_tokenizer()`：真正的训练主线

函数一开头先创建 tokenizer：

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

这两行基本决定了 MiniMind tokenizer 的骨架：

- `models.BPE()`：表示底层词表是通过 merge 学出来的 BPE。
- `ByteLevel`：表示切分前先退回字节级处理，而不是直接假设“中文按字、英文按词”。

这套组合的好处是稳定、通用，对中英混合文本也比较鲁棒。代价是压缩率通常不如为某种语言深度定制过的 tokenizer。

### 4. 控制 token 是怎么组织的

脚本把 token 分成三层：

```python
special_tokens_list
additional_tokens_list
buffer_tokens
```

它们对应的角色分别是：

- `special_tokens_list`：核心系统 token，如 `<|im_start|>`、`<|im_end|>`、视觉/音频相关 token。
- `additional_tokens_list`：教学和 Agent 场景常用标签，如 `<tool_call>`、`<tool_response>`、`<think>`。
- `buffer_tokens`：预留 token 位，便于以后扩展而不打乱整体规划。

这里有一个非常值得带读的点：

```python
num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]
```

这说明脚本不是“想到什么 token 就塞什么”，而是先给控制 token 预留一个总预算 `36`，再把剩余位置留给 buffer。对一个教学仓库来说，这种写法很适合用来讲“协议位设计”。

### 5. `BpeTrainer`：训练器真正吃进去的是什么

训练器配置如下：

```python
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=all_special_tokens
)
```

这段要重点讲两件事：

- `initial_alphabet=ByteLevel.alphabet()`：保证所有基础字节都能被表示。它是 ByteLevel tokenizer 稳定性的关键之一。
- `special_tokens=all_special_tokens`：训练时会把这些 token 作为保留项直接放进词表，而不是让 BPE 自己从文本里“碰运气学出来”。

然后脚本调用：

```python
tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.decoder = decoders.ByteLevel()
```

前者完成训练，后者补上解码器。到这里，tokenizer 的“算法层”就完整了。

### 6. 为什么还要手改 `tokenizer.json`

训练完后，脚本没有直接结束，而是重新打开 `tokenizer.json`：

```python
for token_info in tokenizer_data.get('added_tokens', []):
    if token_info['content'] not in special_tokens_list:
        token_info['special'] = False
```

这一步非常关键。它表达的是：

- 核心系统 token 仍然视为真正的 special token。
- `<tool_call>`、`<tool_response>`、`<think>` 和 buffer token 虽然被预留进词表，但不希望在下游完全按“特殊控制符”对待。

教学时可以这样解释：MiniMind 既希望这些标签是稳定的单个 token，又希望它们还能正常参与文本建模、显示和生成，而不是被很多工具链默认跳过。

### 7. `tokenizer_config.json`：不只是元信息，还是协议定义

后半段的 `config = {...}` 很重要，因为它不只是 Hugging Face 配置，还把 MiniMind 的对话协议写进去了：

- `bos_token = <|im_start|>`
- `eos_token = <|im_end|>`
- `pad_token = <|endoftext|>`
- 多模态相关 token 映射
- 最关键的 `chat_template`

要特别强调：`tokenizer.json` 决定“怎么切 token”，`tokenizer_config.json` 决定“上层应用怎么组织对话”。  
MiniMind 的 Tool Call、Thinking、system/user/assistant/tool 角色格式，实际上有很大一部分是由 `chat_template` 固化下来的。

### 8. `eval_tokenizer()`：这不是附赠代码，而是教学入口

`eval_tokenizer()` 做了四类检查：

1. `apply_chat_template()` 看模板展开结果。
2. 编码再解码，验证 round-trip 是否成立。
3. 统计 `Chars / Tokens`，观察压缩率。
4. 按 token 增量解码，帮助理解 ByteLevel tokenizer 的输出边界。

如果你带新人读源码，这一段非常适合现场演示。因为它把“抽象的 tokenizer 规则”变成了可观察的字符串和 token id。

## 三、MiniMind tokenizer 的结构拆解

训练脚本最终导出的 tokenizer，至少要分成两个文件来看。

### 1. `tokenizer.json`：算法与词表主体

可以把它理解成“真正的 tokenizer 内核”。当前仓库中的结构关键信息是：

- `model.type = BPE`
- `pre_tokenizer.type = ByteLevel`
- `decoder.type = ByteLevel`
- `vocab = 6400`
- `merges = 6108`
- `added_tokens = 36`

其中 `tokenizer.json` 里最关键的几段是：

- `added_tokens`：预留和控制 token
- `pre_tokenizer`：切分前的字节级策略
- `decoder`：把 token 还原回字符串的规则
- `model.vocab`：最终词表
- `model.merges`：BPE merge 规则

### 2. `tokenizer_config.json`：运行时语义

这个文件更像“给 Transformers 和上层应用看的说明书”。它告诉加载器：

- 哪些 token 是 `bos/eos/pad/unk`
- 哪些 token 属于 `additional_special_tokens`
- 多模态 token 各自代表什么
- 最后如何根据消息列表拼装 prompt

对 MiniMind 来说，这个文件尤其重要，因为它把 `chat_template` 和工具调用协议放进了 tokenizer 层。也就是说，tokenizer 在这里不只是“分词器”，还是“提示词格式化器”。

### 3. 36 个预留 token 是怎样分层的

当前配置里，一共预留了 36 个 token：

- 21 个核心 special token
- 6 个教学/Agent 标签 token  
  也就是 `<tool_call>`、`</tool_call>`、`<tool_response>`、`</tool_response>`、`<think>`、`</think>`
- 9 个 buffer token

更细一点看：

- `special=true` 的是系统级控制 token，共 21 个。
- `special=false` 的是工具标签、思维标签和 buffer token，共 15 个。

这个分法很像“分层协议设计”：

- 第一层保证系统和多模态边界稳定。
- 第二层给推理、Tool Use、Thinking 预留显式标记。
- 第三层给未来兼容性留出扩展槽位。

## 四、为什么 MiniMind 要这样设计 tokenizer

从教学角度，可以把这套设计总结成 4 个取舍：

### 1. 小词表优先

`6400` 词表明显不是为了追求最强压缩率，而是为了控制 embedding 和输出层参数占比，让小模型也能承受。

### 2. ByteLevel 优先稳健性

ByteLevel 会让 tokenizer 对中英混合、标点、特殊符号更稳，不容易出现完全不可编码的字符。

### 3. 先把协议 token 设计好

MiniMind 不是先训出词表，再临时补 `<think>` 或 `<tool_call>`；而是一开始就把这些 token 当作协议的一部分纳入设计。

### 4. tokenizer 不只服务训练，也服务推理接口

因为 `chat_template` 写在 tokenizer 配置里，所以同一套 tokenizer 同时服务：

- 训练阶段的数据格式对齐
- `eval_llm.py` 的推理
- `serve_openai_api.py` 的接口适配
- `web_demo.py` 的前端交互展示

这也是为什么教程里一定要把 `tokenizer.json` 和 `tokenizer_config.json` 放在一起讲。

## 五、带新人讲这一课时，建议这样讲

建议按这个顺序：

1. 先讲“tokenizer 不只是分词，而是协议入口”。
2. 再读 `get_texts()`，说明训练语料从哪里来。
3. 再读 `BPE + ByteLevel + special tokens` 这条主线。
4. 然后切到 `tokenizer.json`，让学生看到训练产物长什么样。
5. 最后讲 `tokenizer_config.json` 和 `chat_template`，把 tokenizer 和推理接口连起来。

如果教学时间有限，这一课最少要确保学生真正理解两句话：

- MiniMind 的 tokenizer 是 `BPE + ByteLevel`，目标是让小模型在稳定性和参数规模之间取得平衡。
- MiniMind 的 tokenizer 不只负责“切词”，还负责把对话、工具调用和 thinking 协议组织成模型真正看到的 prompt。

## 六、推荐的课后动手实验

可以让学生自己做 3 个小实验：

1. 把 `VOCAB_SIZE` 改大或改小，观察词表规模和压缩率变化。
2. 注释掉 `<think>` 或 `<tool_call>`，观察 `chat_template` 和推理格式会受到什么影响。
3. 在 `eval_tokenizer()` 里加入自己的中英混合文本，比较 `Chars / Tokens`。

这三个实验足够让新人把“代码、结构、现象”连起来。
