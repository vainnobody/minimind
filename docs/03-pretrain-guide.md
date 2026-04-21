# 第 4 课：读 Pretrain 主线

这节课的目标不是复现大规模训练，而是把“一个训练脚本到底在做什么”看明白。

如果你第一次接触大模型训练，最容易被参数和工程细节淹没。MiniMind 的好处是主线够短，适合用来建立一条稳定的阅读框架。

## 这节课的目标

- 看懂 `trainer/train_pretrain.py` 的训练循环
- 看懂 `PretrainDataset` 如何把原始数据变成 `(input_ids, labels)`
- 知道 loss 是怎么从模型输出里算出来的
- 知道学习率、梯度累积、保存权重分别在代码哪里出现

## 先跑一次最小命令

```bash
cd trainer
python train_pretrain.py
```

如果你的显存比较紧张，可以先尝试减小 batch size 或 sequence length，例如：

```bash
cd trainer
python train_pretrain.py --batch_size 4 --accumulation_steps 4 --max_seq_len 256
```

第一目标不是追求效果，而是看到训练真的启动、loss 真的打印。

## 这节课先看哪些文件

- `trainer/train_pretrain.py`
- `dataset/lm_dataset.py`
- `trainer/trainer_utils.py`
- `model/model_minimind.py`

阅读顺序建议：

1. 先看 `train_pretrain.py` 里的参数和主流程
2. 再跳到 `PretrainDataset`
3. 再回来看 `model(input_ids, labels=labels)`
4. 最后看保存 checkpoint 的逻辑

## 整个 Pretrain 流程可以先记成 7 步

1. 解析训练参数
2. 初始化分布式和随机种子
3. 构建模型和 tokenizer
4. 构建 `PretrainDataset` 与 `DataLoader`
5. 前向计算得到 loss
6. 反向传播并更新参数
7. 定期打印日志和保存权重

## 从源码里分别看什么

### 1. 参数解析：训练脚本最外层的控制台接口

先看 `train_pretrain.py` 里的这些参数：

- `save_dir`
- `epochs`
- `batch_size`
- `learning_rate`
- `accumulation_steps`
- `max_seq_len`
- `data_path`
- `from_weight`

你第一次读训练脚本时，要养成一个习惯：

- 先看参数，弄清楚这个脚本控制了哪些东西

这里你要特别记住三个参数：

- `data_path` 决定训练数据从哪里来
- `max_seq_len` 决定每条样本最多截断到多长
- `accumulation_steps` 决定是否用梯度累积来省显存

### 2. `PretrainDataset`：训练数据怎么变成 token

`PretrainDataset` 在 `dataset/lm_dataset.py` 里。

这部分建议你重点看 `__getitem__()`：

1. 先读出一条样本的 `text`
2. 用 tokenizer 编码成 token ids
3. 在最前面加 `bos_token_id`
4. 在最后面加 `eos_token_id`
5. padding 到固定长度
6. `labels = input_ids.clone()`
7. 把 pad 的位置改成 `-100`

这一步是理解 Pretrain 的关键。

可以把它记成一句话：

- Pretrain 的监督信号，就是“让模型预测下一个 token”

为什么 `labels` 基本等于 `input_ids`？

- 因为训练目标是语言建模，当前位置看前文，去预测后一个 token

为什么 pad 位置要改成 `-100`？

- 因为这些位置不是有效文本，不应该参与 loss 计算

### 3. `init_model()`：模型和 tokenizer 怎么准备好

`train_pretrain.py` 里会调用 `trainer_utils.py` 里的 `init_model()`。

你第一次看这里只需要抓三件事：

- tokenizer 从 `../model` 加载
- 模型是 `MiniMindForCausalLM`
- 如果 `from_weight != 'none'`，就会加载已有权重继续训练

这里顺便也能把前一课的模型理解接上：

- 训练脚本不会重新定义模型结构
- 它只是把 `MiniMindForCausalLM` 实例化出来，然后喂数据、算 loss、更新参数

### 4. `train_epoch()`：训练循环最核心的部分

这部分建议你把每一行都当成“训练流水线”的一步。

主线可以拆成下面几步：

1. 从 `loader` 拿到 `input_ids, labels`
2. 把 tensor 放到设备上
3. 计算当前 step 的学习率
4. 调用 `model(input_ids, labels=labels)`
5. 得到 `res.loss + res.aux_loss`
6. 用 `scaler.scale(loss).backward()` 反向传播
7. 到了累积步数就执行 `optimizer.step()`
8. 打印日志
9. 定期保存权重

你读到 `res = model(input_ids, labels=labels)` 这一行时，要立刻想到前一课：

- 这会进入 `MiniMindForCausalLM.forward()`
- forward 里会先拿到 logits
- 再根据 `labels` 算交叉熵 loss

这就是“训练脚本”和“模型源码”真正接上的地方。

### 5. 梯度累积：为什么不是每个 step 都 `optimizer.step()`

代码里你会看到：

```python
loss = loss / args.accumulation_steps
```

以及：

```python
if step % args.accumulation_steps == 0:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

这说明脚本允许你把多个小 batch 的梯度累起来，再做一次参数更新。

对新手来说，你可以先这样理解：

- batch 太大显存放不下时，就拆成多个小步累计

### 6. 学习率调度：不是固定学习率一路到底

脚本里会调用 `get_lr(...)`。

你第一次不用记公式，只要知道：

- 当前学习率会随着训练步数变化
- 脚本每个 step 都会把新的学习率写回 optimizer

### 7. checkpoint 保存：训练结果怎么落盘

脚本里有两类保存：

- 直接把模型权重保存到 `save_dir`
- 用 `lm_checkpoint(...)` 保存 resume 信息

你可以这样理解：

- 前者更像“导出当前模型”
- 后者更像“保留训练现场，方便断点续训”

## 这节课最重要的三个问题

### 1. Pretrain 的监督信号是什么

答案：

- 对每个位置做 next-token prediction
- 也就是让模型根据前文预测下一个 token

### 2. “loss 下降”在这里具体意味着什么

答案：

- 模型对训练语料中的下一个 token 预测得更准了
- 换句话说，模型更好地拟合了当前语料里的语言分布

### 3. 为什么这个阶段更像“学语言统计规律”

答案：

- 数据主要是普通文本序列
- 训练目标是统一的 next-token prediction
- 模型还没有被强调成某种固定助手风格，也没有重点学习多轮角色模板

## 读代码时最容易混淆的点

### 1. Pretrain 和 SFT 都在算交叉熵，为什么还要分阶段

因为它们虽然“损失形式相似”，但“数据分布和学习目标不同”。

可以简单记成：

- Pretrain：广泛读书，学语言和世界知识分布
- SFT：对着更目标化的数据，学助手格式、回答风格和任务模式

### 2. `aux_loss` 是什么

对 Dense 主线来说，你可以先把它当成 0。

它主要和 MoE 路由辅助损失有关，不是这节课的重点。

### 3. 为什么 `labels` 和 `input_ids` 长得几乎一样

因为语言模型训练本质上是在“右移一个位置后预测下一个 token”。

真正的位移发生在模型 forward 里：

- `logits[..., :-1, :]`
- `labels[..., 1:]`

## 这节课读完后必须能回答的问题

1. 一条 pretrain 文本样本是怎么变成 `(input_ids, labels)` 的？
2. `model(input_ids, labels=labels)` 最终会在哪一层算 loss？
3. 为什么 pad 位置不会参与 loss？
4. 梯度累积是为了解决什么问题？
5. 模型权重保存和 resume checkpoint 保存有什么区别？

## 你当天应该留下的学习产物

建议你自己写一页笔记，至少包含下面三段：

### 1. 一句话解释 Pretrain

- Pretrain 是用大规模文本做 next-token prediction，让模型先学会语言统计规律

### 2. 一次 batch 的流动路径

- `json/text -> tokenizer -> input_ids/labels -> model.forward -> logits -> cross_entropy -> backward -> optimizer.step`

### 3. 你今天在代码里定位到的 5 个入口

- 参数解析
- `PretrainDataset`
- `model(...)`
- `loss.backward()`
- checkpoint 保存
