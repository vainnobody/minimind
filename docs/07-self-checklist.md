# MiniMind 第一周自测清单

这份清单用来检查你是否真的“跑通并读懂了主线”，而不是只是执行过几个命令。

建议你在第 7 天结束时完整过一遍。

## 一、跑通验证

### 1. Tokenizer

- 我已经读过 `docs/01-train_tokenizer-guide.md`
- 我已经打开过 `trainer/train_tokenizer.py`
- 我知道 `trainer/train_tokenizer.py` 默认使用的是 `dataset/sft_t2t_mini.jsonl`
- 我已经执行过：

```bash
cd trainer
python train_tokenizer.py
```

- 我知道输出目录里至少会出现 tokenizer 相关文件
- 我能解释 `tokenizer.json` 和 `tokenizer_config.json` 各自负责什么

### 2. Pretrain

- 我已经打开过 `trainer/train_pretrain.py`
- 我已经打开过 `dataset/lm_dataset.py` 里的 `PretrainDataset`
- 我已经执行过：

```bash
cd trainer
python train_pretrain.py
```

- 我看过训练日志里的 loss 打印
- 我知道一条 `text` 样本是怎么变成 `(input_ids, labels)` 的

### 3. SFT

- 我已经打开过 `trainer/train_full_sft.py`
- 我已经打开过 `dataset/lm_dataset.py` 里的 `SFTDataset`
- 我已经执行过：

```bash
cd trainer
python train_full_sft.py
```

- 我知道 `generate_labels()` 为什么只让 assistant 回答区间参与 loss
- 我知道 `apply_chat_template()` 在 SFT 数据构造里扮演什么角色

### 4. 本地推理

- 我已经打开过 `eval_llm.py`
- 我已经执行过：

```bash
python eval_llm.py --weight full_sft
python eval_llm.py --weight full_sft --open_thinking 1
```

- 如果本地没有现成权重，我已经至少读懂了 `eval_llm.py` 的推理流程

### 5. API 服务

- 我已经打开过 `scripts/serve_openai_api.py`
- 我已经执行过：

```bash
cd scripts
python serve_openai_api.py
```

- 我知道它为什么说自己是 OpenAI 兼容接口

## 二、源码定位验证

下面这些问题，你应该能不看提示就指出文件位置。

### 1. 模型主文件在哪里

标准答案：

- `model/model_minimind.py`

### 2. 数据主文件在哪里

标准答案：

- `dataset/lm_dataset.py`

### 3. Pretrain 主入口在哪里

标准答案：

- `trainer/train_pretrain.py`

### 4. SFT 主入口在哪里

标准答案：

- `trainer/train_full_sft.py`

### 5. 最直接的本地推理入口在哪里

标准答案：

- `eval_llm.py`

### 6. OpenAI 兼容服务入口在哪里

标准答案：

- `scripts/serve_openai_api.py`

## 三、理解验证

### 1. 模型主链路

请你自己不看文档，复述下面这条链：

- `Config -> Embedding -> Attention -> FFN -> LM Head -> generate`

如果你复述不出来，回到 `docs/02-read_model_guide.md`。

### 2. 数据格式

请你自己解释：

- `jsonl` 一条样本是什么
- `conversations` 里为什么有 `role`
- `<tool_call>`、`<tool_response>`、`<think>` 分别服务什么阶段

如果你解释不清，回到：

- `docs/01-train_tokenizer-guide.md`
- `docs/04-sft-thinking-tool-guide.md`

### 3. 训练 batch 如何变成 loss

请你自己说出这条链：

- `dataset -> tokenizer/template -> input_ids/labels -> model.forward -> logits -> cross_entropy`

如果你说不清，回到：

- `docs/03-pretrain-guide.md`
- `docs/04-sft-thinking-tool-guide.md`

### 4. chat template 的角色

请你自己回答：

- 为什么训练和推理都要经过 `apply_chat_template()`

如果你说不清，说明你还没有真正把“结构化对话”和“模型实际输入文本”连接起来。

## 四、表达验证

这部分建议你口头讲给别人，或者讲给自己录音。

### 1. 3 分钟讲清楚 “Pretrain 和 SFT 的区别”

最低要求：

- 讲清数据分布不同
- 讲清训练目标侧重点不同
- 讲清为什么两者不能简单混为一谈

### 2. 3 分钟讲清楚 “为什么 Tool Call 仍然是语言建模问题”

最低要求：

- 讲清模型先生成协议文本
- 讲清外部系统再去解析工具调用
- 讲清模型本身学到的是“按格式生成”

### 3. 3 分钟讲清楚 “为什么 RL 要放在最后学”

最低要求：

- 讲清 RL 是在 SFT 之后继续优化
- 讲清它比 SFT 多了 rollout / reward / preference 这些概念
- 讲清如果前面的主线没稳住，RL 很容易变成只背术语

## 五、最终通关标准

如果下面 5 条里你有 4 条能做到，说明这周目标已经基本完成：

- 我能跑通至少一次 tokenizer、pretrain、SFT 或对应的最小启动验证
- 我能指出模型、数据、训练、推理、服务的主入口文件
- 我能解释一次训练 batch 是怎么变成 loss 的
- 我能解释 thinking 和 tool call 为什么首先是协议问题
- 我能说清 SFT、DPO、PPO、GRPO、Agentic RL 的训练信号差别

## 六、下一步建议

如果这份清单大部分都能完成，下一步最推荐的方向有两个：

### 路线 A：继续深挖主线

- 重新精读 `model/model_minimind.py`
- 重新精读 `dataset/lm_dataset.py`
- 跑更多不同参数的 Pretrain / SFT 小实验

### 路线 B：进入进阶专题

- 深读 `train_grpo.py`
- 深读 `train_agent.py`
- 把 `serve_openai_api.py` 接到外部 UI 或工具链

如果你发现自己大部分问题都卡在模板、labels、prompt 组织上，不要急着进 RL，优先再回看 SFT 这一课。
