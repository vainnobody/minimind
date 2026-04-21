# 第 3 课：读 MiniMind 模型主线

这一课只做一件事：把 `model/model_minimind.py` 读成一条清晰的前向链路。

第一次读模型源码时，不要试图“全懂”。你只需要先把 Dense 主线读通，知道输入 token 是怎么一步步变成 logits 的。

## 这节课的目标

- 看懂 `MiniMindConfig` 在定义什么
- 看懂 `RMSNorm`、`Attention`、`FeedForward` 的职责
- 看懂 `MiniMindBlock -> MiniMindModel -> MiniMindForCausalLM` 的层级关系
- 能用自己的话讲出一次前向传播的 8 到 10 个步骤

## 先看哪些类

推荐按下面顺序阅读：

1. `MiniMindConfig`
2. `RMSNorm`
3. `Attention`
4. `FeedForward`
5. `MiniMindBlock`
6. `MiniMindModel`
7. `MiniMindForCausalLM`
8. `generate()`

## 先建立一个最小心智模型

先把整个模型理解成下面这条链：

1. 输入文本先被 tokenizer 变成 token ids
2. token ids 进入 embedding 层，变成向量
3. 每一层 block 先做注意力计算
4. 注意力输出加上残差
5. 再经过前馈网络
6. 前馈网络输出再加上残差
7. 多层堆叠后做最终 norm
8. 通过 `lm_head` 投影到词表大小
9. 得到每个位置的 logits
10. 训练时用 logits 和 labels 算交叉熵，推理时用 logits 继续采样下一个 token

你第一次读代码时，只要能把这 10 步和源码对上，就已经很成功了。

## 从源码里分别看什么

### 1. `MiniMindConfig`

重点看这些字段：

- `hidden_size`
- `num_hidden_layers`
- `vocab_size`
- `num_attention_heads`
- `num_key_value_heads`
- `intermediate_size`
- `max_position_embeddings`
- `use_moe`

你可以把它理解成“模型的结构参数表”。后面所有模块都会从这里取维度和行为配置。

第一次阅读时要特别记住：

- `hidden_size` 决定隐藏层向量宽度
- `num_hidden_layers` 决定堆几层 block
- `vocab_size` 决定 embedding 和 `lm_head` 的大小
- `use_moe` 决定前馈层走普通 FFN 还是 MoE

### 2. `RMSNorm`

这一层的职责很单纯：做归一化，让后续计算更稳定。

你第一次读它时，不需要和 LayerNorm 做完整对比。先记一句话：

- `RMSNorm` 是每层里的“数值稳定器”

### 3. `Attention`

这是第一次阅读最容易卡住的地方。建议分两段看。

先看“输入输出”：

- 输入：`x`
- 输出：注意力结果 `output` 和缓存用的 `past_kv`

再看“内部大致流程”：

1. 线性映射出 `q / k / v`
2. reshape 成多头形式
3. 做 `q_norm` 和 `k_norm`
4. 加上 RoPE 位置编码
5. 如果有 cache，就把旧的 `k / v` 拼回来
6. 做缩放点积注意力
7. 多头结果拼回去
8. 过输出投影 `o_proj`

第一次阅读时，不要求你手推矩阵维度，只要知道这里在做“让每个位置去看其他位置”。

### 4. `FeedForward`

这一层负责把每个 token 的隐藏表示做非线性变换。

代码里你会看到三层投影：

- `gate_proj`
- `up_proj`
- `down_proj`

核心可记成一句话：

- 注意力负责“信息交互”，FFN 负责“局部特征加工”

### 5. `MiniMindBlock`

这是理解整个模型最关键的一层。

你可以把它记成两段：

1. `Attention + residual`
2. `FFN + residual`

代码里对应的是：

- `self_attn(...)`
- `hidden_states += residual`
- `hidden_states + self.mlp(...)`

第一次读完这一层，你就已经抓住 Transformer block 的骨架了。

### 6. `MiniMindModel`

这一层负责把很多个 block 串起来。

你重点看下面几个成员：

- `embed_tokens`
- `layers`
- `norm`

然后看 `forward()` 里这几步：

1. 先做 token embedding
2. 取当前位置对应的 RoPE 位置编码
3. 依次过每一层 `MiniMindBlock`
4. 最后再做一次 `norm`
5. 如果用了 MoE，就累计 `aux_loss`

第一次阅读时，建议你在纸上画一个“大框套小框”的结构图：

- `MiniMindModel`
- 多个 `MiniMindBlock`
- 每个 block 里有 `Attention` 和 `FeedForward`

### 7. `MiniMindForCausalLM`

这是训练和推理最直接用到的封装。

重点看两件事：

- `self.model = MiniMindModel(...)`
- `self.lm_head = nn.Linear(...)`

这说明它是在基础模型外面再套一层“语言建模头”。

然后重点看 `forward()`：

1. 先得到 `hidden_states`
2. 用 `lm_head` 变成 `logits`
3. 如果传了 `labels`，就计算交叉熵 loss

这里是你把“模型结构”和“训练脚本”接起来的关键。

### 8. `generate()`

第一次看 `generate()` 时，不要被循环细节吓住。先抓住它的本质：

- 每次拿当前输入做一次前向
- 取最后一个位置的 logits
- 按采样策略得到下一个 token
- 拼回输入
- 循环直到结束

这就是自回归生成。

## 用自己的话写一遍前向传播

你可以照着下面这个模板练习复述：

1. tokenizer 先把文本变成 token ids
2. `embed_tokens` 把 token ids 变成向量
3. 模型根据当前位置取出对应的 RoPE 编码
4. 每层 block 先做 `RMSNorm`
5. 进入 self-attention 计算多头注意力
6. 注意力输出和残差相加
7. 再做一次 `RMSNorm`
8. 进入 FFN 做非线性变换
9. FFN 输出和残差相加
10. 所有层结束后做最终 norm
11. `lm_head` 投影到词表维度得到 logits
12. 训练时对 logits 算交叉熵，推理时根据 logits 采样下一个 token

## 这节课先不要深挖的点

下面这些点你会在代码里看到，但第一遍不要陷进去：

- `precompute_freqs_cis()` 的数学推导
- `apply_rotary_pos_emb()` 的复数视角解释
- `past_key_values` 的缓存优化细节
- `MOEFeedForward` 的路由细节
- `scaled_dot_product_attention` 和手写 attention 分支的性能差异

你只需要知道：

- RoPE 是位置编码方案
- KV Cache 是推理提速手段
- MoE 是可选扩展分支
- flash attention 是性能优化分支

## 这一课读完后必须能回答的问题

1. `MiniMindConfig` 在整个模型里扮演什么角色？
2. 为什么 `MiniMindBlock` 可以概括成 “Attention + FFN + residual”？
3. `MiniMindModel` 和 `MiniMindForCausalLM` 的关系是什么？
4. 为什么训练时需要 `labels`，推理时不需要？
5. `generate()` 为什么本质上是在重复做前向？

## 和后面课程的衔接

读完这节课后，你应该把下面这条关系记住：

- `model/model_minimind.py` 负责定义“模型会怎么算”
- `trainer/train_pretrain.py` 和 `trainer/train_full_sft.py` 负责定义“模型拿什么数据、按什么损失去学”
- `eval_llm.py` 和 `scripts/serve_openai_api.py` 负责定义“模型学完后怎么被调用”

这就是你后面读训练脚本和推理脚本时的总坐标。
