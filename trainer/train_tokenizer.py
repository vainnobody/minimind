"""
这个脚本演示 MiniMind tokenizer 的一个最小训练流程，适合作为教学材料阅读。

阅读重点建议按下面顺序理解：
1. `get_texts`：训练语料是如何从 jsonl 文件里抽取出来的。
2. `train_tokenizer`：BPE tokenizer 如何初始化、训练、保存，以及为何要补齐特殊 token。
3. `eval_tokenizer`：如何快速检查 tokenizer 是否可用、压缩率如何、流式解码是否稳定。

注：不建议在正式训练中随意重新训练 tokenizer。MiniMind 已经提供了可复用的 tokenizer，
如果模型和 tokenizer 不匹配，即使权重结构相同，最终输出格式、分词边界和对话模板也会出现偏差。
"""

# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one.
# This script is for learning and reference only. Training models with different tokenizers
# will lead to inconsistent outputs and reduce model reusability in the community.
import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 这里默认直接读取一份精简版 SFT 数据。
# 对 tokenizer 来说，它并不关心“监督信号”，而是只关心“有哪些文本分布需要覆盖”。
# 之所以选 SFT mini 数据，是因为它同时包含中文、英文、对话角色和特殊标记，足够教学演示。
DATA_PATH = os.path.join(BASE_DIR, "data", "sft_t2t_mini.jsonl")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "dataset", "sft_t2t_mini.jsonl")

# 输出目录会保存：
# 1. `tokenizer.json`：tokenizer 的完整定义；
# 2. `vocab.json` / `merges.txt`：BPE 模型核心文件；
# 3. `tokenizer_config.json`：给 Transformers 加载时使用的配置。
TOKENIZER_DIR = "./model_learn_tokenizer/"

# 词表大小越大，通常单条文本会被切成更少 token，但模型的 embedding / lm_head 也要更大。
# 这里用 6400 做教学演示，兼顾效果与训练成本。
VOCAB_SIZE = 6400

# MiniMind 约定预留 36 个特殊 token 位。
# 其中一部分已经被明确分配给对话、视觉、音频、工具调用等场景，
# 其余位置会用 buffer token 占住，方便后续扩展时保持词表结构稳定。
SPECIAL_TOKENS_NUM = 36


def get_texts(data_path):
    """
    从 jsonl 数据中抽取可用于 tokenizer 训练的纯文本。

    数据格式预期类似：
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

    这里把一条样本里的多轮 `content` 用换行拼起来，原因是：
    1. tokenizer 训练只需要“文本流”，不需要标签；
    2. 保留换行可以让 BPE 看到更接近真实对话的边界形态；
    3. 对教学数据来说，这样实现最直接，也足够稳定。
    """
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            # 这里只取前 10000 行，避免教学演示时训练过慢。
            # 真正大规模训练 tokenizer 时，通常会喂入更大、更干净、分布更全面的文本集合。
            if i >= 10000:
                break
            try:
                data = json.loads(line)
                # conversations 里每个 turn 只抽取 content 字段。
                # role 暂时不参与拼接，因为 role 控制通常由 chat template 和特殊 token 负责。
                contents = [
                    item.get("content")
                    for item in data.get("conversations", [])
                    if item.get("content")
                ]
                if contents:
                    yield "\n".join(contents)
            except json.JSONDecodeError:
                # 教学数据里如果偶尔混入坏行，这里选择跳过而不是让整个训练中断。
                continue


def train_tokenizer(
    data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM
):
    """
    训练并保存一个基于 BPE 的 tokenizer。

    关键流程：
    1. 初始化一个空的 BPE tokenizer；
    2. 指定 ByteLevel 预分词器，保证任意字节序列都可编码；
    3. 注入 MiniMind 需要的特殊 token；
    4. 用语料迭代器训练 merges 和 vocab；
    5. 保存 tokenizer 文件，并补写 Transformers 所需配置。
    """
    # BPE（Byte Pair Encoding）会从字节/字符级别开始，不断合并高频片段，
    # 最终得到兼顾词表大小和压缩率的子词表示。
    tokenizer = Tokenizer(models.BPE())

    # ByteLevel 的好处是鲁棒：无论是中文、英文、标点还是罕见字符，
    # 最差都能回退到字节层面编码，不容易出现“完全无法切分”的问题。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 这一组 token 是模型明确依赖的“核心特殊 token”。
    # 它们通常不应该被 BPE 再拆开，否则对话边界、多模态占位符等协议会被破坏。
    special_tokens_list = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|audio_pad|>",
        "<tts_pad>",
        "<tts_text_bos>",
        "<tts_text_eod>",
        "<tts_text_bos_single>",
    ]

    # 这一组 token 更多偏向“训练数据协议”：
    # 比如工具调用、工具返回、显式思维链边界等。
    # 这些 token 需要出现在词表里，但不一定都要在 Transformers 侧被当作 special token 处理。
    additional_tokens_list = [
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
    ]

    # 为了让总特殊 token 数固定到 `SPECIAL_TOKENS_NUM`，
    # 把剩余槽位用 buffer token 预留出来，方便未来扩展。
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens

    # BpeTrainer 负责真正的词表统计与合并规则学习。
    # `initial_alphabet=ByteLevel.alphabet()` 很关键：
    # 它确保所有基础字节都在初始字母表里，从而维持 ByteLevel tokenizer 的可逆性。
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=all_special_tokens,
    )

    # 这里传入的是 Python 生成器，`train_from_iterator` 会边消费边训练，
    # 不需要先把所有文本一次性读入内存。
    texts = get_texts(data_path)
    # breakpoint()
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 训练完成后，解码器也要与 ByteLevel 对齐。
    # 否则 encode/decode 可能出现空格、字节还原不一致的问题。
    tokenizer.decoder = decoders.ByteLevel()

    # 再次把核心特殊 token 注册到 tokenizer 中。
    # 这样它们在 encode 时会被整体识别，不会被拆成普通子词。
    tokenizer.add_special_tokens(special_tokens_list)

    os.makedirs(tokenizer_dir, exist_ok=True)

    # `tokenizer.json` 保存完整 tokenizer 图结构；
    # `tokenizer.model.save(...)` 会额外导出 BPE 的 vocab / merges 文件，
    # 方便 Hugging Face 生态中的其他工具识别。
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)

    # `tokenizer.json` 里会记录所有 added_tokens。
    # 这里手动调整一个细节：
    # 只有 `special_tokens_list` 中的 token 被标成真正的 special token，
    # `<think>` / `<tool_call>` 等虽然保留为独立 token，但不强制在解码或跳过 special token 时按同样规则处理。
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    for token_info in tokenizer_data.get("added_tokens", []):
        if token_info["content"] not in special_tokens_list:
            token_info["special"] = False
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    # 下面这段配置主要服务于 `PreTrainedTokenizerFast.from_pretrained(...)`。
    # Transformers 读取目录时，会综合 tokenizer 文件和 tokenizer_config.json
    # 还原一个可直接推理、套模板、处理特殊 token 的 tokenizer 对象。
    added_tokens_decoder = {}
    for token in all_special_tokens:
        idx = tokenizer.token_to_id(token)
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True if token in special_tokens_list else False,
        }

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [
            t for t in special_tokens_list if t not in ["<|endoftext|>"]
        ],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "unk_token": "<|endoftext|>",
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        # `chat_template` 定义“消息列表 -> 训练/推理 prompt 字符串”的映射规则。
        # 这一步非常重要：即使 tokenizer 能正常切词 ，如果模板不一致，模型学习到的对话格式也会错位。
        # 当前模板支持：
        # 1. system / user / assistant 多角色对话；
        # 2. <think> 显式思维内容；
        # 3. <tool_call> / <tool_response> 工具调用链路；
        # 4. `add_generation_prompt` 与 `open_thinking` 推理时的起始提示。
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        "tokenizer_class": "PreTrainedTokenizerFast",
    }

    # 写出 Hugging Face 风格配置后，这个目录就能被 `AutoTokenizer.from_pretrained(...)`
    # 直接加载，用法会和常见开源模型保持一致。
    with open(
        os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer training completed.")


def eval_tokenizer(tokenizer_dir):
    """
    训练完成后的最小验收脚本。

    这里不是做严格 benchmark，而是回答三个最实际的问题：
    1. tokenizer 能不能被 Transformers 正常加载；
    2. chat template 套出来的 prompt 能不能稳定 encode / decode；
    3. 常见中英混合文本的切分压缩率和流式解码表现是否合理。
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 这组测试消息故意很小，便于快速查看模板展开结果。
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自月球"},
        {"role": "user", "content": "你到底来自哪里？"},
        {"role": "assistant", "content": "我来自地球"},
    ]

    # `apply_chat_template(..., tokenize=False)` 会先把消息排版成完整 prompt 字符串，
    # 方便我们先检查“模板结果对不对”，再检查“切词结果对不对”。
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("-" * 100)
    print(new_prompt)
    print("-" * 100)
    print("tokenizer词表长度：", len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    print("encoder长度：", len(model_inputs["input_ids"]))
    response = tokenizer.decode(model_inputs["input_ids"], skip_special_tokens=False)
    print("decoder一致性：", response == new_prompt, "\n")
    print("-" * 100)
    print("压缩率测试（Chars/Tokens）：")
    test_texts = [
        # 中文样本 (约200字)
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。",
        # 英文样本 (约200词/字符)
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
        # 混合样本
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",
    ]

    total_compression = 0
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)
        token_count = len(encoded)
        char_count = len(text)
        compression_ratio = char_count / token_count
        total_compression += compression_ratio
        print(
            f"样本 {i+1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}"
        )

    print(f"平均压缩率: {total_compression / len(test_texts):.2f}")
    print("-" * 100)
    print("流式解码（字节缓冲）测试：")
    input_ids = model_inputs["input_ids"]
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        # ByteLevel tokenizer 在逐 token 解码时，可能先得到半个字节片段，
        # 此时字符串里会出现替代符 `\ufffd`。只有当缓存凑成完整字符后再打印，
        # 才更接近实际流式输出系统里的行为。
        if current_decode and "\ufffd" not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [
                tokenizer.convert_ids_to_tokens(int(t))
                for t in (
                    token_cache if isinstance(token_cache, list) else [token_cache]
                )
            ]
            print(
                f"Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}"
            )
            token_cache = []


if __name__ == "__main__":
    # 直接运行脚本时，先训练，再做一次最小验证。
    # 这样课堂演示只需执行一个文件，就能把“训练 + 验收”完整跑通。
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
