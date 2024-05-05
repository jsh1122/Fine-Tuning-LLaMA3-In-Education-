# 完整图片可下载本仓库Education.pdf查看

# 如何使用

## 下载微调好的教育模型

下载GGUF模型（点击files）[Starxx/LLaMa3-Fine-Tuning-Classical-GGUF · HF Mirror (hf-mirror.com)](https://hf-mirror.com/Starxx/LLaMa3-Fine-Tuning-Classical-GGUF)

![image-20240505231420957](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505231420957.png)

## GPT4ALL

下载百度网盘中的maintenancetool软件安装GPT4ALL（一直下一步完成安装）

![image-20240505230457670](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505230457670.png)

安装结束后打开AppData文件夹（需打开隐藏目录）

![image-20240505230814571](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505230814571.png)

在将下载好的GGUF模型放在如下图所示的路径里

![image-20240505231113435](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505231113435.png)

打开GPT4ALL点击设置

![image-20240505231755665](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505231755665.png)

将设备改为CPU驱动，修改模型路径

![image-20240505231720215](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505231720215.png)

然后选择加载模型，进行相关提问操作等

![image-20240505232515540](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505232515540.png)

# 仓库地址

## 模型

https://huggingface.co/Starxx/LLaMa3-Fine-Tuning-Classical-GGUF/tree/main

![image-20240505230300993](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505230300993.png)

# 微调过程

![image-20240505224432338](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505224432338.png)

```python
%%capture
# 安装 Unsloth、Xformers （Flash Attention） 和所有其他软件包！
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
```

![image-20240505225103798](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505225103798.png)

```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # 任选其选！我们在内部自动支持 RoPE 扩展！
dtype = None # 无自动检测。Float16 用于 Tesla T4、V100，Bfloat16 用于 Ampere+
load_in_4bit = True # 使用 4 位量化来减少内存使用量。可以是 False。

# 我们支持 4 位预量化模型，下载速度提高 4 倍 + 无 OOM。
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Gemma 7b 的指导版本
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Gemma 2b 的 Instruct 版本
    "unsloth/llama-3-8b-bnb-4bit", # [新] Llama-3
] # 更多模型在 https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Starxx/LLaMa3-Fine-Tuning-Math",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = “hf_...”， # 如果使用像 meta-llama/Llama-2-7b-hf 这样的门控模型，则使用一个
)
```

![image-20240505225130488](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505225130488.png)

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # 选择任意数字> 0 ！建议 8、16、32、64、128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # 支持任何，但 = 0 是优化的
    bias = "none",    # 支持任何，但 = “none” 已优化
    # [新增]“unsloth”使用的 VRAM 减少了 30%，适合 2 倍大的批量大小！
    use_gradient_checkpointing = "unsloth", # True 或“unsloth”表示很长的上下文
    random_state = 3407,
    use_rslora = False,  # 我们支持排名稳定的 LoRA
    loftq_config = None, # 和 LoftQ
)
```

![image-20240505225156070](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505225156070.png)

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # 必须添加EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["info"]
    inputs       = examples["modern"]
    outputs      = examples["classical"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必须添加EOS_TOKEN，否则你们这一代人将永远持续下去！
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("xmj2002/Chinese_modern_classical", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

![image-20240505225217118](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505225217118.png)

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # 可以使短序列的训练速度提高 5 倍。
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```

![image-20240505225253822](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505225253822.png)

![image-20240505225307867](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505225307867.png)

```python
trainer_stats = trainer.train()
```

![image-20240505225354964](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505225354964.png)

```python
# alpaca_prompt = 从上面复制
FastLanguageModel.for_inference(model) # 将原生推理速度提高 2 倍
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
```

![image-20240505225418300](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505225418300.png)

```python
# model.save_pretrained("lora_model") # Local saving
# tokenizer.save_pretrained("lora_model")
model.push_to_hub("Starxx/LLaMa3-Fine-Tuning-Classical", token = "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") # Online saving
tokenizer.push_to_hub("Starxx/LLaMa3-Fine-Tuning-Classical", token = "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") # Online saving
```

![image-20240505225501874](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505225501874.png)

![image-20240505225533767](C:\Users\21453\AppData\Roaming\Typora\typora-user-images\image-20240505225533767.png)

```python
# Save to q4_k_m GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if True: model.push_to_hub_gguf("Starxx/LLaMa3-Fine-Tuning-Classical-GGUF", tokenizer, quantization_method = "q4_k_m", token = "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")
```

