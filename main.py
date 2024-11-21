from datetime import datetime
from pytz import timezone
import time
import os
import fire
import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer
import json

from model_utils.modeling_llama import LlamaForCausalLM

from main_utils import (
    load_jsonl_examples,
    get_cosine_lr_decay_fn,
    get_grad_norm,
    save_checkpoint,
    get_last_ckpt_idx
)

# 全局配置
TIMEZONE = timezone('EST')
DATE = str(datetime.now(tz=TIMEZONE)).split()[0]
MODEL_SIZE = '7b'
PROJECT_NAME = f'amber_{MODEL_SIZE}'
RUN_NAME = f'pretraining_{MODEL_SIZE}_{DATE}'
HF_MODEL_NAME_OR_PATH = f'huggyllama/llama-{MODEL_SIZE}'
WORKDIR = f'workdir_{MODEL_SIZE}'

LEARNING_RATE = 3e-4
LR_SCHEDULE_TYPE = 'cosine'
END_LEARNING_RATE = 3e-5
WARMUP_GRAD_STEPS = 2000
GRAD_NORM_CLIP = 1.0
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
PRECISION = 'bf16'  # 使用 bf16 精度
RANDOM_SEED = 11111

TRAIN_DATA_DIR = './data'
TRAIN_EXAMPLES_PER_CHUNK = 1706976
N_CHUNKS = 360
MAX_SEQ_LENGTH = 1024  # 减少序列长度

config_path = "./config.json"  # 模型参数配置
with open(config_path, "r") as f:
    config_dict = json.load(f)

tokenizer_path = "./tokenizer"

# 数据处理函数


def collate_fn(examples, device):
    token_ids = torch.tensor(
        [example['token_ids'] for example in examples], device=device)
    return {'input_ids': token_ids[:, :-1], 'labels': token_ids[:, 1:]}

# 单块训练逻辑


def train_chunk(
        tokenizer,
        model,
        optimizer,
        lr_schedule_fn,
        examples,
        per_device_batch_size,
        accumulate_grad_batches,
        chunk_idx):
    step = chunk_idx * (len(examples) // per_device_batch_size)

    example_batch_idxes = tqdm.trange(
        0, len(examples), per_device_batch_size,
        desc=f'Training chunk {chunk_idx}')
    for i in example_batch_idxes:
        t0 = time.time()

        lr = lr_schedule_fn(step)
        step += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        is_accumulating = (step % accumulate_grad_batches != 0)

        batch = collate_fn(
            examples=examples[i:i + per_device_batch_size], device="cuda")
        input_ids, labels = batch['input_ids'], batch['labels']

        logits = model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        if not is_accumulating:
            grad_norm = get_grad_norm(model=model)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        log = {
            'loss': loss.item(),
            'learning_rate': lr,
            'step': step,
            'speed(#tok/s)': int(input_ids.numel() / (time.time() - t0))
        }
        if not is_accumulating:
            log['grad_norm'] = grad_norm

        example_batch_idxes.set_postfix(log)

    save_checkpoint(
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        save_dir=f'{WORKDIR}/ckpt_{chunk_idx}')

# 主函数


def main(per_device_batch_size=2, accumulate_grad_batches=16):
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)

    # 初始化 tokenizer 和 model
    # tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME_OR_PATH)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # config = AutoConfig.from_pretrained(HF_MODEL_NAME_OR_PATH)
    config = AutoConfig.from_dict(config_dict)

    # 调整模型结构以适应单卡显存
    config.update({
        "hidden_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "intermediate_size": 8192,
        "max_position_embeddings": MAX_SEQ_LENGTH,
        "vocab_size": 16000
    })

    model = LlamaForCausalLM(config=config)
    model = model.to("cuda")
    model.gradient_checkpointing_enable()  # 启用激活检查点

    # 初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2)
    )

    # 获取学习率调度函数
    total_steps = TRAIN_EXAMPLES_PER_CHUNK // per_device_batch_size * N_CHUNKS
    lr_schedule_fn = get_cosine_lr_decay_fn(
        total_steps=total_steps,
        warmup_steps=WARMUP_GRAD_STEPS * accumulate_grad_batches,
        learning_rate=LEARNING_RATE,
        end_learning_rate=END_LEARNING_RATE)

    # 检查点恢复
    last_ckpt_idx = get_last_ckpt_idx(workdir=WORKDIR)
    if last_ckpt_idx != -1:
        checkpoint = torch.load(f'{WORKDIR}/ckpt_{last_ckpt_idx}/fabric_ckpt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 开始训练
    for chunk_idx in range(last_ckpt_idx + 1, N_CHUNKS):
        examples = load_jsonl_examples(
            filename=f'{TRAIN_DATA_DIR}/train_{chunk_idx}.jsonl',
            n_examples=TRAIN_EXAMPLES_PER_CHUNK,
            shuffle=True)

        train_chunk(
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            lr_schedule_fn=lr_schedule_fn,
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            chunk_idx=chunk_idx)


if __name__ == '__main__':
    fire.Fire(main)
