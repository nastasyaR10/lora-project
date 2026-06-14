import torch
import json
import os
import time
import math
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

import signal

parser = argparse.ArgumentParser(description="Обучение LoRA")

parser.add_argument("model_path", type=str, help="Путь к базовой модели")
parser.add_argument("adapters_output", type=str, help="Директория для сохранения адаптеров")
parser.add_argument("output_dir", type=str, help="Директория для логов и графиков")
parser.add_argument("train_data", type=str, help="Путь к файлу с обучающими данными (.json)")
parser.add_argument("test_data", type=str, help="Путь к файлу с тестовыми данными (.json)")
parser.add_argument("epochs", type=int, help="Количество эпох")
parser.add_argument("batch_size", type=int, help="Размер батча")
parser.add_argument("learning_rate", type=float, help="Скорость обучения")
parser.add_argument("lora_r", type=int, help="LoRA ранг")
parser.add_argument("lora_alpha", type=int, help="LoRA alpha")
parser.add_argument("lora_dropout", type=float, help="LoRA dropout")
parser.add_argument("target_modules", type=str, help="Модули для LoRA через запятую")
parser.add_argument("system_prompt", type=str, help="Системный промпт для модели")
parser.add_argument("device", type=str, choices=["cpu", "cuda"], help="Устройство для обучения (cpu или cuda)")

args = parser.parse_args()


def signal_handler(sig, frame):
    raise KeyboardInterrupt()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGBREAK, signal_handler)

if args.device == "cuda" and not torch.cuda.is_available():
    print("ОШИБКА: CUDA не доступна")
    sys.exit(1)

TARGET_MODULES = [m.strip() for m in args.target_modules.split(",")]

LOGS_DIR = os.path.join(args.output_dir, "logs")
PLOTS_DIR = os.path.join(args.output_dir, "plots")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(args.adapters_output, exist_ok=True)

# Перенаправление вывода и в stdout и в файл одновременно
class TeeLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("")
    
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message)
            f.flush()
    
    def flush(self):
        self.terminal.flush()
        with open(self.log_file, "a") as f:
            f.flush()

sys.stdout = TeeLogger(os.path.join(LOGS_DIR, "training_log.txt"))

print(f"Старт: {time.strftime('%H:%M:%S')}")
print(f"r={args.lora_r}, alpha={args.lora_alpha}, epochs={args.epochs}, lr={args.learning_rate}")

print("[1] Загрузка данных...")

with open(args.train_data, "r", encoding="utf-8") as f:
    train_raw = json.load(f)
with open(args.test_data, "r", encoding="utf-8") as f:
    test_raw = json.load(f)

print(f"  Обучающих примеров: {len(train_raw)}")
print(f"  Тестовых примеров: {len(test_raw)}")

train_dataset = Dataset.from_list(train_raw)
test_dataset = Dataset.from_list(test_raw)

print("[2] Загрузка модели...")

tokenizer = AutoTokenizer.from_pretrained(
    args.model_path, trust_remote_code=True, local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float32 if args.device == "cpu" else torch.float16,
    device_map=args.device,
    trust_remote_code=True,
    local_files_only=True
)
model.config.pad_token_id = tokenizer.pad_token_id
print(f"  Параметров: {model.num_parameters():,}")

print("[3] Подготовка данных...")

def format_chat(example):
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": f"{example['instruction']}\nТекст: {example['input']}"})
    messages.append({"role": "assistant", "content": example['output']})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

train_chat = train_dataset.map(format_chat)
test_chat = test_dataset.map(format_chat)

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    tokenized["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in ids]
        for ids in tokenized["input_ids"]
    ]
    return tokenized

train_tokenized = train_chat.map(
    tokenize_function, batched=True, remove_columns=train_chat.column_names
)
test_tokenized = test_chat.map(
    tokenize_function, batched=True, remove_columns=test_chat.column_names
)
print(f"  Размер train: {len(train_tokenized)}, test: {len(test_tokenized)}")

print("[4] Настройка LoRA...")
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=TARGET_MODULES,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Обучаемых: {trainable:,} ({trainable/model.num_parameters()*100:.2f}%)")

print("[5] Обучение...")
training_args = TrainingArguments(
    output_dir="./models/lora_checkpoints",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=1,
    learning_rate=args.learning_rate,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    optim="adamw_torch",
    dataloader_num_workers=0,
    fp16=(args.device == "cuda"),
    load_best_model_at_end=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

losses = []
eval_losses = []
perplexities = []
interrupted = False
start_time = time.time()

print(f"\n{'Эпоха':<8} {'Train Loss':<12} {'Eval Loss':<12} {'Perplexity':<12} {'Время(мин)':<10}")
print("-" * 60)

try:
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Обучение
        train_result = trainer.train(resume_from_checkpoint=False)
        train_loss = train_result.training_loss
        losses.append(train_loss)

        # Оценка на тестовых данных
        eval_result = trainer.evaluate()
        eval_loss = eval_result.get("eval_loss", 0)
        eval_losses.append(eval_loss)
        perplexity = math.exp(eval_loss) if eval_loss > 0 else 0
        perplexities.append(perplexity)

        epoch_time = time.time() - epoch_start

        print(f"{epoch:<8} {train_loss:<12.4f} {eval_loss:<12.4f} {perplexity:<12.2f} {epoch_time/60:<10.1f}")

        # Обновление JSON с метриками после каждой эпохи
        with open(os.path.join(LOGS_DIR, "training_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "epochs": list(range(1, len(losses) + 1)),
                "train_losses": losses,
                "eval_losses": eval_losses,
                "perplexities": perplexities
            }, f, indent=2, ensure_ascii=False)

except KeyboardInterrupt:
    print(f"\nПрервано пользователем на эпохе {epoch}")
    interrupted = True

total_time = time.time() - start_time
print(f"\nПройдено эпох: {len(losses)}/{args.epochs}")
print(f"Затрачено времени: {total_time/60:.1f} мин")

print("Сохранение адаптеров...")
model.save_pretrained(args.adapters_output)
tokenizer.save_pretrained(args.adapters_output)
print(f"Адаптеры сохранены: {args.adapters_output}")

if losses:
    print("Сохранение графиков...")
    epochs_range = list(range(1, len(losses) + 1))

    # Train Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, losses, 'b-o', linewidth=2, markersize=8)
    plt.title(f'Training Loss ({len(losses)} epochs)', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs_range)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_plot.png'), dpi=150)
    plt.close()

    # Perplexity
    if perplexities:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, perplexities, 'c-s', linewidth=2, markersize=8)
        plt.title(f'Perplexity ({len(perplexities)} epochs)', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.grid(True, alpha=0.3)
        plt.xticks(epochs_range)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'perplexity_plot.png'), dpi=150)
        plt.close()

    print(f"Графики: {PLOTS_DIR}/loss_plot.png, {PLOTS_DIR}/perplexity_plot.png")
    print(f"Метрики: {LOGS_DIR}/training_metrics.json")
else:
    print("Нет данных для графиков")
