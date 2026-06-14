import sys
import json
import os
import re
import importlib.util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer


class RussianTokenizer(Tokenizer):
    def tokenize(self, text):
        return re.findall(r'[а-яёА-ЯЁa-zA-Z]+', text.lower())


def load_extractor(script_path):
    if not os.path.exists(script_path):
        print(f"ОШИБКА: файл {script_path} не найден")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location("extractor_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, "extract"):
        print("ОШИБКА: функция extract не найдена в файле")
        sys.exit(1)
    
    return module.extract


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Использование:")
        print("  python compute_metrics.py base <model_path> <test_data> <system_prompt> <extractor_script> <metrics>")
        print("  python compute_metrics.py lora <model_path> <adapters_path> <test_data> <system_prompt> <extractor_script> <metrics>")
        sys.exit(1)

    mode = sys.argv[1]
    model_path = sys.argv[2]
    
    if mode == "lora":
        if len(sys.argv) < 8:
            print("Для LoRA укажите путь к адаптерам")
            sys.exit(1)
        adapters_path = sys.argv[3]
        test_data_path = sys.argv[4]
        system_prompt = sys.argv[5] if sys.argv[5] != "none" else ""
        extractor_script = sys.argv[6]
        metrics_str = sys.argv[7]
    else:
        adapters_path = None
        test_data_path = sys.argv[3]
        system_prompt = sys.argv[4] if sys.argv[4] != "none" else ""
        extractor_script = sys.argv[5]
        metrics_str = sys.argv[6]

    selected_metrics = [m.strip() for m in metrics_str.split(",")]

    print(f"Загрузка функции для извлечения ответа из {extractor_script}...", flush=True)
    extract_fn = load_extractor(extractor_script)

    print("Загрузка тестовых данных...", flush=True)
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Загружено примеров: {len(test_data)}", flush=True)

    print("Загрузка модели...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cpu",
        trust_remote_code=True, local_files_only=True
    )

    if mode == "lora":
        model = PeftModel.from_pretrained(base_model, adapters_path)
        print(f"Адаптеры загружены из {adapters_path}", flush=True)
    else:
        model = base_model

    print("Вычисление ответов модели...", flush=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=RussianTokenizer())
    y_true = []
    y_pred = []
    rouge_scores = []
    
    total = len(test_data)
    print('')
    for i, item in enumerate(test_data):
        input_text = item.get("input", "")
        instruction = item.get("instruction", "")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"{instruction}\nТекст: {input_text}"})
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        true_class = extract_fn(item['output'])
        pred_class = extract_fn(response)
        
        y_true.append(true_class)
        y_pred.append(pred_class if pred_class else "не определён")

        score = scorer.score(item['output'], response)
        rouge_scores.append(score["rougeL"].fmeasure)

        print(f'	{i} / {total} ')

    print()

    print("\nРезультаты:", flush=True)
    
    if "accuracy" in selected_metrics or "f1" in selected_metrics:
        if "accuracy" in selected_metrics and "f1" in selected_metrics:
            print(f"\nОтчёт по классам:")
            print(classification_report(y_true, y_pred, zero_division=0))

        if "accuracy" in selected_metrics:
            acc = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {acc:.4f}")
        
        if "f1" in selected_metrics:
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            print(f"F1: {f1_weighted:.4f}")

    
    if "rouge_l" in selected_metrics:
        rouge_l = sum(rouge_scores) / len(rouge_scores)
        print(f"ROUGE-L: {rouge_l:.4f}")
