import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Шаблон использования:")
        print("  python inference.py base <model_path>")
        print("  python inference.py lora <model_path> <adapters_path>")
        sys.exit(1)

    mode = sys.argv[1]
    model_path = sys.argv[2]

    print(f"Загрузка модели ({mode})...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True
    )

    if mode == "lora":
        if len(sys.argv) < 4:
            print("ОШИБКА: укажите путь к адаптерам", flush=True)
            sys.exit(1)
        adapters_path = sys.argv[3]
        model = PeftModel.from_pretrained(base_model, adapters_path)
        print(f"Адаптеры загружены из {adapters_path}", flush=True)
    else:
        model = base_model

    print("READY", flush=True)

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        line = line.strip()

        if line == "exit":
            print("Завершение диалога...", flush=True)
            break

        parts = line.split("|", 1)
        if len(parts) == 2:
            system_prompt, text = parts
        else:
            system_prompt = ""
            text = parts[0]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        response = response.replace('\n', '\\n')
        print(response, flush=True)