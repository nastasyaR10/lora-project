import json
import sys
import random
import os


def validate_record(record, index):
    errors = []

    for field in ["instruction", "input", "output"]:
        if field not in record:
            errors.append(f"отсутствует поле '{field}'")
        elif not isinstance(record[field], str):
            errors.append(f"поле '{field}' должно быть строкой")
        elif len(record[field].strip()) == 0:
            errors.append(f"поле '{field}' не может быть пустым")

    if "output" in record and isinstance(record["output"], str):
        output = record["output"]
        if not output.startswith("Период:"):
            errors.append("поле 'output' должно начинаться с 'Период:'")
        if "\nОбоснование:" not in output:
            errors.append("поле 'output' должно содержать '\\nОбоснование:'")

    valid_periods = [
        "Золотой век",
        "Реализм",
        "Серебряный век",
        "Советский период",
        "Современная литература",
    ]

    if "output" in record and isinstance(record["output"], str):
        output = record["output"]
        period_found = False
        for period in valid_periods:
            if f"Период: {period}" in output:
                period_found = True
        if not period_found:
            errors.append(
                f"поле 'output' содержит недопустимый период. "
                f"Допустимые: {', '.join(valid_periods)}"
            )

    if errors:
        print(f"Ошибки в записи {index}: {'; '.join(errors)}")
        return False
    return True


def load_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        print(f"Файл {filepath} пуст")
        return []

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    records = []
    for line_num, line in enumerate(content.split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Ошибка в файле {filepath}, строка {line_num}: {e}")
            sys.exit(1)
    return records


random.seed(42)
if len(sys.argv) < 4:
    print("<выходной_файл.json> <количество_файлов> <файл1.json> <файл2.json> ...")
    sys.exit(1)

output_filename = sys.argv[1]

try:
    num_files = int(sys.argv[2])
except ValueError:
    print("Количество файлов должно быть целым числом")
    sys.exit(1)

filenames = sys.argv[3:]

if len(filenames) != num_files:
    print("Некорректное количество файлов")
    sys.exit(1)

for filename in filenames:
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден")
        sys.exit(1)

all_records = []
total_errors = 0

for filename in filenames:
    records = load_json_file(filename)

    valid_records = []
    for i, record in enumerate(records, 1):
        if validate_record(record, i):
            valid_records.append(record)
        else:
            total_errors += 1

    all_records.extend(valid_records)

random.shuffle(all_records)

with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(all_records, f, ensure_ascii=False, indent=2)

print(f"Загружено записей: {len(all_records)}")
print(f"Ошибок валидации: {total_errors}")
print(f"Итоговый файл: {output_filename}")