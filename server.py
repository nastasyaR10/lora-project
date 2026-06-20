from flask import Flask, send_from_directory, jsonify, request
import subprocess
import threading
import signal
import os
import sys
import time
import json
import atexit
from pathlib import Path
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', static_url_path='')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logs = []
training_logs = []
training_started = False

training_process = None
is_training = False
training_model_path = None

inference_process = None
is_inference = False
inference_model_path = None

compute_process = None
compute_logs = []

download_process = None
is_downloading = False


def add_log(message, to_training=False):
    logs.append(message)
    if len(logs) > 1000:
        logs.pop(0)
    if to_training:
        training_logs.append(message)
        if len(training_logs) > 1000:
            training_logs.pop(0)
    print(message)


def is_busy():
    if is_training:
        return True
    if is_inference:
        return True
    if compute_process and compute_process.poll() is None:
        return True
    if download_process and download_process.poll() is None:
        return True
    return False


def check_model_conflict(new_model_path, exclude_training=False, exclude_inference=False):
    conflicts = []
    if not exclude_training and is_training and training_model_path:
        if os.path.normpath(new_model_path) == os.path.normpath(training_model_path):
            conflicts.append("обучения")
    if not exclude_inference and is_inference and inference_model_path:
        if os.path.normpath(new_model_path) == os.path.normpath(inference_model_path):
            conflicts.append("инференса")
    return conflicts


def monitor_inference():
    global is_inference, inference_model_path, inference_process
    try:
        inference_process.wait()
    except:
        pass
    is_inference = False
    inference_model_path = None
    inference_process = None


def cleanup_processes():
    for proc in [training_process, inference_process, compute_process, download_process]:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass

atexit.register(cleanup_processes)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/log')
def get_log():
    running_process = None
    if is_training:
        running_process = 'обучение'
    elif is_inference:
        running_process = 'инференс'
    elif compute_process and compute_process.poll() is None:
        running_process = 'вычисление метрик'
    elif download_process and download_process.poll() is None:
        running_process = 'загрузка модели'
    
    return jsonify({
        'log': '\n'.join(logs),
        'training_log': '\n'.join(training_logs),
        'is_training': is_training,
        'is_inference': is_inference,
        'is_computing': compute_process is not None,
        'is_downloading': download_process is not None,
        'running_process': running_process,
        'training_finished': training_started and not is_training and len(training_logs) > 0
    })


@app.route('/clear', methods=['POST'])
def clear_log():
    if is_training or is_inference:
        return jsonify({'error': 'Нельзя очистить лог во время выполнения процессов'}), 400
    logs.clear()
    training_logs.clear()
    return jsonify({'status': 'ok'})


@app.route('/upload', methods=['POST'])
def upload_files():
    train_file = request.files.get('train')
    test_file = request.files.get('test')
    paths = {}
    if train_file and train_file.filename:
        filename = secure_filename(train_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"train_{filename}")
        train_file.save(filepath)
        paths['train'] = filepath
    if test_file and test_file.filename:
        filename = secure_filename(test_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"test_{filename}")
        test_file.save(filepath)
        paths['test'] = filepath
    return jsonify({'status': 'ok', 'paths': paths})


@app.route('/upload_test', methods=['POST'])
def upload_test_file():
    file = request.files.get('file')
    if file and file.filename:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"metrics_{filename}")
        file.save(filepath)
        return jsonify({'status': 'ok', 'path': filepath})
    return jsonify({'error': 'Файл не загружен'}), 400


@app.route('/start', methods=['POST'])
def start_training():
    global training_process, is_training, training_model_path, training_started
    if is_training:
        return jsonify({'error': 'Обучение уже запущено'}), 400
    if is_busy():
        return jsonify({'error': 'Дождитесь завершения текущего процесса'}), 400
    config = request.json
    model_path = config['model_path']
    conflicts = check_model_conflict(model_path, exclude_training=True)
    if conflicts:
        return jsonify({'error': f'Модель уже используется в процессе: {", ".join(conflicts)}'}), 400

    train_data = config.get('train_path', str(Path("data") / "train.json"))
    test_data = config.get('test_path', str(Path("data") / "test.json"))

    cmd = [
        sys.executable, "-u", "train.py",
        model_path, config['adapters_path'], config['output_path'],
        train_data, test_data,
        str(config['epochs']), str(config['batch_size']), str(config['lr']),
        str(config['r']), str(config['alpha']), str(config['dropout']),
        config['modules'], config['system_prompt'], config['device']
    ]

    training_started = True
    add_log("=" * 50, to_training=True)
    add_log(f"Старт обучения: {time.strftime('%H:%M:%S')}", to_training=True)
    add_log(f"Модель: {model_path}", to_training=True)
    add_log(f"r={config['r']}, alpha={config['alpha']}, epochs={config['epochs']}, lr={config['lr']}", to_training=True)
    add_log("=" * 50, to_training=True)

    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        training_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding='utf-8', errors='replace',
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, env=env
        )
        is_training = True
        training_model_path = model_path

        def read_output():
            global is_training, training_model_path
            try:
                for line in training_process.stdout:
                    add_log(line.rstrip(), to_training=True)
                training_process.wait()
                exit_code = training_process.returncode
                if exit_code == 0:
                    add_log("\nОбучение завершено успешно", to_training=True)
                else:
                    add_log(f"\nПроцесс завершен с кодом: {exit_code}", to_training=True)
            except Exception as e:
                add_log(f"Ошибка при чтении вывода: {str(e)}", to_training=True)
            finally:
                is_training = False
                training_model_path = None

        threading.Thread(target=read_output, daemon=True).start()
        return jsonify({'status': 'ok'})
    except Exception as e:
        add_log(f"ОШИБКА: {str(e)}", to_training=True)
        is_training = False
        training_model_path = None
        return jsonify({'error': str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop_training():
    global is_training, training_model_path
    if not is_training or training_process is None:
        return jsonify({'status': 'no_process'})
    add_log("\nОстановка обучения...", to_training=True)
    try:
        training_process.send_signal(signal.CTRL_BREAK_EVENT)
    except Exception as e:
        add_log(f"Ошибка отправки сигнала: {e}", to_training=True)
    try:
        training_process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        try:
            training_process.terminate()
            training_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            training_process.kill()
            training_process.wait()
    is_training = False
    training_model_path = None
    add_log("Обучение остановлено", to_training=True)
    return jsonify({'status': 'ok'})


@app.route('/infer', methods=['POST'])
def run_inference():
    global inference_process, is_inference, inference_model_path

    config = request.json
    mode = config.get('mode', 'base')
    model_path = config['model_path']
    adapter_path = config.get('adapter_path')
    system_prompt = config.get('system_prompt', '')
    text = config.get('text', '')

    if not text.strip():
        return jsonify({'error': 'Пустой текст для анализа'}), 400

    if text.strip().lower() == 'exit':
        if not is_inference or inference_process is None:
            return jsonify({'error': 'Нет активного диалога'}), 400
        add_log("Завершение диалога...")
        try:
            inference_process.stdin.write('exit\n')
            inference_process.stdin.flush()
        except:
            pass
        try:
            inference_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                inference_process.terminate()
                inference_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                inference_process.kill()
                inference_process.wait()
        is_inference = False
        inference_model_path = None
        inference_process = None
        add_log("Диалог завершен.")
        return jsonify({'status': 'stopped'})

    if is_inference and inference_process is not None:
        try:
            message = f"{system_prompt}|{text}"
            inference_process.stdin.write(message + '\n')
            inference_process.stdin.flush()
            response_line = inference_process.stdout.readline().rstrip()
            add_log(response_line)
            return jsonify({'status': 'ok', 'response': response_line})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    if is_busy():
        return jsonify({'error': 'Дождитесь завершения текущего процесса'}), 400

    conflicts = check_model_conflict(model_path, exclude_inference=True)
    if conflicts:
        return jsonify({'error': f'Модель уже используется в процессе: {", ".join(conflicts)}'}), 400

    if mode == 'base':
        cmd = [sys.executable, "-u", "inference.py", "base", model_path]
    elif mode == 'lora':
        if not adapter_path:
            return jsonify({'error': 'Укажите путь к адаптеру для LoRA режима'}), 400
        cmd = [sys.executable, "-u", "inference.py", "lora", model_path, adapter_path]
    else:
        return jsonify({'error': f'Неизвестный режим: {mode}'}), 400

    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        inference_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE, text=True, bufsize=1,
            encoding='utf-8', errors='replace',
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, env=env
        )
        is_inference = True
        inference_model_path = model_path
        threading.Thread(target=monitor_inference, daemon=True).start()

        while True:
            line = inference_process.stdout.readline().rstrip()
            if line == "READY":
                break

        message = f"{system_prompt}|{text}"
        inference_process.stdin.write(message + '\n')
        inference_process.stdin.flush()
        response_line = inference_process.stdout.readline().rstrip()

        return jsonify({'status': 'ok', 'response': response_line})
    except Exception as e:
        add_log(f"ОШИБКА: {str(e)}")
        is_inference = False
        inference_model_path = None
        inference_process = None
        return jsonify({'error': str(e)}), 500


@app.route('/compute_metrics', methods=['POST'])
def compute_metrics():
    global compute_process, compute_logs
    if is_busy():
        return jsonify({'error': 'Дождитесь завершения текущего процесса'}), 400
    config = request.json
    mode = config.get('mode', 'base')
    model_path = config['model_path']
    adapter_path = config.get('adapter_path')
    test_path = config.get('test_path')
    system_prompt = config.get('system_prompt', '')
    metrics = config.get('metrics', '')
    extractor_code = config.get('extractor_code', '')

    if not test_path:
        return jsonify({'error': 'Не указан файл с тестовыми данными'}), 400

    extractor_path = os.path.join(UPLOAD_FOLDER, f"extractor_{int(time.time())}.py")
    with open(extractor_path, 'w', encoding='utf-8') as f:
        f.write(extractor_code)

    sp = system_prompt if system_prompt.strip() else "none"

    if mode == 'base':
        cmd = [sys.executable, "-u", "compute_metrics.py", "base", model_path, test_path, sp, extractor_path, metrics]
    else:
        cmd = [sys.executable, "-u", "compute_metrics.py", "lora", model_path, adapter_path, test_path, sp, extractor_path, metrics]

    compute_logs = []
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        compute_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding='utf-8', errors='replace',
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, env=env
        )

        def read_compute():
            global compute_process, compute_logs
            try:
                for line in compute_process.stdout:
                    line = line.rstrip()
                    if 'UserWarning' in line or 'warnings.warn' in line or 'configuration_utils.py' in line:
                        continue
                    compute_logs.append(line)
                compute_process.wait()
            except:
                pass
            finally:
                compute_process = None

        threading.Thread(target=read_compute, daemon=True).start()
        return jsonify({'status': 'ok'})
    except Exception as e:
        compute_process = None
        compute_logs = []
        return jsonify({'error': str(e)}), 500


@app.route('/compute_log')
def get_compute_log():
    return jsonify({'log': '\n'.join(compute_logs), 'running': compute_process is not None})


@app.route('/stop_compute', methods=['POST'])
def stop_compute():
    global compute_process, compute_logs
    if compute_process and compute_process.poll() is None:
        try:
            compute_process.send_signal(signal.CTRL_BREAK_EVENT)
            compute_process.wait(timeout=10)
        except:
            try:
                compute_process.terminate()
                compute_process.wait(timeout=5)
            except:
                compute_process.kill()
                compute_process.wait()
    compute_process = None
    compute_logs.append("Вычисление остановлено.")
    return jsonify({'status': 'ok'})


@app.route('/plots_data')
def get_plots_data():
    output_path = request.args.get('output_path', 'output')
    metrics_file = os.path.join(output_path, 'logs', 'training_metrics.json')
    try:
        if not os.path.exists(metrics_file):
            return jsonify({'epochs': [], 'train_losses': [], 'eval_losses': [], 'perplexities': [], 'is_training': is_training, 'current_epoch': 0, 'training_started': training_started})
        mtime = os.path.getmtime(metrics_file)
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        epochs = data.get('epochs', [])
        return jsonify({
            'epochs': epochs,
            'train_losses': data.get('train_losses', []),
            'eval_losses': data.get('eval_losses', []),
            'perplexities': data.get('perplexities', []),
            'is_training': is_training,
            'current_epoch': len(epochs),
            'training_started': training_started,
            'file_mtime': mtime
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download_model', methods=['POST'])
def download_model():
    global download_process, is_downloading
    if is_busy():
        return jsonify({'error': 'Дождитесь завершения текущего процесса'}), 400
    config = request.json
    model_name = config.get('model_name', '')
    save_path = config.get('save_path', '')
    if not model_name.strip():
        return jsonify({'error': 'Укажите название модели'}), 400
    if not save_path.strip():
        return jsonify({'error': 'Укажите директорию для сохранения'}), 400
    cmd = [sys.executable, "-u", "download_model.py", model_name, save_path]
    add_log(f"Скачивание модели {model_name} в {save_path}")
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        download_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding='utf-8', errors='replace',
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, env=env
        )
        is_downloading = True
        
        output = []
        def read_output():
            global is_downloading, download_process
            try:
                for line in download_process.stdout:
                    output.append(line.rstrip())
                download_process.wait()
            except:
                pass
            finally:
                is_downloading = False
                download_process = None
        
        t = threading.Thread(target=read_output, daemon=True)
        t.start()
        t.join(timeout=600)
        
        if download_process.poll() is None:
            download_process.kill()
            download_process.wait()
            is_downloading = False
            download_process = None
            return jsonify({'error': 'Превышено время загрузки'}), 500
        
        if download_process.returncode == 0:
            add_log("Модель успешно скачана.")
            return jsonify({'status': 'ok'})
        else:
            add_log(f"Ошибка скачивания (код: {download_process.returncode})")
            return jsonify({'error': '\n'.join(output)}), 500
    except Exception as e:
        add_log(f"ОШИБКА: {str(e)}")
        is_downloading = False
        download_process = None
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def get_status():
    running_process = None
    if is_training:
        running_process = 'обучение'
    elif is_inference:
        running_process = 'инференс'
    elif compute_process and compute_process.poll() is None:
        running_process = 'вычисление метрик'
    elif download_process and download_process.poll() is None:
        running_process = 'загрузка модели'
    
    return jsonify({
        'is_training': is_training,
        'is_inference': is_inference,
        'is_computing': compute_process is not None,
        'is_downloading': download_process is not None,
        'running_process': running_process,
        'training_model_path': training_model_path,
        'inference_model_path': inference_model_path
    })


if __name__ == '__main__':
    os.makedirs("models/lora_adapters", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print(f"Запуск сервера на http://0.0.0.0:5000")
    print(f"Python: {sys.executable}")
    print(f"Рабочая директория: {os.getcwd()}")
    app.run(host='0.0.0.0', port=5000, debug=False)