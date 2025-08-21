import os
import re
import wget
import glob
import multiprocessing
import warnings
import requests
import fake_useragent
import pymorphy2
import unicodedata
import zipfile


from pick import pick
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.corpora.wikicorpus import WikiCorpus
from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm
from bs4 import BeautifulSoup
from itertools import islice
from colorama import init, Fore, Style
from dotenv import load_dotenv

import time
from urllib3.exceptions import IncompleteRead

TOKENIZER = WordPunctTokenizer()
MORPH = pymorphy2.MorphAnalyzer()

class TrainingWord2VecEvents(CallbackAny2Vec):
    def __init__(self, epoch_current, epochs_count, epochs_dir):
        self.epochs_dir = epochs_dir
        os.makedirs(self.epochs_dir, exist_ok=True)
        self.epoch_current = epoch_current
        self.epochs_count = epochs_count
        self.pbar = None

    def on_train_begin(self, model):
        self.pbar = tqdm(total=self.epochs_count, desc="Training Word2Vec model...", unit=' epoch', initial=self.epoch_current)

    def on_epoch_end(self, model):
        if self.epochs_dir != '':
            epochs_path = os.path.join(self.epochs_dir, f'word2vec_epoch_{self.epoch_current}.model')
            model.save(epochs_path)
        print(f'\n Epoch {self.epoch_current+1} of {self.epochs_count} completed. Model saved to {epochs_path}')
        self.epoch_current += 1
        self.pbar.update(1)

    def on_train_end(self, model):
        self.pbar.close()


class LoadCacheSentences:
    def __init__(self, path):
        self.path = path
    def __iter__(self):
        with open(self.path, 'r', encoding='utf-8') as file:
            for line in file:
                yield line.strip().split()


def get_server_config():
    """Получение конфигурации сервера из переменных окружения"""
    load_dotenv()
    return {
        'ip': os.getenv('SERVER_IP', 'localhost'),
        'ptr': os.getenv('PTR_NAME', 'unknown'),
        'port': int(os.getenv('SERVER_PORT', 8080)),
        'protocol': os.getenv('SERVER_PROTOCOL', 'http'),
        'api_endpoint': os.getenv('API_ENDPOINT', '/api/models')
    }

def download_from_server():
    server_configs = get_server_config()
    
    servers_to_try = [
        f"{server_configs['protocol']}://{server_configs['ip']}:{server_configs['port']}"
    ]
    
    for base_url in servers_to_try:
        try:
            response = requests.get(f"{base_url}{server_configs['api_endpoint']}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('models'):
                    return select_model_on_server(base_url, data['models'])
                    
        except requests.exceptions.RequestException:
            print(f"Сервер недоступен: {base_url}")

    return None

def select_model_on_server(base_url, models):
    if not models:
        print("Модели не найдены на сервере")
        return None
    
    print(f"\nНайдено моделей на сервере: {len(models)}")
    # Показываем список моделей
    model_options = []
    for model in models:
        display_name = f"{model['name']} ({model['size_mb']} MB)"
        model_options.append(display_name)
    
    model_options.append("Отмена")
    
    selected_idx = pick(model_options, "Выберите модель для скачивания:", indicator='>')[1]
    
    if selected_idx == len(model_options) - 1:  # Отмена
        return None
    
    selected_model = models[selected_idx]

    if not selected_model:
        return None

    download_url = f"{base_url}{selected_model['download_url']}"
    return download_and_extract(download_url, selected_model['name'])


def download_and_extract(url, filename, max_retries=5, chunk_size=1024*1024):
    """Надежное скачивание с улучшенной обработкой ошибок"""
    try:
        if not os.path.exists('./models'):
            os.makedirs('./models')
        
        local_path = f"./models/{filename}"
        temp_path = f"{local_path}.tmp"  # Временный файл
        
        print(f"Скачивание {filename}...")
        
        # Получаем размер файла
        try:
            head_response = requests.head(url, timeout=30)
            if head_response.status_code == 200:
                total_size = int(head_response.headers.get('content-length', 0))
            else:
                total_size = 0
        except Exception as e:
            total_size = 0
        
        # Проверяем частично скачанный файл
        initial_pos = 0
        if os.path.exists(temp_path):
            initial_pos = os.path.getsize(temp_path)
        
        for attempt in range(max_retries):
            try:
                ua = fake_useragent.UserAgent()
                headers = {
                    'User-Agent': ua.random,
                    'Accept': '*/*',
                    'Accept-Encoding': 'identity',  # Отключаем сжатие
                    'Connection': 'keep-alive'
                }
                
                # Докачка с позиции
                if initial_pos > 0:
                    headers['Range'] = f'bytes={initial_pos}-'
                
                # Делаем запрос с увеличенным таймаутом
                session = requests.Session()
                session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
                
                response = session.get(url, headers=headers, stream=True, timeout=(30, 300))
                
                if response.status_code not in [200, 206]:
                    print(f"HTTP ошибка: {response.status_code}")
                    if response.status_code == 416:  # Range not satisfiable
                        print("Файл уже скачан полностью")
                        if os.path.exists(temp_path):
                            os.rename(temp_path, local_path)
                        return process_downloaded_file(local_path, filename)
                    continue
                
                # Обновляем общий размер если получили
                if response.status_code == 206:
                    content_range = response.headers.get('content-range', '')
                    if content_range:
                        total_size = int(content_range.split('/')[-1])
                elif response.status_code == 200 and initial_pos > 0:
                    # Сервер не поддерживает Range, начинаем сначала
                    initial_pos = 0
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                file_mode = 'ab' if initial_pos > 0 else 'wb'
                
                # Скачивание
                with open(temp_path, file_mode) as file:
                    with tqdm(
                        total=total_size,
                        initial=initial_pos,
                        desc=f"{filename}",
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        miniters=1,  # Обновляем чаще
                        maxinterval=1.0  # Максимальный интервал обновления
                    ) as pbar:
                        
                        downloaded = initial_pos
                        consecutive_errors = 0
                        last_downloaded = downloaded
                        
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                try:
                                    file.write(chunk)
                                    downloaded += len(chunk)
                                    pbar.update(len(chunk))
                                    consecutive_errors = 0
                                    
                                    # Принудительно сбрасываем буфер каждые 10MB
                                    if downloaded - last_downloaded > 10 * 1024 * 1024:
                                        file.flush()
                                        os.fsync(file.fileno())
                                        last_downloaded = downloaded
                                        
                                except IOError as io_error:
                                    consecutive_errors += 1
                                    print(f"Ошибка записи: {io_error}")
                                    if consecutive_errors > 5:
                                        raise io_error
                                    time.sleep(0.1)
                            else:
                                # Пустой chunk может означать конец или временную проблему
                                time.sleep(0.01)
                
                # Проверяем результат
                actual_size = os.path.getsize(temp_path)
                
                if total_size > 0 and actual_size >= total_size:
                    print(f"Файл успешно скачан: {actual_size / (1024*1024):.1f} MB")
                    # Перемещаем из временного в финальный
                    os.rename(temp_path, local_path)
                    return process_downloaded_file(local_path, filename)
                    
                elif total_size == 0:
                    print(f"Файл скачан")
                    os.rename(temp_path, local_path)
                    return process_downloaded_file(local_path, filename)
                    
                else:
                    print(f"Файл неполный: {actual_size}/{total_size} байт ({(actual_size/total_size*100):.1f}%)")
                    initial_pos = actual_size
                    if attempt < max_retries - 1:
                        print(f"Попытка {attempt + 2}/{max_retries} через 3 секунды...")
                        time.sleep(3)
                        continue
                        
            except (requests.exceptions.RequestException, 
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    IncompleteRead,
                    IOError) as e:
                print(f"Ошибка соединения (попытка {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = min((attempt + 1) * 5, 30)  # Максимум 30 сек
                    print(f"Пауза {wait_time} секунд...")
                    time.sleep(wait_time)
                    
                    # Проверяем размер частично скачанного файла
                    if os.path.exists(temp_path):
                        initial_pos = os.path.getsize(temp_path)
                        print(f"Продолжаем с позиции: {initial_pos / (1024*1024):.1f} MB")
                else:
                    print(f"Попытки исчерпаны")

        # Очистка при неудаче
        if os.path.exists(temp_path):
            os.remove(temp_path)            
        return None
        
    except Exception as e:
        print(f"Ошибка скачивания: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def process_downloaded_file(local_path, filename):
    """Обработка скачанного файла"""
    try:
        if filename.endswith('.zip'):
            return extract_zip(local_path)
    except Exception as e:
        print(f"Ошибка обработки файла: {e}")
        return None

def extract_zip(zip_path):
    """Распаковка ZIP"""
    try:
        print(f"Распаковка {os.path.basename(zip_path)}...")
        
        # Проверяем целостность ZIP файла
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Тестируем архив
            bad_file = zip_ref.testzip()
            if bad_file:
                print(f"Поврежденный файл в архиве: {bad_file}")
                return None
            
            file_list = zip_ref.namelist()
            
            # Распаковываем все файлы с прогресс-баром
            extract_dir = './models/'
            
            with tqdm(total=len(file_list), desc="Extracting", unit="file") as pbar:
                for file_info in zip_ref.infolist():
                    zip_ref.extract(file_info, extract_dir)
                    pbar.update(1)
            
            # Находим файл модели (обычно .model)
            model_files = [f for f in file_list if f.endswith('.model')]
            if model_files:
                extracted_path = os.path.join(extract_dir, model_files[0])
                
                # Если файл в подпапке, перемещаем в корень ./models/
                if '/' in model_files[0]:
                    final_path = os.path.join(extract_dir, os.path.basename(model_files[0]))
                    if os.path.exists(extracted_path):
                        os.rename(extracted_path, final_path)
                        
                        # Удаляем пустые папки
                        try:
                            parent_dir = os.path.dirname(extracted_path)
                            if parent_dir != extract_dir and os.path.exists(parent_dir):
                                os.rmdir(parent_dir)
                        except:
                            pass
                            
                        extracted_path = final_path
        
        if os.path.exists(extracted_path):
            user_choise = input('Удалить ZIP архив? (y/n): ')
            if user_choise.lower() == 'y' or user_choise.lower() == 'yes':
                os.remove(zip_path)
                print(f"ZIP файл удален: {zip_path}")

            return load_model()
        else:
            print(f"Модель не найдена после распаковки: {extracted_path}")
            return None
        
    except zipfile.BadZipFile:
        print(f"Поврежденный ZIP файл: {zip_path}")
        # Удаляем поврежденный файл
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None
    except Exception as e:
        print(f"Ошибка распаковки: {e}")
        return None


def download_data(download_link, filename, download_method=0):
    dest_path = os.path.join('./data/downloads/', filename)
    response_file_size = 0
    local_file_size = 0

    response = requests.get(download_link, stream=True)

    if  os.path.exists(dest_path):
        local_file_size = os.path.getsize(dest_path)
        response_file_size = int(response.headers.get('content-length', 0))

    if response_file_size == local_file_size and response_file_size > 0 and local_file_size > 0:
        pass
    else:
        if download_method == 0:
            wget.download(download_link, f'./{filename}')
        elif download_method == 1:
            with open(dest_path, 'wb') as file, tqdm(desc=dest_path, total=response_file_size, unit='B', unit_scale=True) as bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    bar.update(len(data))


def extract_wiki_articles(file_path, min_articles_length=10, max_articles=None):
    data = WikiCorpus(file_path, dictionary={}, tokenizer_func=custom_tokenizer)
    
    article_count = 0
    
    if max_articles:
        pbar = tqdm(total=max_articles, desc=f"Extracting articles from {os.path.basename(file_path)}", unit=' articles')
    else:
        pbar = tqdm(desc=f"Extracting articles from {os.path.basename(file_path)}", unit=' articles')
    
    try:
        for article in data.get_texts():
            if max_articles and article_count >= max_articles:
                break

            if len(article) > min_articles_length:
                yield article
                article_count += 1
                pbar.update(1)
                
                if not max_articles:
                    pbar.set_description(f"Extracting articles from {os.path.basename(file_path)} ({article_count} extracted)")
    finally:
        pbar.close()
        print(f"\nExtracted {article_count} articles from {os.path.basename(file_path)}")


def custom_tokenizer(content, token_min_len=2, token_max_len=20, lower=True):
    if not content or not content.strip():
        return []
    
    if lower:
        content = content.lower()
    
    content = unicodedata.normalize('NFD', content)
    content = ''.join(char for char in content if unicodedata.category(char) != 'Mn')
    content = unicodedata.normalize('NFC', content)

    tokens = TOKENIZER.tokenize(content)
    
    clean_tokens = []
    for token in tokens:
        if any(char.isalpha() for char in token):
            cleaned_token = re.sub(r'^[^\w]+|[^\w]+$', '', token, flags=re.UNICODE)
            filtered_token = ''.join(char for char in cleaned_token if char.isalpha())

            if (filtered_token and len(filtered_token) >= token_min_len and len(filtered_token) <= token_max_len):
                clean_tokens.append(filtered_token)
                

    lemmatized_tokens = []
    for token in clean_tokens:
        try:
            parsed = MORPH.parse(token)[0]
            lemma = parsed.normal_form.lower()

            if (lemma and len(lemma) >= token_min_len and len(lemma) <= token_max_len and lemma.isalpha()):
                lemmatized_tokens.append(lemma)
        except (IndexError, AttributeError):
            if token.isalpha():
                lemmatized_tokens.append(token)
    
    return lemmatized_tokens


def get_wiki_dumps_links(base_url, download_type):
    headers_fake_useragent = fake_useragent.UserAgent()

    response = requests.get(base_url, headers={'user-agent': headers_fake_useragent.random})
    links = []

    if response.status_code == 200:
        bs_data = BeautifulSoup(response.content, features="html.parser")
        if download_type == 0:
            pattern = re.compile(r"ruwiki-latest-pages-articles-multistream\.xml\.bz2")
        elif download_type == 1:
            pattern = re.compile(r"ruwiki-latest-pages-articles-multistream\d+\.xml-p.*\.bz2")
        
        for link_a in bs_data.find_all('a', href=True):
            if pattern.fullmatch(link_a['href']):
                links.append(link_a['href'])
    else:
        raise ValueError('Wrong response from the server')
    return links


def get_data_for_train(files_limit):
    base_url = 'https://dumps.wikimedia.org/ruwiki/latest/' 
    download_type = ['Full', 'Parts']
    
    index = pick(download_type, 'Please choose a download type:', indicator='>',)[1]

    download_links = get_wiki_dumps_links(base_url, index)
    if not download_links:
        return ValueError("Download links not found")
    else:
        if not os.path.exists('./data/downloads'):
            os.makedirs('./data/downloads')

        for i in range(len(download_links)-(len(download_links)-files_limit)):
            file_download_link = f'{base_url}{download_links[i]}'
            print(f'Downloading: {i+1} of {len(download_links)-(len(download_links)-files_limit)}')
            download_data(file_download_link, download_links[i], 1)

    print('Data downloaded successfully.')
    return download_links


def count_examples(data):
    """Подсчитывает количество предложений в данных"""
    if hasattr(data, '__len__'):
        return len(data)
    elif isinstance(data, LoadCacheSentences):
        count = 0
        with open(data.path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    else:
        count = 0
        for _ in data:
            count += 1
        return count

def train_model():
    if not os.path.exists('./models'):
        os.makedirs('./models')

    init(autoreset=True)
    files_limit = 1
    max_articles = None
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
    
    wiki_clean_cache_data_path = './data/cache/wiki_clean_cache_data.txt'
    train_data = None

    user_choise_load_train_data_file = 'No'

    if os.path.exists(wiki_clean_cache_data_path):
        user_choise_load_train_data_file = pick(['Yes', 'No'], f"Generate 'train_data' from {wiki_clean_cache_data_path}", indicator='>')[0]

    if user_choise_load_train_data_file == 'Yes':
        train_data = LoadCacheSentences(wiki_clean_cache_data_path)
    else:
        user_choise_save_sentences_file = pick(['Yes', 'No'], 'Save preparing data?:', indicator='>')[1]

        files_for_train = get_data_for_train(files_limit)
        print('Processing the data...')
        
        for i in range(len(files_for_train)-(len(files_for_train)-files_limit)):
            file_path = f'./data/downloads/{files_for_train[i]}'
            if not os.path.exists(file_path):
                print(f'File {file_path} does not exist')
                continue
            
            print(f'\nProcessing file {i+1} of {files_limit}: {os.path.basename(file_path)}')
            
            if user_choise_save_sentences_file == 0:
                os.makedirs(os.path.dirname(wiki_clean_cache_data_path), exist_ok=True)
                with open(wiki_clean_cache_data_path, 'a', encoding='utf-8') as wiki_clean_cache_data_file:
                    for article in extract_wiki_articles(file_path, min_articles_length=10, max_articles=max_articles):
                        wiki_clean_cache_data_file.write(' '.join(article) + '\n')
            else:
                articles_from_file = list(extract_wiki_articles(file_path, min_articles_length=10, max_articles=max_articles))
                if isinstance(train_data, list):
                    train_data.extend(articles_from_file)
                else:
                    train_data = articles_from_file

            if train_data is None and os.path.exists(wiki_clean_cache_data_path):
                train_data = LoadCacheSentences(wiki_clean_cache_data_path)
            elif train_data is None and not os.path.exists(wiki_clean_cache_data_path):
                raise FileNotFoundError(f"File {wiki_clean_cache_data_path} does not exist and no train_data iterable was provided.")

    model = None
    epochs_dir='word2vec_epochs'
    epochs_count = 1
    vector_size = 300
    window = 2
    min_count = 10

    params = {'vector_size': vector_size, 'window': window, 'min_count': min_count, 'workers': multiprocessing.cpu_count()}
    previously_epochs_files = glob.glob(f'./models/{epochs_dir}/word2vec_epoch_*.model')
    if previously_epochs_files:
        user_choice = pick(['Yes', 'No'], 'Do you want to continue training from the last epoch?', indicator='>')[0]
        if user_choice == 'Yes':
            last_epoch_file = max(previously_epochs_files, key=os.path.getctime)
            match = re.search(r'word2vec_epoch_(\d+)\.model', os.path.basename(last_epoch_file))
            epoch_current = int(match.group(1)) + 1 if match else 0
            model = Word2Vec.load(last_epoch_file)
            print(f"Continuing training from the last epoch: {os.path.basename(last_epoch_file)}")
            model.train(train_data, 
                        total_examples=count_examples(train_data),
                        epochs=model.epochs,
                        callbacks=[TrainingWord2VecEvents(epoch_current, model.epochs, epochs_dir)])
        else:
            for file in previously_epochs_files:
                try:
                    os.remove(file)
                except Exception as err:
                    print(f"Failed to remove {file}: {err}")

            model = Word2Vec(train_data,
                            total_examples=count_examples(train_data),
                            epochs=epochs_count,
                            callbacks=[TrainingWord2VecEvents(0, epochs_count, epochs_dir)],
                            **params)
    
    if model is None:
        print(f"Creating Word2Vec model and train with {epochs_count} epochs...")
        model = Word2Vec(train_data,
                        epochs=epochs_count,
                        callbacks=[TrainingWord2VecEvents(0, epochs_count, epochs_dir)],
                        **params)
    
    if model is not None:         
        model.save(f'./models/wikidumps-{vector_size}.model')
        print(Style.BRIGHT + Fore.GREEN + f"Model training completed and saved as 'wikidumps-{vector_size}.model'")
        return model


def load_model():
    model_files = glob.glob('./models/*-*.model')
    try:
        if not model_files:
            return print("Нет доступных моделей для загрузки. Пожалуйста, обучите модель или скачайте её с сервера.")
        else:
            options = [os.path.basename(f) for f in model_files]
            idx = pick(options, "Выберите модель для загрузки:", indicator='>')[1]
            model_path = model_files[idx]
            model = Word2Vec.load(model_path)

            file_size = os.path.getsize(model_path) / (1024*1024)        
            vocab_size = len(model.wv.key_to_index)
            vector_size = model.wv.vector_size

            print(f"Model [{model_path}] loaded.")
            print(f"   - Размер словаря: {vocab_size:,} слов")
            print(f"   - Размер векторов: {vector_size}")
            print(f"   - Размер файла: {file_size:.1f} MB")
            print(f"   - Путь: {model_path}")

            return model
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print(f"Возможно, файл поврежден или это не файл модели Word2Vec")
        return None


def word2vec_menu():
    init(autoreset=True)
    
    action = ''
    model = None

    while action != 'Выход':
        if model is not None:
            select_mode = ['Обучить модель', 'Тестировать модель', 'Скачать модель', 'Выход']
        else: 
            select_mode = ['Обучить модель', 'Загрузить модель', 'Скачать модель', 'Выход']

        if action == '':
            action = pick(select_mode, 'Пожалуйста, выберите опцию:', indicator='>')[0]

        try:
            if action == 'Обучить модель':
                model = train_model()
                action = ''
            elif action == 'Загрузить модель':
                try:
                    print("Загрузка модели...")
                    model = load_model()
                except FileNotFoundError as err:
                    print(err)
                input("Нажмите любую клавишу для продолжения...")
                action = ''
            elif action == 'Тестировать модель':
                if model is not None:
                    user_input = input("Введите слово, чтобы найти похожие слова (или введите '/exit' для выхода): ").lower()
                    if user_input == '/exit':
                        action = ''
                    else:
                        try:
                            similar_words = model.wv.most_similar(user_input, topn=5)
                            print(f"Наиболее похожие слова к '{user_input}':")
                            for word, similarity in similar_words:
                                print(f"    ~{word}: {similarity:.4f}")
                        except KeyError:
                            print(f"'{user_input}' не найдено в словаре.")
                else:
                    print("Модель не загружена. Пожалуйста, сначала загрузите модель.")
                    action = ''
            elif action == 'Скачать модель':
                model = download_from_server()
                action = ''
            else:
                print("Выход из модуля...")
                break
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            input("Нажмите любую клавишу для продолжения...")
            action = ''


if __name__ == "__main__":
    word2vec_menu()
