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

from pick import pick
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.corpora.wikicorpus import WikiCorpus
from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm
from bs4 import BeautifulSoup
from itertools import islice
from colorama import init, Fore, Style

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
    if not model_files:
        return print("No Word2Vec models found in. Please train the model first.")
    else:
        options = [os.path.basename(f) for f in model_files]
        idx = pick(options, "Select a model to load:", indicator='>')[1]
        model_path = model_files[idx]
        model = Word2Vec.load(model_path)
        print(f"Model [{model_path}] loaded.")
        return model

def word2vec_menu():
    init(autoreset=True)
    print(Style.BRIGHT + Fore.YELLOW + "Welcome to the Word2Vec training and testing menu!")
    print("You can train a new model, load an existing model, or test the model with your own words.")
    
    action = ''
    model = None

    while action != 'Выход':
        if model is not None:
            select_mode = ['Обучить модель', 'Тестировать модель', 'Выход']
        else: 
            select_mode = ['Обучить модель', 'Загрузить модель', 'Выход']

        if action == '':
            action = pick(select_mode, 'Пожалуйста, выберите опцию:', indicator='>')[0]

        if action == 'Обучить модель':
            model = train_model()
            action = ''
        elif action == 'Загрузить модель':
            try:
                print("Загрузка модели...")
                model = load_model()
            except FileNotFoundError as err:
                print(err)
            action = ''
        elif action == 'Тестировать модель':
            if model is not None:
                user_input = input("Введите слово, чтобы найти похожие слова (или введите '/exit' для выхода): ").lower()
                if user_input == '/exit':
                    action = ''
                else:
                    try:
                        similar_words = model.wv.most_similar(user_input, topn=5)
                        print(f"Most similar words to '{user_input}':")
                        for word, similarity in similar_words:
                            print(f"    ~{word}: {similarity:.4f}")
                    except KeyError:
                        print(f"'{user_input}' not found in the vocabulary.")
            else:
                print("No model loaded. Please load a model first.")
                action = ''
        else:
            print("Выход из модуля...")
            break


if __name__ == "__main__":
    word2vec_menu()
