import requests
import fake_useragent
import re
import json
import pandas as pd
import numpy as np
import os
import glob

from pick import pick
from datetime import datetime
from bs4 import BeautifulSoup
from time import sleep
from tqdm import tqdm

SLEEP_TIME = 3  # Время ожидания между запросами к API hh.ru

class ResumeData:
    def __init__(self, soup_resume_data=None, resume_url=''):
        if soup_resume_data is not None and resume_url:
            self.soup = soup_resume_data
            self.resume_id = self._extract_resume_id(resume_url)
            self.name = self._extract_name()
            self.area = self._extract_area()
            self.salary, self.currency = self._extract_salary()
            self.experience_ID = self._extract_experience_id()
            self.education = self._extract_education()
            self.skills = self._extract_skills()
            self.description = self._extract_description()
            self.description_about_me = self._extract_description_about_me()
            self.url = resume_url

    def inject_data(self, data):
        """Внедряет данные в объект"""
        for key, value in data.items():
            setattr(self, key, value)

    def _extract_resume_id(self, url):
        """Извлекает ID резюме из URL"""
        resume_id = None
        try:
            match = re.search(r'/resume/([a-f0-9]+)', url)
            if match:
                resume_id = match.group(1)
        except:
            pass
        return resume_id

    def _extract_name(self):
        """Извлекает наименование резюме"""
        name = None
        try:
            name_item = self.soup.find('span', {'data-qa': 'resume-block-title-position'})
            if name_item:
                name = name_item.get_text(separator=' ', strip=True)

            # Альтернативный селектор
            name_item = self.soup.find('h1', class_=re.compile(r'resume-block-container'))
            if name_item:
                name = name_item.get_text(separator=' ', strip=True)
        except:
            pass
        return name

    def _extract_area(self):
        """Извлекает местоположение"""
        area = None
        try:
            # Основной селектор
            area_item = self.soup.find('span', {'data-qa': 'resume-personal-address'})
            if area_item:
                area = area_item.get_text(separator=' ', strip=True)

            # Альтернативный поиск
            area_item = self.soup.find('div', class_=re.compile(r'resume-personal-address'))
            if area_item:
                area = area_item.get_text(separator=' ', strip=True)
        except:
            pass
        return area

    def _extract_salary(self):
        """Извлекает зарплату и валюту"""
        amount = None
        currency = None
        try:
            salary_elem = self.soup.find('span', {'data-qa': 'resume-block-salary'})
            if salary_elem:
                salary_text = salary_elem.get_text(separator=' ', strip=True)
                
                amount_match = re.search(r'([\d\s]+)', salary_text.replace('\xa0', ' '))
                if amount_match:
                    amount_str = amount_match.group(1).replace(' ', '').strip()
                    try:
                        amount = int(amount_str)
                    except:
                        pass
                
                currency_dict = get_hh_dictionaries('currency')
                currency_symbols = []
                currency_codes = []
                for cur in currency_dict:
                    if 'abbr' in cur and 'code' in cur:
                        currency_symbols.append(cur['abbr'])
                        currency_codes.append(cur['code'])

                if currency_symbols:
                    escaped_patterns = [re.escape(pattern) for pattern in currency_symbols]
                    currency_pattern = '|'.join(escaped_patterns)
                    currency_match = re.search(f'({currency_pattern})', salary_text, re.IGNORECASE)
                    
                    if currency_match:
                        currency = currency_match.group(1)
                        currency_lower = currency.lower()
                        currency_mapping = {key: value for key, value in zip(currency_symbols, currency_codes)}
                        currency = currency_mapping.get(currency_lower, currency_mapping.get(currency, currency.upper()))
        except:
            pass
        return amount, currency


    def _extract_experience_id(self):
        """Извлекает опыт работы"""
        exp_id = None
        try:
            experience_section = self.soup.find('div', {'data-qa': 'resume-block-experience'})
            total_experience_header = experience_section.find('span', class_=re.compile(r'resume-block__title-text_sub'))
            if total_experience_header:
                total_experience_time = total_experience_header.get_text(strip=True).lower()

                years = 0
                months = 0

                ru_patterns = [
                    r'(\d+)\s*(?:лет|года|год)[\s,]*(\d+)?\s*(?:месяц|месяца|месяцев)?',
                    r'(\d+)\s*(?:лет|года|год)',
                    r'(\d+)\s*(?:месяц|месяца|месяцев)'
                ]
                
                en_patterns = [
                    r'(\d+)\s*(?:years?|yrs?)[\s,]*(\d+)?\s*(?:months?|mos?)?',
                    r'(\d+)\s*(?:years?|yrs?)',
                    r'(\d+)\s*(?:months?|mos?)'
                ]
                
                # Пробуем русские паттерны
                for pattern in ru_patterns:
                    match = re.search(pattern, total_experience_time)
                    if match:
                        if 'лет' in pattern or 'года' in pattern or 'год' in pattern:
                            years = int(match.group(1))
                            if match.group(2):
                                months = int(match.group(2))
                        elif 'месяц' in pattern:
                            months = int(match.group(1))
                        break
                
                # Если русские паттерны не сработали, пробуем английские
                if years == 0 and months == 0:
                    for pattern in en_patterns:
                        match = re.search(pattern, total_experience_time)
                        if match:
                            if 'year' in pattern or 'yr' in pattern:
                                years = int(match.group(1))
                                if match.group(2):
                                    months = int(match.group(2))
                            elif 'month' in pattern or 'mo' in pattern:
                                months = int(match.group(1))
                            break

                experience_dict = get_hh_dictionaries('experience')
                total_months = years * 12 + months

                for exp in experience_dict:
                    name = exp.get('name', '').lower()
                    if 'нет опыта' in name and total_months == 0:
                        exp_id = exp['id']
                        break
                    elif 'от 1 года' in name and 12 <= total_months < 36:
                        exp_id = exp['id']
                        break
                    elif 'от 3 до 6' in name and 36 <= total_months < 72:
                        exp_id = exp['id']
                        break
                    elif 'более 6' in name and total_months >= 72:
                        exp_id = exp['id']
                        break

                if exp_id is None:
                    exp_id = experience_dict[0].get('id')
        except:
            pass
        return exp_id

    def _extract_education(self):
        """Извлекает образование"""
        education = None
        try:
            education_section = self.soup.find('div', {'data-qa': 'resume-block-education'})
            if education_section:
                education_list = []
                education_blocks = education_section.find_all('div', {'data-qa': 'resume-block-education-item'})
                
                # Альтернативный поиск блоков образования
                if not education_blocks:
                    education_blocks = education_section.find_all('div', class_=re.compile(r'resume-block-item'))

                for block in education_blocks:
                    try:
                        institution_elem = block.find('div', {'data-qa': 'resume-block-education-name'})
                        institution = institution_elem.get_text(separator=' ', strip=True) if institution_elem else ''

                        specialty_elem = block.find('div', {'data-qa': 'resume-block-education-organization'})
                        specialty = specialty_elem.get_text(separator=' ', strip=True) if specialty_elem else ''

                        if institution != '' and specialty != '':
                            education_list.append((f"{institution}, {specialty}").replace('\n', ' ').replace('\r', ' '))
                    except:
                        continue
                education = '; '.join(education_list)
        except:
            pass
        return education

    def _extract_skills(self):
        """Извлекает навыки"""
        skills = None
        try:
            skills_list = []
            skills_section = self.soup.find('div', {'data-qa': 'resume-block-skills'})
            
            if skills_section:
                skill_elements = skills_section.find_all('span', {'data-qa': 'resume-block-skills-element'})
                for skill_elem in skill_elements:
                    skill = skill_elem.get_text(separator=' ', strip=True)
                    if skill:
                        skills_list.append(skill.replace('\n', ' ').replace('\r', ' '))

            # Альтернативный поиск
            if not skills_list:
                skill_elements = self.soup.find_all('span', class_=re.compile(r'tag'))
                for skill_elem in skill_elements:
                    skill = skill_elem.get_text(separator=' ', strip=True)
                    if skill and len(skill) > 1:
                        skills_list.append(skill.replace('\n', ' ').replace('\r', ' '))

            skills = ', '.join(skills_list)
        except:
            pass
        return skills


    def _extract_description(self):
        description = None
        try:
            experience_section = self.soup.find('div', {'data-qa': 'resume-block-experience'})
            if experience_section:
                experience_items = experience_section.find_all('div', class_=re.compile(r'resume-block-item-gap'))
                descriptions = []
                for item in experience_items:
                    text = item.find('div', {'data-qa': 'resume-block-experience-description'})
                    if text:
                        descriptions.append(text.get_text(separator=' ', strip=True).replace('\n', ' ').replace('\r', ' '))
                descriptions = [desc for desc in descriptions if desc]
                if descriptions:
                    description = ' '.join(descriptions)
        except:
            pass
        return description


    def _extract_description_about_me(self):
        description_about_me = None
        try:
            about_me_section = self.soup.find_all('div', {'data-qa': 'resume-block-skills-content'})
            if about_me_section:
                descriptions = []
                for item in about_me_section:
                    descriptions.append(item.get_text(separator=' ', strip=True).replace('\n', ' ').replace('\r', ' '))
                descriptions = [desc for desc in descriptions if desc]
                if descriptions:
                    description_about_me = ' '.join(descriptions)
        except:
            pass
        return description_about_me

    def is_valid(self):
        """Проверяет валидность данных резюме."""
        return all([
            self.name is not None,
            self.skills is not None,
            self.description is not None,
        ])

    def get_header_text(self):
        """Возвращает заголовок резюме."""
        text = f"{self.resume_id} ({self.name})"
        return text

    def get_field(self, field_name):
        """Возвращает значение указанного поля вакансии."""
        try:
            return getattr(self, field_name, None)
        except AttributeError:
            return None

    def get_resume_data(self):
        """Возвращает данные резюме в виде словаря."""
        return {
            'resume_id': self.resume_id,
            'name': self.name,
            'area': self.area,
            'salary': self.salary,
            'currency': self.currency,
            'experience_ID': self.experience_ID,
            'education': self.education,
            'skills': ', '.join(self.skills) if isinstance(self.skills, list) else str(self.skills),
            'description': self.description,
            'description_about_me': self.description_about_me,
            'url': self.url
            }


class VacancyData:
    def __init__(self, json_vacancy_data=None):
        if json_vacancy_data is not None:
            self.vacancy_id = json_vacancy_data.get('id', np.nan)
            self.name = json_vacancy_data.get('name', np.nan)
            self.area = json_vacancy_data.get('area', np.nan).get('name', np.nan)
            self.published_at = (datetime.strptime(json_vacancy_data.get('published_at', '1900-01-01T00:00:00+0300'), "%Y-%m-%dT%H:%M:%S%z")).strftime("%d.%m.%Y")
            self.salary_from = json_vacancy_data.get('salary', np.nan).get('from', np.nan)
            self.salary_to = json_vacancy_data.get('salary', np.nan).get('to', np.nan)
            self.currency = json_vacancy_data.get('salary', np.nan).get('currency', np.nan)
            self.gross = json_vacancy_data.get('salary', np.nan).get('gross', np.nan)
            self.experience_id = json_vacancy_data.get('experience', np.nan).get('id', np.nan)
            self.key_skills = self._extract_skills(json_vacancy_data.get('key_skills', np.nan))
            self.employer_name = json_vacancy_data.get('employer', np.nan).get('name', np.nan)
            self.employer_accredited_it = json_vacancy_data.get('employer', np.nan).get('accredited_it_employer', np.nan)
            self.description = BeautifulSoup(json_vacancy_data.get('description', '<span></span>'), features="html.parser").get_text()
            self.url = json_vacancy_data.get('alternate_url', np.nan)

    def inject_data(self, data):
        """Внедряет данные в объект"""
        for key, value in data.items():
            setattr(self, key, value)

    def _extract_published_at(self, parsed_date):
        published_at = None
        try:
            published_at = parsed_date.strftime("%d.%m.%Y")
        except:
            pass
        return published_at

    def get_header_text(self):
        """Возвращает заголовок."""
        text = f"{self.vacancy_id} ({self.name})"
        return text

    def get_field(self, field_name, default=None):
        """Возвращает значение указанного поля вакансии."""
        try:
            return getattr(self, field_name, default)
        except AttributeError:
            return None

    def get_vacancy_data(self):
        """Возвращает данные вакансии в виде словаря."""
        return {'vacancy_id': self.vacancy_id,
                'name': self.name,
                'area': self.area,
                'published_at': self.published_at,
                'salary_from': self.salary_from,
                'salary_to': self.salary_to,
                'currency': self.currency,
                'gross': self.gross,
                'experience_id': self.experience_id,
                'key_skills': self.key_skills,
                'employer_name': self.employer_name,
                'employer_accredited_it': self.employer_accredited_it,
                'description': self.description,
                'url': self.url
                }

def get_hh_dictionaries(key):
    url = 'https://api.hh.ru/dictionaries'
    data = get_hh_response(url)[1]
    dictionary_data = []
    if data:
        dictionary_data = json.loads(data)[key]
    return dictionary_data

def get_resume_links(region_id=1, search_text=''):    
    resume_url_list = []
    if not search_text:
        search_text = 'Data Science'
    search_text = search_text.replace(' ', '+').lower()
    
    url = (f'https://hh.ru/search/resume'
           f'?area={region_id}'
           f'&text={search_text}'
           f'&isDefaultArea=true'
           f'&pos=full_text'
           f'&logic=normal'
           f'&exp_period=all_time'
           f'&ored_clusters=true'
           f'&order_by=relevance'
           f'&search_period=0'
           f'&hhtmFrom=resume_search_result'
           f'&hhtmFromLabel=resume_search_line')
    
    data = get_hh_response(url)[1]

    if data != '':
        soup = BeautifulSoup(data, 'html.parser')
        resume_items = soup.find_all('a', href=re.compile(r'^/resume/'))
        
        if not resume_items:
            print(f'Не найдено резюме для региона с ID {region_id}')
            return resume_url_list
        
        for item in resume_items:
            resume_url_list.append('https://hh.ru/' + item['href'].split("?")[0])
        
    return resume_url_list
    
def get_vacancy_links_api(region_id=1, search_text=''):
    """Получает ссылки на вакансии по API hh.ru для заданного региона.
    Параметры: region_id - ID региона, для которого нужно получить вакансии.
    Возвращает список ссылок на вакансии."""
    
    vacancy_url_list = []
    if not search_text:
        search_text = 'Data Science'
    search_text = search_text.replace(' ', '%')  # Заменяем пробелы на % для корректного URL

    url = ('https://api.hh.ru/vacancies'
                    f'?area={region_id}'
                    '&only_with_salary=true'
                    f'&text={search_text}'
                    '&per_page=100')

    data = get_hh_response(url)[1]

    if data != '':
        vacancies_found = json.loads(data)['found']
        pages = json.loads(data)['pages']
        alternate_url = json.loads(data)['alternate_url']
        print(f'Найдено вакансий: {vacancies_found}\n'
              f'Страниц с вакансиями: {pages}\n'
              f'Ссылка для просмотра: {alternate_url}')
        
        
        items = json.loads(data)['items']
        vacancy_url_list = extract_vacancy_links_api(items, vacancy_url_list)
        sleep(SLEEP_TIME) 

        for page in range(1, pages):
            next_url = f'{url}&page={page}'
            status, data = get_hh_response(next_url)
            if data != '':
                items = json.loads(data)['items']
                vacancy_url_list = extract_vacancy_links_api(items, vacancy_url_list)
            else:
                print(f'\nНе удалось извлечь данные из: {next_url}\n{status}')
            sleep(SLEEP_TIME)

    return vacancy_url_list

def extract_vacancy_links_api(vacancies_list, url_list):
    for vacancy in vacancies_list:
        url_list.append(vacancy['url'])
    return url_list

def get_hh_response(url):
    """Получает данные по указанному URL.
    Параметры: url - URL для запроса.
    Возвращает текст ответа или сообщение об ошибке."""

    if not isinstance(url, str):
        raise ValueError('URL должен быть строкой')
    if not re.match(r'^https?://', url):
        raise ValueError('URL должен начинаться с http:// или https://')

    response_data = ''
    headers_fake_useragent = fake_useragent.UserAgent()
    
    response = requests.get(url, headers = {'user-agent': headers_fake_useragent.random})
    if response.status_code == 200:
        status = 'success - Данные получены'
        response_data = response.text
    elif response.status_code == 400:
        status = 'error - Параметры переданы с ошибкой'
    elif response.status_code == 403:
        status = 'error - Требуется ввести капчу'
    elif response.status_code == 404:
        status = 'error - Указанная вакансия не существует'
    else:
        status = 'error - Не удалось получить данные'
    return status, response_data

def save_to_csv(data, data_name, search_text, region_name):
    """Сохраняет данные в CSV файл"""
    if not os.path.exists('./data/hh'):
        os.makedirs('./data/hh')

    # Создаем имя файла на основе поискового запроса и региона
    safe_search_text = re.sub(r'[^\w\s-]', '', search_text).strip()
    safe_region_name = re.sub(r'[^\w\s-]', '', region_name).strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    file_name = f'hh_{data_name}_{safe_search_text}_{safe_region_name}_{timestamp}.csv'
    file_path = f'./data/hh/{file_name}'
    
    headers = list(data.columns)
    
    data.to_csv(file_path, sep=';', encoding='utf-8-sig', index=False, header=headers)
    print(f'Данные сохранены в файл: {file_path}')

def load_from_csv(filter=''):
    """Загружает данные из указанного CSV файла.
    Возвращает DataFrame с данными."""

    hh_csv_files = glob.glob(f'./data/hh/*{filter}*.csv')
    if not hh_csv_files:
        return print("Нет доступных файлов с данными в папке ./data/hh/")
    else:
        options = [os.path.basename(f) for f in hh_csv_files]
        idx = pick(options, "Выберите файл с данными:", indicator='>')[1]
        file_path = hh_csv_files[idx]
        try:
            data = pd.read_csv(file_path, sep=';', encoding='utf-8-sig')
            return data
        except Exception as err:
            raise ValueError(f'Ошибка при загрузке данных из файла: {err}')

def select_from_dataframe(object, dataframe):
    """Выбор из DataFrame."""
    if dataframe is None or dataframe.empty:
        print(f"Нет доступных данных для выбора.")
        return None

    options = ['(Выбрать все)']
    options.extend([f"{idx+1}: {dataframe.iloc[idx]['name']}" for idx in range(len(dataframe))])
    idx = pick(options, f"Выберите данные:", indicator='>')[1]

    if idx == 0:
        items = []
        for i in range(len(dataframe)):
            if isinstance(object, ResumeData):
                item = ResumeData()
            elif isinstance(object, VacancyData):
                item = VacancyData()
            item.inject_data(dataframe.iloc[i])
            items.append(item)
        return items
    else:
        object.inject_data(dataframe.iloc[idx-1])
        return object

def get_available_regions():
    """Получает список доступных городов для поиска."""
    url = 'https://api.hh.ru/areas'
    data = get_hh_response(url)[1]
    filter_city = ['Москва', 'Санкт-Петербург']
    available_regions = []
    if data:
        json_data = json.loads(data)
        russian_areas = []
        for country in json_data:
            if country['name'] == 'Россия':
                russian_areas = country['areas']
                break
        
        for area in russian_areas:
            if area['name'] in filter_city:
                available_regions.append((area['id'], area['name']))
    return available_regions

def download_vacancies_from_hh():
    action = ''

    data_vacancy = None
    vacancies_objects = []

    print('Получаем список доступных регионов...')
    region_selection = get_available_regions()
    if len(region_selection) == 0:
        print('Нет доступных регионов для поиска.')
        return

    region_selection.append(('Exit', 'Выход из меню'))
    print('Список доступных регионов:')
    action = pick(region_selection, 'Пожалуйста, выберите регион (по умолчанию 1):', indicator='>')[1]

    if region_selection[action][0] != 'Exit':
        region_id = region_selection[action][0]
        print(f'Выбран регион: {region_selection[action][1]} (ID: {region_id})')

        user_text = input('Введите текст для поиска вакансий (по умолчанию "Data Science"): ')
        vacancy_url_list = get_vacancy_links_api(region_id, user_text)

        if not vacancy_url_list:
            raise ValueError(f'Не найдено вакансий для региона с ID {region_id}')

        vacancy_data_table = []
        for url in tqdm(vacancy_url_list, desc='Извлечение данных о вакансиях', unit='vacancy'):
            status, data = get_hh_response(url)
            if data != '':
                vacancy_data = json.loads(data)
                vacancy = VacancyData(vacancy_data)
                vacancy_data_table.append(vacancy.get_vacancy_data())
                vacancies_objects.append(vacancy)
                sleep(SLEEP_TIME)
            else:
                print(f'\nНе удалось извлечь данные из: {url}\n{status}')

        if vacancy_data_table:
            data_vacancy = pd.DataFrame(vacancy_data_table)
            data_vacancy = data_vacancy.replace({np.nan: None})

    if data_vacancy is not None:
        user_choise = input('Сохранить данные в файл? (y/n): ')
        if user_choise.lower() == 'y':
            save_to_csv(data_vacancy, 'vacancies', user_text, region_id)
    return data_vacancy, vacancies_objects

def download_resumes_from_hh(specific_links_resumes = []):
    action = ''
    region_id = 'variously_id'
    user_text = 'variously_text'

    data_resumes = None
    resumes_data_table = []
    resumes_objects = []
    resumes_url_list = []

    if not specific_links_resumes:
        print('Получаем список доступных регионов...')
        region_selection = get_available_regions()
        if len(region_selection) == 0:
            print('Нет доступных регионов для поиска.')
            return

        region_selection.append(('Exit', 'Выход из меню'))
        print('Список доступных регионов:')
        action = pick(region_selection, 'Пожалуйста, выберите регион (по умолчанию 1):', indicator='>')[1]

        if region_selection[action][0] != 'Exit':
            region_id = region_selection[action][0]
            print(f'Выбран регион: {region_selection[action][1]} (ID: {region_id})')

            user_text = input('Введите текст для поиска вакансий (по умолчанию "Data Science"): ')
            resumes_url_list = get_resume_links(region_id, user_text)
    else:
        resumes_url_list = specific_links_resumes

    if resumes_url_list:
        for url in tqdm(resumes_url_list, desc='Извлечение данных о резюме', unit='resume'):
            status, data = get_hh_response(url)
            if data != '':
                resume_data = BeautifulSoup(data, 'html.parser')
                resume = ResumeData(resume_data, url)
                if resume.is_valid():
                    resumes_data_table.append(resume.get_resume_data())
                    resumes_objects.append(resume)
            else:
                print(f'\nНе удалось извлечь данные из: {url}\n{status}')

        if resumes_data_table:
            data_resumes = pd.DataFrame(resumes_data_table)
            data_resumes = data_resumes.replace({np.nan: None})

    if data_resumes is not None:
        user_choise = input('Сохранить данные в файл? (y/n): ')
        if user_choise.lower() == 'y':
            save_to_csv(data_resumes, 'resumes', user_text, region_id)
    return data_resumes, resumes_objects

def preview_dataframe(df):
    """Предпросмотр DataFrame"""
    if df is None or df.empty:
        print(f"Нет данных для предпросмотра")
        return

    print(f"\n=== ПРЕДПРОСМОТР ДАННЫХ ===")
    print(f"Размер DataFrame: {df.shape[0]} строк, {df.shape[1]} столбцов")    
    print(f"\n--- Первые 5 записей ---")
    print(df.head())    
    print(f"\n--- Пропущенные значения ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])


def hh_menu():
    action = ''
    data_vacancy = None
    data_resumes = None
    specific_links_resumes = ['https://hh.ru/resume/e0030b08ff0ccd25890039ed1f4d706b6f636e']

    while action != 'Выход':
        select_mode = [
            'Скачать вакансии с HH.ru',
            'Загрузить вакансии из файла',
            'Скачать резюме с HH.ru',
            'Скачать резюме по ссылке',
            'Загрузить резюме из файла',
            'Выход'
        ]

        if action == '':
            action = pick(select_mode, 'Пожалуйста, выберите опцию:', indicator='>')[0]

        if action == 'Скачать вакансии с HH.ru':
            data_vacancy = download_vacancies_from_hh()[0]
            if data_vacancy is not None:
                print("Данные вакансий успешно загружены!")
                preview_dataframe(data_vacancy)
            input("Нажмите любую клавишу для продолжения...")
        elif action == 'Загрузить вакансии из файла':
            data_vacancy = load_from_csv('vacancies')
            if data_vacancy is not None:
                print("Данные вакансий успешно загружены!")
                preview_dataframe(data_vacancy)
            input("Нажмите любую клавишу для продолжения...")
        elif action == 'Скачать резюме с HH.ru':
            data_resumes = download_resumes_from_hh()[0]

        elif action == 'Скачать резюме по ссылке':
            data_resumes = download_resumes_from_hh(specific_links_resumes)[0]
            if data_resumes is not None:
                print("Резюме успешно обработано!")
                preview_dataframe(data_resumes)
            input("Нажмите любую клавишу для продолжения...")
        elif action == 'Загрузить резюме из файла':
            data_resumes = load_from_csv('resumes')
            if data_resumes is not None:
                print("Данные резюме успешно загружены!")
                preview_dataframe(data_resumes)
            input("Нажмите любую клавишу для продолжения...")
        else:
            print("Выход из модуля...")
            break
        action = ''
        continue


if __name__ == "__main__":
    hh_menu()