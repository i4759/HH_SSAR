import numpy as np
import word2vec
import HH
import json
import os

from pick import pick
from sklearn.metrics.pairwise import cosine_similarity

class SemanticMatcher:
    def __init__(self):
        self.model = None
        self.resumes = []
        self.vacancies = []
    
    def load_word2vec_model(self):
        """Загрузка обученной модели Word2Vec"""
        load_status = False
        try:
            self.model = word2vec.load_model()
            if self.model is None:
                print("Не удалось загрузить модель Word2Vec")
            else:
                print("Модель Word2Vec загружена успешно")
                load_status = True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
        return load_status

    def input_resume(self, resume_data):
        """Загрузка резюме"""
        try:
            if isinstance(resume_data, list):
                self.resumes = resume_data
                print(f"Загружено {len(resume_data)} резюме")
            else:
                self.resumes.append(resume_data)
                print(f"Резюме добавлено: {resume_data.get('title', '---')} ({resume_data.get('resume_id', '---')})")
        except Exception as e:
            print(f"Ошибка загрузки резюме: {e}")

    def input_vacancy(self, vacancy_data):
        """Загрузка вакансий"""
        try:
            if isinstance(vacancy_data, list):
                self.vacancies = vacancy_data
                print(f"Загружено {len(vacancy_data)} вакансий")
            else:
                self.vacancies.append(vacancy_data)
                print(f"Вакансия добавлена: {vacancy_data.get('name', '---')} ({vacancy_data.get('id', '---')})")
        except Exception as e:
            print(f"Ошибка загрузки вакансий: {e}")
    
    def text_to_vector(self, text):
        """Преобразование текста в вектор используя Word2Vec"""
        if not self.model:
            return None
        
        words = text.lower().split()
        vectors = []
        
        for word in words:
            try:
                vector = self.model[word]
                vectors.append(vector)
            except KeyError:
                continue
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.vector_size)

    def calculate_similarity(self, resume_text, vacancy_text):
        """Вычисление семантического сходства между резюме и вакансией"""
        resume_vector = self.text_to_vector(resume_text)
        vacancy_vector = self.text_to_vector(vacancy_text)
        
        if resume_vector is None or vacancy_vector is None:
            return 0.0
        
        resume_vector = resume_vector.reshape(1, -1)
        vacancy_vector = vacancy_vector.reshape(1, -1)
        
        similarity = cosine_similarity(resume_vector, vacancy_vector)[0][0]
        return similarity
    
    def match_resumes_to_vacancies(self, threshold=0.5):
        """Сопоставление резюме с вакансиями"""
        if not self.model:
            print("Сначала загрузите модель Word2Vec")
            return
        
        if not self.resumes or not self.vacancies:
            print("Загрузите резюме и вакансии")
            return
        
        matches = []
        
        for i, resume in enumerate(self.resumes):
            resume_text = f"{resume.get('title', '')} {resume.get('skills', '')} {resume.get('experience', '')}"
            best_matches = []
            
            for j, vacancy in enumerate(self.vacancies):
                vacancy_text = f"{vacancy.get('name', '')} {vacancy.get('description', '')} {vacancy.get('key_skills', '')}"
                
                similarity = self.calculate_similarity(resume_text, vacancy_text)
                
                if similarity >= threshold:
                    best_matches.append({
                        'vacancy_id': j,
                        'vacancy_name': vacancy.get('name', 'Без названия'),
                        'similarity': similarity,
                        'company': vacancy.get('employer', {}).get('name', 'Неизвестно')
                    })
            
            # Сортировка по убыванию сходства
            best_matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            matches.append({
                'resume_id': i,
                'resume_title': resume.get('title', 'Без названия'),
                'matches': best_matches[:5]  # Топ 5 совпадений
            })
        
        return matches
    
    def display_matches(self, matches):
        """Отображение результатов сопоставления"""
        if not matches:
            print("Совпадений не найдено")
            return
        
        for match in matches:
            print(f"\n{'='*60}")
            print(f"Резюме: {match['resume_title']}")
            print(f"{'='*60}")
            
            if not match['matches']:
                print("Подходящих вакансий не найдено")
                continue
            
            for i, vacancy_match in enumerate(match['matches'], 1):
                print(f"{i}. {vacancy_match['vacancy_name']}")
                print(f"   Компания: {vacancy_match['company']}")
                print(f"   Сходство: {vacancy_match['similarity']:.3f}")
                print()


def proc_inputs_vacancies(matcher, proc_type='download'):
    """Обработка входных вакансий"""
    if proc_type == 'download':
        vacancies_objects = HH.download_vacancies_from_hh()[1]
    elif proc_type == 'file':
        vacancies_data = HH.load_from_csv('vacancies')
        if vacancies_data is not None:
            vacancies_objects = HH.select_from_dataframe(HH.VacancyData(), vacancies_data)

    if vacancies_objects is not None:
        matcher.input_vacancy(vacancies_objects)
    else:
        print("Не удалось загрузить данные вакансий.")
    return matcher

def proc_inputs_resumes(matcher, proc_type='download'):
    """Обработка входных резюме"""
    if proc_type == 'download':
        resume_objects = HH.download_resumes_from_hh()[1]
    elif proc_type == 'file':
        resume_data = HH.load_from_csv('resumes')
        if resume_data is not None:
            resume_objects = HH.select_from_dataframe(HH.ResumeData(), resume_data)

    if resume_objects is not None:
        matcher.input_resume(resume_objects)
    else:
        print("Не удалось загрузить данные резюме.")
    return matcher


def semantic_matching_menu():
    action = ''
    matcher = SemanticMatcher()

    while action != 'Выход':
        select_mode = [
            'Загрузить модель Word2Vec',
            'Загрузить вакансии',
            'Загрузить резюме',
            'Выход'
        ]

        if action == '':
            action = pick(select_mode, 'Пожалуйста, выберите опцию:', indicator='>')[0]


        if action == 'Загрузить модель Word2Vec':
            model_status = matcher.load_word2vec_model()
            if not model_status:
                print("Не удалось загрузить модель Word2Vec. Войдите в модуль Word2Vec и обучите модель.")
            input("Нажмите любую клавишу для продолжения...")
        elif action == 'Загрузить вакансии':
            action = pick(['Загрузить вакансии из файла', 'Скачать вакансии с HH.ru', 'Назад'], 'Пожалуйста, выберите опцию:', indicator='>')[0]
            if action == 'Загрузить вакансии из файла':
                matcher = proc_inputs_vacancies(matcher, 'file')
            elif action == 'Скачать вакансии с HH.ru':
                matcher = proc_inputs_vacancies(matcher, 'download')
            else:
                pass
        elif action == 'Загрузить резюме':
            action = pick(['Загрузить резюме из файла', 'Скачать резюме с HH.ru', 'Назад'], 'Пожалуйста, выберите опцию:', indicator='>')[0]
            if action == 'Загрузить резюме из файла':
                matcher = proc_inputs_resumes(matcher, 'file')
            elif action == 'Скачать резюме с HH.ru':
                matcher = proc_inputs_resumes(matcher, 'download')
            else:
                pass
        else:
            print("Выход из модуля...")
            break
        action = ''
        continue


if __name__ == "__main__":
    semantic_matching_menu()