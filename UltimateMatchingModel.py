import numpy as np
import word2vec
import HH
import json
import os
import re

from pick import pick
from sklearn.metrics.pairwise import cosine_similarity

class UltimateMatcher:
    def __init__(self, word2vec_model, resume_data, vacancy_data):
        self.word2vec_model = word2vec_model
        self.resume_object = resume_data
        self.vacancy_object = vacancy_data
        self.description_matcher = Word2VecMatcher(self.word2vec_model)
        self.classification_matcher = SimplyClassificationMatcher()

        self.weights = {
            'semantic': 0.35,
            'multi_criteria': 0.25,
        }
        
    def ultimate_match(self):
        """Комплексное сопоставление"""
        results = {}
        # 1. Семантическое сходство описаний
        results['semantic'] = self.description_matcher.calculate_similarity(self.resume_object, self.vacancy_object)
        # 2. Многокритериальная оценка
        results['multi_criteria'] = self.classification_matcher.calculate_multi_criteria_score(self.resume_object, self.vacancy_object)

        final_score = sum(results[match] * self.weights[match] for match in results)

        return {'final_score': final_score,
            'component_scores': results,
            'confidence_level': self._calculate_confidence(results)
        }

    def _calculate_confidence(self, results):
        """Расчет уровня уверенности в результате"""
        scores = [v for k, v in results.items() if isinstance(v, (int, float))]
        #if len(scores) <= 1:
            #return 0.5
        
        # Низкая дисперсия = высокая уверенность
        variance = np.var(scores)
        confidence = max(0.1, 1.0 - variance)
        return min(confidence, 0.95)

class SimplyClassificationMatcher:
    def __init__(self):
        # Весовые коэффициенты для различных критериев
        self.weights = {
            'skills_match': 0.35,      # Совпадение навыков
            'experience_match': 0.25,   # Соответствие опыта
            'salary_match': 0.20,       # Соответствие зарплаты
            'location_match': 0.15,     # Соответствие местоположения
            'education_match': 0.05     # Соответствие образования
        }
                
    def calculate_multi_criteria_score(self, resume_data, vacancy_data):
        """Многокритериальная оценка совпадения"""
        criteria_scores = {}
        
        # 1. Оценка совпадения навыков
        criteria_scores['skills_match'] = self._calculate_skills_match(resume_data, vacancy_data)
        # 2. Оценка соответствия опыта работы
        criteria_scores['experience_match'] = self._calculate_experience_match(resume_data, vacancy_data)
        # 3. Оценка соответствия зарплатных ожиданий
        criteria_scores['salary_match'] = self._calculate_salary_match(resume_data, vacancy_data)
        # 4. Оценка соответствия местоположения
        criteria_scores['location_match'] = self._calculate_location_match(resume_data, vacancy_data)
        # 5. Оценка соответствия образования
        criteria_scores['education_match'] = self._calculate_education_match(resume_data)
        
        # Вычисляем взвешенную итоговую оценку
        final_score = sum(criteria_scores[criterion] * self.weights[criterion] for criterion in criteria_scores)
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_skills_match(self, resume_data, vacancy_data):
        """Вычисляет соответствие навыков между резюме и вакансией"""
        try:
            resume_skills = self._extract_skills(resume_data.get_field('skills'))
            vacancy_skills = self._extract_skills(vacancy_data.get_field('key_skills'))
            
            if not vacancy_skills and not resume_skills:
                return 0.5  # Нейтральная оценка если навыки не указаны
            
            if not vacancy_skills:
                return 0.7  # Если в вакансии не указаны навыки, но в резюме есть
            
            if not resume_skills:
                return 0.2  # Если в резюме нет навыков, а в вакансии есть
            
            # Точные совпадения
            exact_matches = resume_skills.intersection(vacancy_skills)
            skills_score = len(exact_matches) / len(vacancy_skills)
            
            return min(1.0, skills_score)
            
        except Exception as e:
            print(f"Ошибка при расчете совпадения навыков: {e}")
            return 0.0

    def _extract_skills(self, skills_data):
        """Извлекает набор навыков"""
        skills = set()
        
        if isinstance(skills_data, str):
            skills_list = [skill.strip().lower() for skill in skills_data.split(',')]
            skills.update(skill for skill in skills_list if skill)

        return skills

    def _calculate_experience_match(self, resume_data, vacancy_data):
        """Вычисляет соответствие опыта работы"""
        try:
            resume_exp_id = resume_data.get_field('experience_ID')
            vacancy_exp_id = vacancy_data.get_field('experience_id')
            
            if not resume_exp_id or not vacancy_exp_id:
                return 0.5  # Нейтральная оценка если данные отсутствуют
            
            # Получаем справочник опыта работы
            experience_dict = self._get_experience_mapping()
            
            resume_exp_level = experience_dict.get(resume_exp_id, 0)
            vacancy_exp_level = experience_dict.get(vacancy_exp_id, 0)
            
            # Вычисляем соответствие уровня опыта
            if resume_exp_level == vacancy_exp_level:
                return 1.0  # Точное совпадение
            elif abs(resume_exp_level - vacancy_exp_level) == 1:
                return 0.7  # Близкое совпадение
            elif resume_exp_level > vacancy_exp_level:
                return 0.8  # Опыт больше требуемого
            else:
                return 0.3  # Опыт меньше требуемого
                
        except Exception as e:
            print(f"Ошибка при расчете соответствия опыта: {e}")
            return 0.0

    def _get_experience_mapping(self):
        """Возвращает маппинг ID опыта работы на числовые уровни"""
        return {
            'noExperience': 0,     # Нет опыта
            'between1And3': 1,     # От 1 года до 3 лет
            'between3And6': 2,     # От 3 до 6 лет
            'moreThan6': 3         # Более 6 лет
        }


    def _calculate_salary_match(self, resume_data, vacancy_data):
        """Вычисляет соответствие зарплатных ожиданий"""
        try:
            resume_salary = resume_data.get_field('salary')
            vacancy_salary_from = vacancy_data.get_field('salary_from')
            vacancy_salary_to = vacancy_data.get_field('salary_to')
            
            if not resume_salary:
                return 0.5  # Если в резюме не указана зарплата
            
            if not vacancy_salary_from and not vacancy_salary_to:
                return 0.5  # Если в вакансии не указана зарплата
            
            # Определяем диапазон зарплаты в вакансии
            if vacancy_salary_from and vacancy_salary_to:
                vacancy_min = vacancy_salary_from
                vacancy_max = vacancy_salary_to
            elif vacancy_salary_from:
                vacancy_min = vacancy_salary_from
                vacancy_max = vacancy_salary_from * 1.3  # Предполагаем верхнюю границу
            else:
                vacancy_min = vacancy_salary_to * 0.8  # Предполагаем нижнюю границу
                vacancy_max = vacancy_salary_to
            
            # Вычисляем соответствие
            if vacancy_min <= resume_salary <= vacancy_max:
                return 1.0  # Зарплата в диапазоне
            elif resume_salary < vacancy_min:
                # Зарплатные ожидания ниже предложения
                diff_ratio = (vacancy_min - resume_salary) / vacancy_min
                return max(0.6, 1.0 - diff_ratio)
            else:
                # Зарплатные ожидания выше предложения
                diff_ratio = (resume_salary - vacancy_max) / vacancy_max
                return max(0.2, 1.0 - diff_ratio * 2)  # Более жесткая оценка
                
        except Exception as e:
            print(f"Ошибка при расчете соответствия зарплаты: {e}")
            return 0.0

    def _calculate_location_match(self, resume_data, vacancy_data):
        """Вычисляет соответствие местоположения"""
        try:
            resume_area = resume_data.get_field('area').lower().strip()
            vacancy_area = vacancy_data.get_field('area').lower().strip()
            
            if not resume_area or not vacancy_area:
                return 0.8  # Нейтральная оценка если данные отсутствуют
            
            if resume_area == vacancy_area:
                return 1.0
            else:
                return 0.3

        except Exception as e:
            print(f"Ошибка при расчете соответствия местоположения: {e}")
            return 0.0

    def _calculate_education_match(self, resume_data):
        """Вычисляет уровень образования"""
        try:
            resume_education = resume_data.get_field('education').lower()
            
            if not resume_education:
                return 0.5  # Нейтральная оценка если образование не указано
            
            # Определяем уровень образования
            if any(keyword in resume_education for keyword in ['университет', 'институт', 'академия']):
                return 1.0  # Высшее образование
            elif any(keyword in resume_education for keyword in ['колледж', 'техникум']):
                return 0.8  # Среднее специальное
            elif any(keyword in resume_education for keyword in ['школа', 'лицей', 'гимназия']):
                return 0.6  # Среднее образование
            else:
                return 0.5  # Неопределенное образование
                
        except Exception as e:
            print(f"Ошибка при расчете соответствия образования: {e}")
            return 0.0

class Word2VecMatcher:
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model

    def calculate_similarity(self, resume_data, vacancy_data):
        """Вычисление семантического сходства между резюме и вакансией"""
        try:
            resume_vector = self.text_to_vector(self._extract_text(resume_data))
            vacancy_vector = self.text_to_vector(self._extract_text(vacancy_data))

            # Проверяем, что векторы получены успешно
            if resume_vector is None or vacancy_vector is None:
                print("Предупреждение: Не удалось получить векторы для резюме или вакансии")
                return 0.0
            
            # Проверяем размерности векторов
            if resume_vector.size == 0 or vacancy_vector.size == 0:
                print("Предупреждение: Получены пустые векторы")
                return 0.0
            
            # Проверяем, что векторы имеют одинаковую размерность
            if resume_vector.shape != vacancy_vector.shape:
                print(f"Предупреждение: Векторы имеют разные размерности: {resume_vector.shape} vs {vacancy_vector.shape}")
                return 0.0
            
            # Проверяем на NaN и бесконечность
            if np.any(np.isnan(resume_vector)) or np.any(np.isnan(vacancy_vector)):
                print("Предупреждение: Обнаружены NaN значения в векторах")
                return 0.0
            
            if np.any(np.isinf(resume_vector)) or np.any(np.isinf(vacancy_vector)):
                print("Предупреждение: Обнаружены бесконечные значения в векторах")
                return 0.0
            
            # Reshape для cosine_similarity (должны быть 2D массивы)
            resume_vector = resume_vector.reshape(1, -1)
            vacancy_vector = vacancy_vector.reshape(1, -1)
            
            # Вычисляем косинусное сходство
            similarity = cosine_similarity(resume_vector, vacancy_vector)[0][0]
            
            # Проверяем результат на валидность
            if np.isnan(similarity) or np.isinf(similarity):
                print("Предупреждение: Получено невалидное значение сходства")
                return 0.0
            
            return max(0.0, min(1.0, similarity))  # Ограничиваем значение между 0 и 1
            
        except Exception as e:
            print(f"Ошибка при вычислении сходства: {e}")
            return 0.0


    def text_to_vector(self, text):
        """Преобразование текста в вектор используя Word2Vec"""
        if not self.word2vec_model or not text:
            return None

        words = self._clean_text(text).split()
        if not words:
            return None
        
        word_vectors = []

        for word in words:
            try:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            except (KeyError, AttributeError):
                continue

        if not word_vectors:
            return np.zeros(self.word2vec_model.vector_size if hasattr(self.word2vec_model, 'vector_size') else 100)

        word_vectors = np.array(word_vectors)

        if word_vectors.ndim == 1:
            return word_vectors
        elif word_vectors.ndim == 2:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size if hasattr(self.word2vec_model, 'vector_size') else 100)

    def _extract_text(self, HH_data):
        """Извлекает текст из данных для анализа"""
        text_parts = []
        
        if isinstance(HH_data, HH.ResumeData):
            fields_to_extract = ['description', 'description_about_me']
        elif isinstance(HH_data, HH.VacancyData):
            fields_to_extract = ['description']

        for field in fields_to_extract:
            try:
                field_value = HH_data.get_field(field)
                if field_value and isinstance(field_value, str):
                    text_parts.append(field_value)
            except (AttributeError, KeyError):
                continue

        full_text = ' '.join(text_parts)
        full_text = self._clean_text(full_text)
        return full_text

    def _clean_text(self, text):
        """Очистка и нормализация текста"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower().strip()
        return text
    

def load_word2vec_model():
    """Загрузка обученной модели Word2Vec"""
    word2vec_model = None
    try:
        word2vec_model = word2vec.load_model()
        if word2vec_model is None:
            print("Не удалось загрузить модель Word2Vec")
        else:
            print("Модель Word2Vec загружена успешно")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
    return word2vec_model

def load_data(data_type='vacancies', proc_type='download'):
    """Обработка входящих данных"""
    data_from_csv = None
    input_object = None
    objects_list = []
    if proc_type == 'download':
        if data_type == 'vacancies':
            input_object = HH.download_vacancies_from_hh()[1]
        elif data_type == 'resumes':
            input_object = HH.download_resumes_from_hh()[1]
    elif proc_type == 'file':
        data_from_csv = HH.load_from_csv(data_type)
        if data_from_csv is not None:
            if data_type == 'vacancies':
                input_object = HH.select_from_dataframe(HH.VacancyData(), data_from_csv)
            elif data_type == 'resumes':
                input_object = HH.select_from_dataframe(HH.ResumeData(), data_from_csv)

    if input_object is not None:
        objects_list = input_data(input_object)
    else:
        print(f"Не удалось загрузить данные {data_type}.")
    return objects_list

def input_data(data):
    """Загрузка данных"""
    objects_list = []
    try:
        if isinstance(data, list):
            for item in data:
                if isinstance(item, HH.ResumeData) or isinstance(item, HH.VacancyData):
                    objects_list.append(item)
            print(f"Загружено {len(data)} элементов")
        elif isinstance(data, HH.VacancyData) or isinstance(data, HH.ResumeData):
            objects_list.append(data)
            print(f"{data.__class__.__name__} добавлено в список: {data.get_header_text()}")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
    return objects_list

def initialize_necessary_components():
    word2vec_model = load_word2vec_model()
    resume_objects_list = load_data('resumes', 'file')
    vacancies_objects_list = load_data('vacancies', 'file')
    return word2vec_model, resume_objects_list, vacancies_objects_list


def run_ultimate_matching():
    word2vec_model, resume_objects_list, vacancies_objects_list = initialize_necessary_components()

    if not word2vec_model:
        print("Сначала загрузите модель Word2Vec")
        return

    if not resume_objects_list or not vacancies_objects_list:
        print("Загрузите резюме и вакансии")
        return
    
    matches = []
    experience_dict = HH.get_hh_dictionaries('experience')

    print(f"Выполнение сопоставления резюме и вакансий...")
    for resume in resume_objects_list:
        best_matches = []

        for vacancy in vacancies_objects_list:
            matcher = UltimateMatcher(word2vec_model, resume, vacancy)
            match_result = matcher.ultimate_match()
            
            if match_result['final_score'] > 0.5:
                best_matches.append({
                    'vacancy_id': vacancy.get_field('vacancy_id'),
                    'vacancy_name': vacancy.get_header_text(),
                    'similarity': round(match_result['final_score'], 4),
                    'confidence_level': round(match_result['confidence_level'], 4),
                    'company': vacancy.get_field('employer_name'),
                    'area': vacancy.get_field('area'),
                    'url': vacancy.get_field('url'),
                    'experience': get_hh_experience_by_id(experience_dict, vacancy.get_field('experience_id')),
                    'component_scores': match_result.get('component_scores', {})
                })

        if best_matches:
            matches.append({
                'resume_id': resume.get_field('resume_id'),
                'resume_name': resume.get_field('name'),
                'url': resume.get_field('url'),
                'matches': best_matches,
                'total_matches_found': len(best_matches)
            })    
    return matches


def get_hh_experience_by_id(experience_dict, experience_id):
    """Возвращает название опыта работы по его ID."""
    try:
        for exp in experience_dict:
            if exp.get('id') == experience_id:
                return exp.get('name')
    except:
        return None

def display_matches(matches):
        """Отображение результатов сопоставления"""
        if not matches:
            print("Совпадений не найдено")
            return
        
        total_resumes = len(matches)
        total_matches = sum(len(match['matches']) for match in matches)
        
        print(f"\n{'='*80}")
        print(f"РЕЗУЛЬТАТЫ СОПОСТАВЛЕНИЯ")
        print(f"{'='*80}")
        print(f"Обработано резюме: {total_resumes}")
        print(f"Всего найдено совпадений: {total_matches}")
        print(f"{'='*80}")
        
        for match in matches:
            print(f"\n📄 РЕЗЮМЕ: {match['resume_name']}")
            print(f"   ID на HeadHunter: {match['resume_id']}")
            print(f"   Ссылка: {match['url']}")

            if not match['matches']:
                print("❌ Подходящих вакансий не найдено")
                continue
            else:
                print(f"   Найдено совпадений: {match['total_matches_found']}")
            
            print(f"{'─'*80}")

            for i, vacancy_match in enumerate(match['matches'], 1):
                similarity_percent = vacancy_match['similarity'] * 100
                
                # Определяем уровень совпадения
                if similarity_percent >= 80:
                    level = "🟢 ОТЛИЧНОЕ"
                elif similarity_percent >= 60:
                    level = "🟡 ХОРОШЕЕ"
                elif similarity_percent >= 40:
                    level = "🟠 СРЕДНЕЕ"
                else:
                    level = "🔴 СЛАБОЕ"
                
                print(f"\n   {i}. 💼 {vacancy_match['vacancy_name']}")
                print(f"      🏢 Компания: {vacancy_match['company']}")
                print(f"      📍 Регион: {vacancy_match['area']}")
                print(f"      📊 Сходство: {similarity_percent:.1f}% ({vacancy_match['confidence_level']:.1f}) ({level})")
                print(f"      🎯 Опыт: {vacancy_match['experience']}")
                print(f"      🔗 Ссылка: {vacancy_match['url']}")
                print(f"      📝 Компоненты: {', '.join([f'{comp}: {score:.1f}' for comp, score in vacancy_match['component_scores'].items()])}")

        print(f"\n{'='*80}")
        print("Совет: Рассматривайте вакансии с сходством выше 60% как приоритетные")
        print(f"{'='*80}")

def ultimate_matching_menu():
    action = ''

    while action != 'Выход':
        select_mode = [
            'Выполнить сопоставление',
            'Выход'
        ]

        if action == '':
            action = pick(select_mode, 'Пожалуйста, выберите опцию:', indicator='>')[0]
            
        try:
            if action == 'Выполнить сопоставление':
                matches = run_ultimate_matching()
                display_matches(matches)
                input("Нажмите любую клавишу для продолжения...")
                action = ''
            else:
                print("Выход из модуля...")
                break
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            input("Нажмите любую клавишу для продолжения...")
            action = ''


if __name__ == "__main__":
    ultimate_matching_menu()