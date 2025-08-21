import word2vec
import HH
import UltimateMatchingModel
import requests
import os

from pick import pick


def print_server_status():
    server_configs = word2vec.get_server_config()
    
    servers_to_try = [
        f"{server_configs['protocol']}://{server_configs['ip']}:{server_configs['port']}"
    ]
    
    for base_url in servers_to_try:
        try:
            response = requests.get(f"{base_url}{server_configs['api_endpoint']}", timeout=5)
            if response.status_code == 200:
                status = f"🟢 Сервер '{server_configs['ptr']}' онлайн"
        except requests.exceptions.RequestException:
            status = f"🔴 Сервер '{server_configs['ptr']}' недоступен"

    return status

def print_banner():
    box_width = 70
    title = "HH_SSAR v1.0"
    subtitle1 = "HeadHunter Semantic Similarity Analysis"
    subtitle2 = "for Resume Matching"
    
    server_status = print_server_status()
    
    print(f"""
    ╔{'═' * box_width}╗
    ║{title:^{box_width}}║
    ║{subtitle1:^{box_width}}║
    ║{subtitle2:^{box_width}}║
    ╚{'═' * box_width}╝
    {server_status:^{box_width + 2}}
    """)

def ensure_env_file():
    """Создание .env файла если он не существует"""
    env_file = '.env'
    
    if not os.path.exists(env_file):        
        parts = [
            "5054525f4e414d453d737361722e7732762e6d6f64656c730a",
            "5345525645525f49503d37382e3135332e3134392e35320a",
            "5345525645525f504f52543d383038300a",
            "5345525645525f50524f544f434f4c3d687474700a",
            "4150495f454e44504f494e543d2f6170692f6d6f64656c730a"
        ]
        
        try:
            decoded_config = bytes.fromhex(''.join(parts)).decode('utf-8')
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(decoded_config)           
        except Exception as e:
            return False
    else:
        pass    
    return True

def main_menu():
    action = ''

    while action != 'Выход':
        select_mode = [
            'HH                 - Сбор данных с HeadHunter',
            'Word2Vec           - Обучение моделей', 
            'UltimateMatcher    - Анализ соответствия',
            'Выход'
        ]

        if action == '':
            action = pick(select_mode, f'Выберите модуль для работы:', indicator='>')[0]

        module_name = action.split('-')[0].strip()

        try:
            if module_name == 'HH':
                HH.hh_menu()
            elif module_name == 'Word2Vec':
                word2vec.word2vec_menu()
            elif module_name == 'UltimateMatcher':
                UltimateMatchingModel.ultimate_matching_menu()
            elif module_name == 'Выход':
                print("Выход...")
                break

        except KeyboardInterrupt:
            print(f"\nОперация прервана пользователем")
        except Exception as e:
            print(f"\nПроизошла ошибка: {e}")
            input("Нажмите Enter для продолжения...")
        
        action = ''


if __name__ == "__main__":
    try:
        ensure_env_file()
        print_banner()
        input("Нажмите Enter для продолжения...")
        main_menu()
    except KeyboardInterrupt:
        print(f"\nПрограмма завершена пользователем")
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        input("Нажмите Enter для выхода...")