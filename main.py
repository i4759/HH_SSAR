import word2vec
import HH

from pick import pick

if __name__ == "__main__":
    action = ''

    while action != 'Выход':
        select_mode = ['HH', 'Word2Vec', 'Выход']

        if action == '':
            action = pick(select_mode, 'Пожалуйста, выберите опцию:', indicator='>')[0]

        if action == 'HH':
            HH.hh_menu()
            action = ''
            continue
        elif action == 'Word2Vec':
            word2vec.word2vec_menu()
            action = ''
            continue
        else:
            print("Выход...")
            break