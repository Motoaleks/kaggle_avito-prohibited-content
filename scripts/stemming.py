import re
import sys
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer

stemmer = RussianStemmer()
stopwords = set(stopwords.words('russian'))


def clear(text):
    """
    Cтемминг строки
    + удаление знаков пунктуации
    + преобразование к мелкому шрифту
    + удаление стоп-слов
    + удаление слов короче 2 букв
    :param text: строчка для преобразования
    :return: обработанная строчка
    """
    # pre. Проверка на нули в данных
    text = str(text)
    if text == "nan":
        return text
    # 1. Убираем не-буквы
    temp = re.sub("[^a-zA-Z|^а-яА-Я]", " ", text)
    # 2. Преобразуем в прописные и делим по словам
    temp = temp.lower().split()
    # 3. Стемминг и уборка стоп-слов
    temp = [stemmer.stem(i) for i in temp if i not in stopwords and len(i) > 2]
    return " ".join(temp)


def main(args):
    """
    Обрабатывает поданый на вход файл и производит очистку(стемминг, стоп-слова, пропись, короткие слова) слов
    указанных столбов.
    :param args:
    [0] - путь до файла
    [1] - list() столбов для очистки
    """
    path = args[0]
    attrs = str(args[1]).strip("[]").replace("'", "").split(",")
    print("CLEARING DATA.........")
    df = pd.DataFrame.from_csv(path, sep='\t')
    for column in attrs:
        print("COLUMN: {0}".format(column))
        if type(column) is str:  # обработка одной колонки
            cleared = [clear(i) for i in df[column]]
            df[column] = cleared
        else:  # обработка 2 колонок
            for col in column:
                cleared = [clear(i) for i in df[column]]
                df[col] = cleared
        print("  - Cleared")
    df.to_csv('cleared.csv')
    print("  - Saved")
    print("CLEARING FINISHED.....")
    sys.exit()


if __name__ == '__main__':
    main(sys.argv[1:])
