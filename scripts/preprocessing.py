import string
from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer

# стемминг и знаки пунктуации
stemmer = RussianStemmer()  # стеммер слов, объект
exclude = string.punctuation + string.digits  # буквы, знаки пунктуации для исключения из итоговых слов
stopwords = set(stopwords.words(
    "russian"))  # стоп-слова, убираемые из общего пула слов, т.к. в данной задаче не несут в себе полезной нагрузке


# Преобразование строки в массив слов со стеммингом и lower()
def clear(text):
    # pre. Проверка на нули в данных
    text = str(text)
    if text == "nan":
        return []
    # 1. Убираем не-буквы
    temp = re.sub("[^a-zA-Z|^а-яА-Я]", " ", text)
    # 2. Преобразуем в прописные и делим по словам
    temp = temp.lower().split()
    # 3. Стемминг и уборка стоп-слов
    temp = [stemmer.stem(i) for i in temp if i not in stopwords]
    temp = [i for i in temp if len(i) > 2]
    return temp


# Создание таблицы BagOfWords из колонки
def frame(df, column, max_features=5000):
    print("COLUMN: {0}".format(column))
    # 1. Получаем очищенные данные и представляем строчкой
    cleared = []
    if type(column) is str:  # обработка одной колонки
        cleared = [" ".join(clear(i)) for i in df[column]]
    else:  # обработка 2 колонок
        temp = [series_.values for id_, series_ in df[column].iterrows()]
        temp = [" ".join(clear(str(i) + str(j))) for i, j in temp]
        cleared = cleared + temp
    print("- Cleared")
    # 2. Создаём CountVectorizer - подсчёта слов
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=max_features)
    print("- Words extracted")
    # 3. Учим словарю и обрабатываем 
    features = vectorizer.fit_transform(cleared)
    print("- Processed\n")
    return features
