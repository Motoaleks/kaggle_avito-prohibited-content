import string
from nltk.stem.snowball import RussianStemmer
from sklearn.feature_extraction.text import CountVectorizer

# стемминг и знаки пунктуации
stemmer = RussianStemmer()
exclude = string.punctuation + string.digits


# Преобразование строки в массив слов со стеммингом и lower()
def clear(text):
    temp = ''.join(text).translate(exclude).lower()
    temp = temp.split()
    temp = [stemmer.stem(i) for i in temp]
    temp = [i for i in temp if len(i) > 2]
    return temp


# Создание таблицы BagOfWords из колонки
def frame(df, column):
    print("COLUMN: {0}".format(column))
    df[column] = df[column].apply(lambda x: clear(x))
    print("-Cleared")
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=None)
    words = [" ".join(row) for row in df[column]]
    print("-Words extracted")
    features = vectorizer.fit_transform(words)
    print("-Processed\n")
    return features
