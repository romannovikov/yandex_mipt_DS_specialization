import re
import numpy as np
from scipy.spatial.distance import cosine


def main(file):
    sentences = read_data(file)
    tokenized_sentences = tokenize(sentences)
    vocabulary = make_vocabulary(sentences)
    bag_of_words = generate_bow(tokenized_sentences, vocabulary)

    neighbors = nearest_neighbors(bag_of_words)
    print(neighbors)


def read_data(file):
    """Функция, читающая файл и возвращающая список предложений в нем"""
    with open(file) as file:
        sentences = [sentence.strip().lower() for sentence in file]
        return sentences


def tokenize(sentences):
    """Функция токенизации предложений"""
    tokenized_sentences = []
    for sentence in sentences:
        # токенизируем предложение с помощью регулярных выражений
        tokenized_sentence = [word for word in re.split('[^a-z]', sentence) if word]
        tokenized_sentences.append(tokenized_sentence)
    return tokenized_sentences


def make_vocabulary(sentences):
    """Функция для создания словаря"""
    vocabulary = list(enumerate(sorted({word \
                                        for sentence in sentences \
                                        for word in re.split('[^a-z]', sentence) \
                                        if word})))
    return vocabulary


def generate_bow(tokenized_sentences, vocabulary):
    """Функция для генерации мешка слов"""
    bag_of_words = []
    for sentence in tokenized_sentences:
        # инициализируем вектор нулевым с длинной, равной размеру словаря
        bag_vector = np.zeros(len(vocabulary))
        for word in sentence:
            for i, w in vocabulary:
                if word == w:
                    bag_vector[i] += 1
        bag_of_words.append(bag_vector)
    return bag_of_words


def nearest_neighbors(bag_of_words):
    """Функция, возвращающая два ближайших по косинусной метрике предложения к самому первому в исходном тексте"""
    distances = sorted([(i, cosine(bag_of_words[0], sentence)) for (i, sentence) in enumerate(bag_of_words)],
                       key=lambda x: x[1])[1:]
    nearest_neighbors = [str(distance[0]) for distance in distances][:2]

    with open('submission-1.txt', 'w') as file:
        file.write(' '.join(nearest_neighbors))
    return ' '.join(nearest_neighbors)


if __name__ == '__main__':
    main('sentences.txt')
