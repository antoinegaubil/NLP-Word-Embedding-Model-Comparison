import gensim.downloader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained Word2Vec model
model = gensim.downloader.load('word2vec-google-news-300')

# Load the Synonym Test dataset
synonym_test_dataset = pd.read_csv('synonym.csv')


# Function to compute cosine similarity between two words
def compute_similarity(word1, word2):
    try:
        embedding1 = model[word1]
        embedding2 = model[word2]
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity
    except KeyError:
        return 0.0  # Return 0 if either word is not in the model


# Function to find the closest synonym for a question word
def find_closest_synonym(question_word, guess_words):
    similarities = [(guess_word, compute_similarity(question_word, guess_word)) for guess_word in guess_words]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[0][0]  # Return the word with the highest similarity


# Evaluate the model and store the results in a CSV file
correct_labels = 0
questions_without_guessing = 0

with open('word2vec-google-news-300-details.csv', 'w') as details_file:
    details_file.write("question_word,correct_answer_word,guess_word,label\n")

    for _, row in synonym_test_dataset.iterrows():
        question_word = row['question']
        correct_answer_word = row['answer']
        guess_words = row[['0', '1', '2', '3']].dropna().tolist()

        if question_word in model.key_to_index and guess_words:
            system_guess_word = find_closest_synonym(question_word, guess_words)
            label = 'correct' if system_guess_word == correct_answer_word else 'wrong'
            correct_labels += 1 if label == 'correct' else 0
        else:
            system_guess_word = ''
            label = 'guess'

        details_file.write(f"{question_word},{correct_answer_word},{system_guess_word},{label}\n")

        questions_without_guessing += 1 if label == 'correct' else 0

# Write analysis results to a CSV file
total_questions = len(synonym_test_dataset)
accuracy = correct_labels / questions_without_guessing if questions_without_guessing > 0 else 0

with open('analysis.csv', 'w') as analysis_file:
    analysis_file.write(f"word2vec-google-news-300,{len(model.key_to_index)},{correct_labels},{questions_without_guessing},{accuracy}\n")
