import gensim.downloader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


print("loading models and calculating data...")
# Load the pre-trained Word2Vec model
model_1 = gensim.downloader.load("word2vec-google-news-300")

model_from_diff_corpus_1 = gensim.downloader.load("fasttext-wiki-news-subwords-300")
model_from_diff_corpus_2 = gensim.downloader.load("glove-twitter-25")


model_same_corpus_100 = gensim.downloader.load("glove-wiki-gigaword-100")
model_same_corpus_300 = gensim.downloader.load("glove-wiki-gigaword-300")


# Load the Synonym Test dataset
synonym_test_dataset = pd.read_csv("synonym.csv")


# Function to compute cosine similarity between two words
def compute_similarity(word1, word2, model):
    try:
        embedding1 = model[word1]
        embedding2 = model[word2]
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity
    except KeyError:
        return 0.0  # Return 0 if either word is not in the model


# Function to find the closest synonym for a question word
def find_closest_synonym(question_word, guess_words, model):
    similarities = [
        (guess_word, compute_similarity(question_word, guess_word, model))
        for guess_word in guess_words
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[0][0]  # Return the word with the highest similarity


# Evaluate the model and store the results in a CSV file
def evaluate_model(modelName, model, corpora_size):
    correct_labels = 0
    questions_without_guessing = 0

    with open(modelName, "w") as details_file:
        details_file.write("question_word,correct_answer_word,guess_word,label\n")

        for _, row in synonym_test_dataset.iterrows():
            question_word = row["question"]
            correct_answer_word = row["answer"]
            guess_words = row[["0", "1", "2", "3"]].dropna().tolist()

            if question_word in model.key_to_index and guess_words:
                system_guess_word = find_closest_synonym(
                    question_word, guess_words, model
                )
                label = (
                    "correct" if system_guess_word == correct_answer_word else "wrong"
                )
                correct_labels += 1 if label == "correct" else 0
                questions_without_guessing += 1  # Increment for every question
            else:
                system_guess_word = ""
                label = "guess"

            details_file.write(
                f"{question_word},{correct_answer_word},{system_guess_word},{label}\n"
            )

    # Write analysis results to a CSV file
    total_questions = len(synonym_test_dataset)
    accuracy = (
        correct_labels / questions_without_guessing
        if questions_without_guessing > 0
        else 0
    )

    # Write the results
    with open("analysis.csv", "a") as analysis_file:
        analysis_file.write(
            f"{corpora_size},{len(model.key_to_index)},{correct_labels},{questions_without_guessing},{accuracy}\n"
        )


# Task 1
evaluate_model("word2vec-google-news-300-details.csv", model_1, "Google-News-300")


# Different Corpora same size
evaluate_model(
    "fasttext-wiki-news-subwords-300-details.csv",
    model_from_diff_corpus_1,
    "Wikipedia-300",
)

evaluate_model(
    "glove-twitter-25-details.csv", model_from_diff_corpus_2, "Twitter-25"
)  # this is no good I need to find a 300 size model b4 demo or submission


# Same Corpora different size
evaluate_model(
    "glove-wiki-gigaword-100-details.csv", model_same_corpus_100, "Wiki-GigaWord-100"
)
evaluate_model(
    "glove-wiki-gigaword-300-details.csv", model_same_corpus_300, "Wiki-GigaWord-300"
)
