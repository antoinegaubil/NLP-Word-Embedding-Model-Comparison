import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
from gensim.models import Word2Vec
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):

    sentences = sent_tokenize(text)

    processed_sentences = []
    
    # Define set of stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower()) 
        words = [word for word in words if word not in stop_words and word not in punctuation]
        processed_sentences.append(words)
    return processed_sentences

books_directory = './books'
all_sentences = []

# Process each book in the directory
for filename in os.listdir(books_directory):
    if filename.endswith(".txt"):
        print(filename)
        with open(os.path.join(books_directory, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            preprocessed_sentences = preprocess_text(text)
            all_sentences.extend(preprocessed_sentences)


window_sizes = [20, 10] 
embedding_sizes = [50, 300]

synonym_test_data = pd.read_csv("synonym.csv")
question_column = 'question'
answer_column = 'answer'
guess_columns = ['0', '1', '2', '3']

for window in window_sizes:
    for size in embedding_sizes:
        model = Word2Vec(all_sentences, window=window, vector_size=size)

        all_details = []

        correct_labels = 0
        questions_without_guessing = 0

        for index, row in synonym_test_data.iterrows():
            question_word = row[question_column]
            correct_answer_word = row[answer_column]

            # Initialize variables to store closest guess and its similarity score
            closest_guess = ''
            closest_similarity = -1

            if any(word in model.wv for word in row[guess_columns]) and correct_answer_word in model.wv:

                for guess_column in guess_columns:
                    guess_word = row[guess_column]

                    if guess_word in model.wv and question_word in model.wv:
                        # Calculate similarity between question and guess word
                        similarity = model.wv.similarity(question_word, guess_word)

                        # Update closest guess if similarity is higher
                        if similarity > closest_similarity:
                            closest_similarity = similarity
                            closest_guess = guess_word

                if closest_guess:
                    if closest_guess == correct_answer_word:
                        label = 'correct'
                        correct_labels += 1
                    else:
                        label = 'wrong'
                    questions_without_guessing += 1
                else:
                    label = 'guess'
            else:
                label = 'guess'

            all_details.append([question_word, correct_answer_word, closest_guess, label])

        # Calculate accuracy
        accuracy = correct_labels / questions_without_guessing if questions_without_guessing > 0 else 0

        # Write details to <model name>-details.csv file
        model_name = f"books-E{size}-W{window}"
        details_df = pd.DataFrame(all_details, columns=['question_word', 'correct_answer_word', 'guess_word', 'label'])
        details_df.to_csv(f"{model_name}-details.csv", index=False)

        # Write analysis results to analysis.csv
        with open("analysis.csv", "a") as analysis_file:
            analysis_file.write(f"{model_name},{len(model.wv.key_to_index)},{correct_labels},{questions_without_guessing},{accuracy}\n")

data = pd.read_csv('analysis.csv')

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
plt.bar(df['model_name'], df['accuracy'], color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison by Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show plot
plt.show()
