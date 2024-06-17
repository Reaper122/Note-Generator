import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Load pre-trained SpaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text and extract entities
def preprocess_text(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# Function to perform topic modeling
def perform_topic_modeling(texts):
    # Convert texts to document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=0.05, stop_words='english')
    dtm = vectorizer.fit_transform(texts)

    # Apply LDA model
    lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
    lda_model.fit(dtm)

    # Print the topics
    topics = []
    for idx, topic in enumerate(lda_model.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]  # Get top 10 words for each topic
        topics.append(topic_words)
        print(f"Topic {idx}: {topic_words}")

    return topics

# Sample text for demonstration
text = """
The United Nations is an international organization founded in 1945. 
Its main goals are to maintain international peace and security, 
promote human rights, foster social and economic development, 
protect the environment, and provide humanitarian aid in cases of famine, 
natural disaster, and armed conflict.
"""

# Preprocess text and extract entities
entities = preprocess_text(text)
print("Entities:")
for entity in entities:
    print(entity)

# Perform topic modeling
texts = [sent.text for sent in nlp(text).sents]  # Split text into sentences
print("\nTopics:")
topics = perform_topic_modeling(texts)

# Save topics to CSV file
df = pd.DataFrame(topics, columns=[f"Word {i}" for i in range(1, 11)])
df.to_csv("topics.csv", index=False)

# Save topics to text file
with open("topics.txt", "w") as f:
    for idx, topic in enumerate(topics):
        f.write(f"Topic {idx}:\n")
        f.write(", ".join(topic))
        f.write("\n\n")

print("Topics saved to 'topics.csv' and 'topics.txt' files.")
