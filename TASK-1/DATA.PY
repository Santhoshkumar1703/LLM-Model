import nltk
from nltk.tokenize import word_tokenize

# Read in the chat conversation data
with open('WhatsApp Chat with IIIT NR 2020 Academics.txt', encoding='utf-8') as f:
    chat_data = f.read()

# Tokenize the chat data
tokens = word_tokenize(chat_data)

# Filter out non-alphabetic tokens and convert to lowercase
clean_tokens = [token.lower() for token in tokens if token.isalpha()]

# Remove stop words
stop_words = set(nltk.corpus.stopwords.words('english'))
filtered_tokens = [token for token in clean_tokens if token not in stop_words]

