# Sample text
text = "Hello! Welcome to Educative. Happy learning."

# Tokenize the text into words
words = text.split()

# Create a set to get unique words (vocabulary)
vocabulary = set(words)

# Generate one-hot encoded vectors for each word in the vocabulary
one_hot_encoded = []
for word in vocabulary:
    # Create a list of zeros with the length of the vocabulary
    encoding = [0] * len(vocabulary)
    
    # Get the index of the word in the vocabulary
    index = list(vocabulary).index(word)
    
    # Set the value at the index to 1 to indicate word presence
    encoding[index] = 1
    one_hot_encoded.append((word, encoding))

# Print the one-hot encoded vectors
for word, encoding in one_hot_encoded:
    print(f"{word}: {encoding}")

print(vocabulary)