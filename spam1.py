import pandas as pd
import numpy as np

# Step 1: Load and inspect the dataset safely
try:
    # Attempt to load the file with different delimiters to identify issues
    df = pd.read_csv('SMSSpamCollection2.csv', delimiter='\t', encoding='latin-1', on_bad_lines='skip')
    print("File loaded successfully.")
except Exception as e:
    print(f"Error loading the file: {e}")
    raise

# Step 2: Check for inconsistent rows
print("\nInspecting the dataset for issues...")
for index, row in df.iterrows():
    if len(row) != 2:
        print(f"Inconsistent row at index {index}: {row}")

# Step 3: Assign column names if missing
if df.columns[0] != 'label' or df.columns[1] != 'text':
    print("Assigning default column names: 'label' and 'text'.")
    df.columns = ['label', 'text']

# Step 4: Preprocess the data
def preprocess_text(text):
    if not isinstance(text, str):
        return ""  # Handle non-string entries
    # Lowercase conversion
    text = text.lower()
    # Remove special characters
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text

df['text'] = df['text'].apply(preprocess_text)

# Step 5: Drop rows with missing or invalid values
df.dropna(inplace=True)
print(f"Dataset cleaned. Remaining rows: {len(df)}")

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 6: Save the cleaned dataset (optional)
cleaned_file_path = 'Cleaned_SMSSpamCollection.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to {cleaned_file_path}.")

# Continue with feature extraction and model training (previous code)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test with custom input
while True:
    user_input = input("Enter a message to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    processed_input = preprocess_text(user_input)
    input_vec = vectorizer.transform([processed_input])
    prediction = model.predict(input_vec)
    print("Spam" if prediction[0] == 1 else "Ham")