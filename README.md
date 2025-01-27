# Spam Detection Model

This repository contains a Python script that implements a spam detection model. The model is trained using a dataset of SMS messages labeled as "spam" or "ham" (not spam). The script preprocesses the data, trains a machine learning model, and provides an interactive interface to classify new messages.

## Features
- Preprocesses text data: lowercase conversion, removal of special characters, and handling of missing values.
- Utilizes a Naive Bayes classifier for spam detection.
- Implements TF-IDF vectorization for feature extraction.
- Allows users to test the model with their own messages via an interactive prompt.

---

## How to Use

### 1. **Setup**
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. **Prepare the Dataset**
This script requires a dataset in CSV format containing two columns:
- `label`: Specifies whether the message is "spam" or "ham".
- `text`: The content of the SMS message.

Ensure the dataset meets the following conditions:
- It should have exactly two columns with the above structure.
- Any inconsistencies in rows (e.g., extra columns) should be corrected.

#### Example Format:
| label | text                                   |
|-------|---------------------------------------|
| ham   | Hello, how are you?                   |
| spam  | Congratulations! You've won $1,000!  |

If you do not have a dataset, you can download one from sources like [Kaggle](https://www.kaggle.com/) or create your own.

Save your dataset as `SMSSpamCollection2.csv` and place it in the project directory.

### 3. **Run the Script**
Run the script using the following command:
```bash
python spam1.py
```

### 4. **Interactive Message Classification**
Once the model is trained, you can enter messages to classify them as "Spam" or "Ham".
To exit the interactive mode, type:
```bash
exit
```

---

## Script Workflow
1. **Load the Dataset**:
   - Reads the CSV file (`SMSSpamCollection2.csv`) using `pandas`.
   - Handles errors in data formatting.

2. **Data Cleaning and Preprocessing**:
   - Handles missing or inconsistent rows.
   - Converts text to lowercase and removes special characters.
   - Encodes the `label` column as `0` for ham and `1` for spam.

3. **Model Training**:
   - Splits the dataset into training and testing subsets.
   - Uses TF-IDF vectorization to transform text data into numerical features.
   - Trains a Multinomial Naive Bayes classifier.

4. **Evaluation**:
   - Prints the model’s accuracy, classification report, and confusion matrix.

5. **Interactive Classification**:
   - Allows users to input custom messages for spam detection.

---

## Files
- `spam1.py`: The main Python script for spam detection.
- `SMSSpamCollection2.csv`: The dataset file (to be added by the user).
- `Cleaned_SMSSpamCollection.csv`: The cleaned dataset (generated during script execution).

---

## Notes
- Ensure that your dataset is properly formatted to avoid errors.
- If using a custom dataset, update the filename in the script if necessary.
- Modify the preprocessing function in the script to handle additional requirements (e.g., stopword removal or stemming).

---

## Requirements
- Python 3.7 or higher
- pandas
- numpy
- scikit-learn

Install the dependencies using:
```bash
pip install pandas numpy scikit-learn
```

---

## Future Enhancements
- Implement additional preprocessing techniques (e.g., lemmatization).
- Allow the use of other classification models (e.g., SVM or deep learning models).
- Add a graphical user interface for easier interaction.

---

## License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

If you encounter any issues or have suggestions, please feel free to open an issue or submit a pull request!

