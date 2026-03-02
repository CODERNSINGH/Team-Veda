# Question Difficulty Classification System - Team Veda
- **Narendra Singh** (2401010292): Sec A

## Project Overview

This project implements a **Machine Learning-based Question Difficulty Classifier** that automatically categorizes educational questions into difficulty levels (Easy, Medium, Hard) based on their content. The system uses Natural Language Processing (NLP) and various machine learning approaches to achieve accurate classification.

### Objective

To build a robust classifier that can:
- Analyze question text and metadata
- Predict difficulty levels automatically
- Help educators create balanced question papers
- Assist in adaptive learning systems

---

## Dataset Information

**Source:** Question Paper Dataset (10,000 entries)

**Features:**
- `QUESTION_TEXT`: The actual question content
- `TOPIC`: Subject/topic of the question
- `MARKS`: Points assigned to the question
- `ANSWER_TEXT`: Expected answer
- `DIFFICULTY_LABEL`: Target variable (Easy/Medium/Hard)

**Initial Observations:**
- Dataset contains 10,000 questions across multiple topics
- No missing values after cleaning
- Imbalanced class distribution across difficulty levels
- Strong correlation between marks and difficulty (potential data leakage)

---

## Iterative Approach: 4 Methods Explored

### **Method 1: Baseline TF-IDF + Logistic Regression**

#### Approach
```python
- Preprocessing: Lowercasing, punctuation removal
- Vectorization: TF-IDF with max_features=3000
- Model: Logistic Regression (max_iter=1000)
- Split: 80-20 train-test split
```

#### Results
- **Accuracy: ~95-98%** (Suspiciously high!)

#### Issues Identified
1. **Data Leakage Problem**: The model was inadvertently learning from the `MARKS` column, which directly correlates with difficulty:
   - 1 mark → Easy
   - 2-3 marks → Medium
   - 4-5 marks → Hard

2. **Overfitting**: The model was memorizing patterns rather than learning generalizable features

3. **No Stratification**: Test set didn't represent the true class distribution

#### Key Insight
> High accuracy doesn't always mean good model! We discovered the model was "cheating" by learning the marks-to-difficulty mapping instead of understanding question complexity.

---

### **Method 2: Removing Data Leakage + Regularization**

#### Approach
```python
- Removed MARKS feature from training
- Applied stratified train-test split (stratify=y)
- Stronger regularization: C=0.5
- Reduced iterations: max_iter=500
- Added 5-fold cross-validation
```

#### Implementation
```python
model = LogisticRegression(
    max_iter=500,
    C=0.5  # Stronger regularization to prevent overfitting
)

# Cross-validation for robust evaluation
scores = cross_val_score(model_cv, X, y, cv=5)
```

#### Results
- **Cross-validation Accuracy: ~60-65%**
- **Test Accuracy: ~62%**

#### ⚠️ Issues Identified
1. **Significant accuracy drop** after removing data leakage
2. **Limited feature representation** - TF-IDF alone wasn't capturing question complexity
3. **Poor performance on minority classes** (Medium difficulty)

#### Key Insight
> Removing data leakage revealed the true challenge: predicting difficulty from question text alone is genuinely difficult and requires more sophisticated features.

---

### **Method 3: Enhanced Feature Engineering + LinearSVC**

#### Approach
```python
- Added n-grams: ngram_range=(1,2) for bigrams
- Increased vocabulary: max_features=5000→7000
- Class balancing: class_weight="balanced"
- Alternative algorithm: LinearSVC instead of Logistic Regression
```

#### Implementation
```python
# Enhanced TF-IDF with bigrams
vectorizer = TfidfVectorizer(
    max_features=7000,
    stop_words="english",
    ngram_range=(1,2)  # Captures phrases like "binary search"
)

# LinearSVC for better text classification
model = LinearSVC(class_weight="balanced")
```

#### Results
- **Test Accuracy: ~58-60%**

#### Issues Identified
1. **Accuracy decreased** instead of improving
2. **LinearSVC was more sensitive** to limited training data
3. **N-grams added noise** without sufficient data volume
4. **Root cause**: Only 10,000 samples insufficient for complex patterns

#### Key Insight
> More features ≠ Better performance. With limited data (10K samples), adding complexity led to worse generalization. We needed ~30K samples for meaningful improvement.

---

### **Method 4: Multi-Feature Fusion (FINAL SOLUTION)**

#### Approach
```python
- Combined multiple text features: QUESTION_TEXT + TOPIC
- Enhanced text preprocessing
- Optimized TF-IDF parameters
- Balanced class weights
- Comprehensive evaluation metrics
```

#### Implementation

**1. Enhanced Text Cleaning**
```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

# Apply to all text columns
df["QUESTION_TEXT"] = df["QUESTION_TEXT"].apply(clean_text)
df["TOPIC"] = df["TOPIC"].apply(clean_text)
```

**2. Feature Combination**
```python
# Combine question text with topic for richer context
df["COMBINED_TEXT"] = df["QUESTION_TEXT"] + " " + df["TOPIC"]
```

**3. Optimized TF-IDF**
```python
vectorizer = TfidfVectorizer(
    max_features=7000,      # Adequate vocabulary size
    stop_words="english",    # Remove common words
    ngram_range=(1,2),       # Unigrams + bigrams
    min_df=2                 # Ignore very rare terms
)

# Critical: Fit only on training data to prevent leakage
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)  # Transform only
```

**4. Balanced Classification**
```python
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"  # Handle class imbalance
)
```

#### 📈 Final Results

**Overall Metrics:**
```
Accuracy: ~68-72%
Precision (macro): ~0.70
Recall (macro): ~0.68
F1-Score (macro): ~0.69
```

**Per-Class Performance:**
```
Easy Questions:
  - Precision: 0.75
  - Recall: 0.78
  - F1-Score: 0.76

Medium Questions:
  - Precision: 0.65
  - Recall: 0.62
  - F1-Score: 0.63

Hard Questions:
  - Precision: 0.71
  - Recall: 0.74
  - F1-Score: 0.72
```

#### Why This Method Worked

1. **Multi-Feature Fusion**: Combining question text with topic provided crucial context
   - "What is a primary key?" + "Database" → More informative than text alone

2. **Proper Data Splitting**: Preventing information leakage
   - Vectorizer fit only on training data
   - Stratified split maintained class distribution

3. **Class Balancing**: Addressed imbalanced dataset
   - Prevented model bias toward majority class

4. **Realistic Expectations**: ~70% accuracy is reasonable given:
   - Limited dataset size (10K samples)
   - Subjective nature of difficulty assessment
   - Need for domain expertise

---

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.7+
pip package manager
```

### Required Libraries
```bash
pip install scikit-learn matplotlib seaborn pandas numpy
```

### Dataset Setup
```python
# Update the path to your dataset
DATA_PATH = "path/to/question_paper_10000.csv"
```

---

## Usage Guide

### 1. Training the Model

```python
# Load and preprocess data
df = pd.read_csv(DATA_PATH, engine="python", quotechar='"', on_bad_lines="skip")
df = df.dropna()

# Clean text
df["QUESTION_TEXT"] = df["QUESTION_TEXT"].apply(clean_text)
df["TOPIC"] = df["TOPIC"].apply(clean_text)

# Combine features
df["COMBINED_TEXT"] = df["QUESTION_TEXT"] + " " + df["TOPIC"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['COMBINED_TEXT'], 
    df["DIFFICULTY_LABEL"],
    test_size=0.2,
    random_state=42,
    stratify=df["DIFFICULTY_LABEL"]
)

# Vectorize
vectorizer = TfidfVectorizer(max_features=7000, stop_words="english", 
                             ngram_range=(1,2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)
```

### 2. Making Predictions

```python
def predict_difficulty(question_text, topic=""):
    """
    Predict difficulty level for a new question
    
    Args:
        question_text (str): The question to classify
        topic (str): Optional topic information
    
    Returns:
        str: Predicted difficulty level (Easy/Medium/Hard)
    """
    combined = clean_text(question_text) + " " + clean_text(topic)
    vectorized = vectorizer.transform([combined])
    prediction = model.predict(vectorized)
    return prediction[0]

# Example usage
question = "What is a primary key in a database?"
difficulty = predict_difficulty(question, topic="Database")
print(f"Predicted Difficulty: {difficulty}")
```

### 3. Batch Prediction

```python
# Predict for multiple questions
questions = [
    "What is a primary key in a database?",
    "Write a Python program to implement merge sort algorithm",
    "Analyze the time complexity of binary search with mathematical proof"
]

for q in questions:
    print(f"Q: {q}")
    print(f"Difficulty: {predict_difficulty(q)}\n")
```

### 4. Model Persistence

```python
import pickle

# Save trained model
with open("difficulty_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Load for future use
with open("difficulty_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    loaded_vectorizer = pickle.load(f)
```

---

## 📊 Evaluation & Visualization

### Confusion Matrix Analysis

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Easy', 'Medium', 'Hard'],
            yticklabels=['Easy', 'Medium', 'Hard'])
plt.title("Confusion Matrix - Question Difficulty Classification")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
```

### Feature Importance

```python
# Top discriminative features
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_

for i, class_name in enumerate(['Easy', 'Medium', 'Hard']):
    top_indices = np.argsort(coefficients[i])[-10:]
    top_features = feature_names[top_indices]
    print(f"\nTop features for {class_name}:")
    print(top_features)
```

---

## Example Predictions

| Question | Actual | Predicted | Correct? |
|----------|--------|-----------|----------|
| What is a primary key in a database? | Easy | Easy | Y |
| Define a list in Python | Easy | Easy | Y |
| Write a program to check palindrome | Medium | Medium | N |
| Explain normalization in DBMS | Medium | Hard | Y |
| Design a file handling system in Python | Hard | Hard | Y |
| Analyze time complexity of Binary Search | Hard | Hard | Y |

---

## 🔍 Key Learnings & Best Practices

### 1. **Data Leakage Prevention**
- Always check for features that directly correlate with target
- In our case, `MARKS` was a proxy for `DIFFICULTY_LABEL`
- Remove or engineer such features carefully

### 2. **Train-Test Contamination**
- **NEVER** fit vectorizers/scalers on entire dataset
- Always fit on training data only, transform test data
- Use stratified splits for imbalanced datasets

### 3. **Feature Engineering Matters**
- Combining related features (question + topic) improved performance by ~8-10%
- Domain knowledge is crucial for effective feature engineering

### 4. **Evaluation Beyond Accuracy**
- Precision, Recall, F1-Score reveal per-class performance
- Confusion matrix shows specific misclassification patterns
- Cross-validation provides robust performance estimates

### 5. **Realistic Expectations**
- 70% accuracy is good for subjective classification tasks
- Limited data (10K samples) constrains model capability
- Difficulty assessment is inherently subjective

---

## Limitations & Future Improvements

### Current Limitations

1. **Dataset Size**: 10,000 samples is relatively small for NLP tasks
   - Ideally need 30,000+ samples for 75-80% accuracy

2. **Subjectivity**: Difficulty is subjective and context-dependent
   - What's "easy" for CS students might be "hard" for beginners

3. **Feature Representation**: TF-IDF has limitations
   - Doesn't capture semantic meaning
   - "What is inheritance?" vs "Explain inheritance" treated differently

4. **Class Imbalance**: Uneven distribution affects minority class performance

### Future Enhancements

#### 1. **Deep Learning Approaches**
```python
# BERT-based classification
from transformers import BertTokenizer, BertForSequenceClassification

# Fine-tune BERT on question difficulty
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=3
)
```

**Expected Improvement**: +10-15% accuracy

#### 2. **Additional Features**
- Question length (word count)
- Presence of keywords: "analyze", "design", "implement"
- Answer length as complexity indicator
- Bloom's Taxonomy keyword detection

```python
# Feature engineering example
df["question_length"] = df["QUESTION_TEXT"].str.split().str.len()
df["has_code_keyword"] = df["QUESTION_TEXT"].str.contains(
    "implement|program|algorithm"
).astype(int)
```

#### 3. **Ensemble Methods**
```python
from sklearn.ensemble import VotingClassifier

# Combine multiple models
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('svc', LinearSVC()),
        ('rf', RandomForestClassifier())
    ],
    voting='soft'
)
```

**Expected Improvement**: +5-7% accuracy

#### 4. **Active Learning**
- Start with current model
- Identify uncertain predictions
- Get expert labels for those
- Retrain iteratively

#### 5. **Data Augmentation**
- Paraphrase existing questions
- Use GPT models to generate synthetic questions
- Expand dataset to 30,000+ samples

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.x | Core implementation |
| NLP | TF-IDF Vectorization | Text feature extraction |
| ML Model | Logistic Regression | Classification |
| Preprocessing | scikit-learn | Data preparation |
| Visualization | Matplotlib, Seaborn | EDA and results |
| Data Handling | Pandas, NumPy | Data manipulation |

---

## 📖 References & Resources

### Academic Papers
1. **TF-IDF**: Salton & McGill (1983) - "Introduction to Modern Information Retrieval"
2. **Text Classification**: Joachims (1998) - "Text Categorization with Support Vector Machines"

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TF-IDF Explanation](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [Logistic Regression Guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

### Related Work
- Automated Question Classification in Education
- Bloom's Taxonomy for Question Difficulty
- Natural Language Processing for Educational Assessment
