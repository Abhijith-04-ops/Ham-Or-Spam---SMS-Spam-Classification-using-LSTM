# SMS Classification using LSTM

This repository contains a Jupyter Notebook demonstrating how to classify SMS messages (spam vs ham) using an LSTM model built with TensorFlow/Keras.

## Steps Covered

- Data Loading and Preprocessing
- Text Cleaning and Tokenization
- Padding Sequences for Uniform Length
- Building the LSTM Model
- Model Compilation and Training
- Model Evaluation and Accuracy Calculation
- Visualization of Training Results
- Predicting New SMS Samples

## Notebook Summary
```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
df = pd.read_csv('SMSSpamCollection.csv')
df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>



**Label Encoding for result**


```python
df['result'] = df['result'].map({'ham': 0, 'spam': 1})
```

**Tokenizing the values**


```python
labels = df['result'].values
messages = df['message'].values
```
```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(messages)
sequences = tokenizer.texts_to_sequences(messages)
```

```python
# Lowercase and simple tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(messages)
sequences = tokenizer.texts_to_sequences(messages)
```

**Padding to make the inputs of same length**


```python
max_len = 100
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
```

**Splitting into training and testing data**


```python
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

```

**Model Compiling with Three Layers**


```python
vocab_size = len(tokenizer.word_index) + 1

model_baseline = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model_baseline.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

    C:\Users\abhij\AppData\Roaming\Python\Python312\site-packages\keras\src\layers\core\embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.
      warnings.warn(


**Model Training upto 5 epochs**


```python
print("\nTraining Baseline Model...")
history_baseline = model_baseline.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
```

    
    Training Baseline Model...
    Epoch 1/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8654 - loss: 0.3995 - val_accuracy: 0.8556 - val_loss: 0.4156
    Epoch 2/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 28ms/step - accuracy: 0.8686 - loss: 0.3907 - val_accuracy: 0.8556 - val_loss: 0.4142
    Epoch 3/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 29ms/step - accuracy: 0.8686 - loss: 0.3902 - val_accuracy: 0.8556 - val_loss: 0.4212
    Epoch 4/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 29ms/step - accuracy: 0.8686 - loss: 0.3909 - val_accuracy: 0.8556 - val_loss: 0.4135
    Epoch 5/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 28ms/step - accuracy: 0.8686 - loss: 0.3909 - val_accuracy: 0.8556 - val_loss: 0.4154


**Baseline Model Evaluation**


```python
print("\nEvaluating Baseline Model...")
y_pred_baseline = (model_baseline.predict(X_test) > 0.5).astype("int32")
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Model Accuracy: {accuracy_baseline:.4f}")
```

    
    Evaluating Baseline Model...
    [1m35/35[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 15ms/step
    Baseline Model Accuracy: 0.8556



```python
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Baseline Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

```

**Adding Dropout to the baseline model**


```python
print("\nBuilding and Training Mini-Experiment Model (with Dropout)...")
model_experiment = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model_experiment.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_experiment = model_experiment.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
```

    
    Building and Training Mini-Experiment Model (with Dropout)...
    Epoch 1/5


    C:\Users\abhij\AppData\Roaming\Python\Python312\site-packages\keras\src\layers\core\embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.
      warnings.warn(


    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 31ms/step - accuracy: 0.8648 - loss: 0.4087 - val_accuracy: 0.8556 - val_loss: 0.4132
    Epoch 2/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 29ms/step - accuracy: 0.8686 - loss: 0.3960 - val_accuracy: 0.8556 - val_loss: 0.4153
    Epoch 3/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 28ms/step - accuracy: 0.8686 - loss: 0.3951 - val_accuracy: 0.8556 - val_loss: 0.4133
    Epoch 4/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 28ms/step - accuracy: 0.8686 - loss: 0.3927 - val_accuracy: 0.8556 - val_loss: 0.4135
    Epoch 5/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 30ms/step - accuracy: 0.8686 - loss: 0.3937 - val_accuracy: 0.8556 - val_loss: 0.4130


**Comapritive Evaluation of the Experimental and Baseline Model**


```python

print("\nEvaluating Mini-Experiment Model...")
y_pred_experiment = (model_experiment.predict(X_test) > 0.5).astype("int32")
accuracy_experiment = accuracy_score(y_test, y_pred_experiment)
print(f"Mini-Experiment Model Accuracy: {accuracy_experiment:.4f}")

print("\n--- Final Comparison ---")
print(f"Baseline Model Test Accuracy: {accuracy_baseline:.4f}")
print(f"Mini-Experiment Model Test Accuracy: {accuracy_experiment:.4f}")
```

    
    Evaluating Mini-Experiment Model...
    [1m35/35[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 15ms/step
    Mini-Experiment Model Accuracy: 0.8556
    
    --- Final Comparison ---
    Baseline Model Test Accuracy: 0.8556
    Mini-Experiment Model Test Accuracy: 0.8556


**Add On (Setting LSTM to 64 -> 128) which is optional**


```python
print("Building and Training Mini-Experiment model(with Drop-out)")
model_experiment1 = Sequential([Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len), LSTM(128),Dense(1, activation='sigmoid')])
model_experiment1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_experiment = model_experiment1.fit( X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)
```

    Building and Training Mini-Experiment model(with Drop-out)
    Epoch 1/5


    C:\Users\abhij\AppData\Roaming\Python\Python312\site-packages\keras\src\layers\core\embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.
      warnings.warn(


    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 50ms/step - accuracy: 0.8686 - loss: 0.4006 - val_accuracy: 0.8556 - val_loss: 0.4169
    Epoch 2/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 44ms/step - accuracy: 0.8686 - loss: 0.3912 - val_accuracy: 0.8556 - val_loss: 0.4163
    Epoch 3/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 48ms/step - accuracy: 0.8686 - loss: 0.3904 - val_accuracy: 0.8556 - val_loss: 0.4142
    Epoch 4/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 46ms/step - accuracy: 0.8686 - loss: 0.3910 - val_accuracy: 0.8556 - val_loss: 0.4130
    Epoch 5/5
    [1m140/140[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 46ms/step - accuracy: 0.8686 - loss: 0.3899 - val_accuracy: 0.8556 - val_loss: 0.4139


**Evaluation of Add-On**


```python
print("\nEvaluating Mini-Experiment Model...")
y_pred_experiment = (model_experiment1.predict(X_test) > 0.5).astype("int32")
accuracy_experiment1 = accuracy_score(y_test, y_pred_experiment)
print(f"Add On Model Accuracy: {accuracy_experiment1:.4f}")
print("\n--- Final Comparison ---")
print(f"Baseline Model Test Accuracy: {accuracy_baseline:.4f}")
print(f"Mini-Experiment Model Test Accuracy: {accuracy_experiment:.4f}")
print(f"Add On Model Test Accuracy: {accuracy_experiment1:.4f}")
```

    
    Evaluating Mini-Experiment Model...
    [1m35/35[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step
    Add On Model Accuracy: 0.8556
    
    --- Final Comparison ---
    Baseline Model Test Accuracy: 0.8556
    Mini-Experiment Model Test Accuracy: 0.8556
    Add On Model Test Accuracy: 0.8556



## Requirements
```
python>=3.8
tensorflow>=2.0
pandas
numpy
matplotlib
scikit-learn
```
