# Fake_news_detection
# News Classification Project

## Overview

This project involves the analysis and classification of news articles using machine learning techniques. We will preprocess the data, visualize it, and train models to classify news as either fake or real.

## Libraries Used

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
```

## Dataset

- The dataset `News.csv` contains 44,919 entries and 5 columns.
- Relevant columns for classification include `text` and `class`.

## Data Preprocessing Steps

1. **Load the Data:**
   ```python
   data = pd.read_csv('News.csv', index_col=0)
   ```

2. **Inspect the Data:**
   ```python
   data.head()
   ```

3. **Drop Irrelevant Columns:**
   ```python
   data = data.drop(["title", "subject", "date"], axis=1)
   ```

4. **Check for Null Values:**
   ```python
   data.isnull().sum()
   ```

5. **Shuffle the Dataset:**
   ```python
   data = data.sample(frac=1)
   data.reset_index(inplace=True)
   data.drop(["index"], axis=1, inplace=True)
   ```

6. **Visualize Class Distribution:**
   ```python
   sns.countplot(data=data, x='class', order=data['class'].value_counts().index)
   ```

## Text Preprocessing

- **Download Required NLTK Resources:**
   ```python
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

- **Define Text Preprocessing Function:**
   ```python
   def preprocess_text(text_data):
       preprocessed_text = []
       for sentence in tqdm(text_data):
           sentence = re.sub(r'[^\w\s]', '', sentence)
           preprocessed_text.append(' '.join(token.lower() for token in str(sentence).split() if token not in stopwords.words('english')))
       return preprocessed_text
   ```

- **Apply Preprocessing:**
   ```python
   preprocessed_review = preprocess_text(data['text'].values)
   data['text'] = preprocessed_review
   ```

## Visualization

1. **WordCloud for Real News:**
   ```python
   consolidated = ' '.join(word for word in data['text'][data['class'] == 1].astype(str))
   wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110, collocations=False)
   plt.figure(figsize=(15, 10))
   plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
   plt.axis('off')
   plt.show()
   ```

2. **WordCloud for Fake News:**
   ```python
   consolidated = ' '.join(word for word in data['text'][data['class'] == 0].astype(str))
   wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110, collocations=False)
   plt.figure(figsize=(15, 10))
   plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
   plt.axis('off')
   plt.show()
   ```

3. **Top 20 Most Frequent Words Bar Chart:**
   ```python
   def get_top_n_words(corpus, n=None):
       vec = CountVectorizer().fit(corpus)
       bag_of_words = vec.transform(corpus)
       sum_words = bag_of_words.sum(axis=0)
       words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
       words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
       return words_freq[:n]

   common_words = get_top_n_words(data['text'], 20)
   df1 = pd.DataFrame(common_words, columns=['Review', 'count'])
   df1.groupby('Review').sum()['count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel="Top Words", ylabel="Count", title="Bar Chart of Top Words Frequency")
   ```

## Model Training and Evaluation

1. **Split the Data:**
   ```python
   x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25)
   ```

2. **Vectorization Using TfidfVectorizer:**
   ```python
   vectorization = TfidfVectorizer()
   x_train = vectorization.fit_transform(x_train)
   x_test = vectorization.transform(x_test)
   ```

3. **Logistic Regression Model:**
   ```python
   model = LogisticRegression()
   model.fit(x_train, y_train)
   print(accuracy_score(y_train, model.predict(x_train)))
   print(accuracy_score(y_test, model.predict(x_test)))
   ```

4. **Decision Tree Classifier:**
   ```python
   model = DecisionTreeClassifier()
   model.fit(x_train, y_train)
   print(accuracy_score(y_train, model.predict(x_train)))
   print(accuracy_score(y_test, model.predict(x_test)))
   ```

5. **Confusion Matrix:**
   ```python
   cm = metrics.confusion_matrix(y_test, model.predict(x_test))
   cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
   cm_display.plot()
   plt.show()
   ```

## Conclusion

This project demonstrates the process of preprocessing news data, visualizing text data, and building a classification model to differentiate between real and fake news. The models can achieve high accuracy, showcasing the effectiveness of the chosen methods.
