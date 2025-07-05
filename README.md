# Iris Flower Classification Project

This is a simple machine learning project that uses the classic Iris dataset to classify flower species based on sepal and petal measurements using the K-Nearest Neighbors (KNN) algorithm.


## 🛠️ Tools and Libraries Used

- Python 3.11
- Pandas – for data manipulation
- NumPy – for numerical operations
- Scikit-learn – for model training and evaluation
- Matplotlib and Seaborn – for visualization

## 📊 Dataset

The dataset used in this project is the well-known Iris flower dataset, which contains 150 samples of Iris flowers, each with four features: sepal length, sepal width, petal length, and petal width. The target variable includes three species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

Source: [Iris Dataset (Kaggle)](https://www.kaggle.com/arshid/iris-flower-dataset )

## 🧪 Model Overview

A K-Nearest Neighbors classifier was trained after standardizing the input features. The model achieved an accuracy of approximately **97.78%** on the test set.

### Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (precision, recall, F1-score per class)

## 📈 Visualizations

Two key visualizations were generated:
- **Pairplot**: Shows feature relationships across different species.
- **Confusion Matrix Heatmap**: Visual representation of model performance.

## 🚀 Future Improvements

Potential enhancements include:
- Trying other classification algorithms (e.g., Random Forest, SVM)
- Performing hyperparameter tuning with GridSearchCV
- Implementing cross-validation
- Building a simple GUI or web app using Streamlit or Tkinter

MIT License

Copyright (c) 2025 Vaishnav Anand

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
