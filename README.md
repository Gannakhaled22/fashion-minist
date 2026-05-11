# fashion-minist
# Fashion MNIST Classification 

## Project Overview

This project implements machine learning models to automatically classify fashion items from the Fashion MNIST dataset. The goal is to build and compare three different algorithms to predict clothing categories from images.

## 🎯 Project Objective

Train and evaluate three machine learning algorithms to classify 28x28 pixel grayscale images of clothing items into 10 different categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## 📊 Project Steps

### 1. **Data Loading & Exploration**
   - Load training (60,000 samples) and test (10,000 samples) datasets
   - Analyze data shape and check for missing values
   - Display sample images

### 2. **Data Preprocessing**
   - Normalize pixel values from [0-255] to [0-1] range
   - Split features and labels
   - Prepare data for model training

### 3. **Dimensionality Reduction (PCA)**
   - Apply Principal Component Analysis to reduce from 784 features to ~100 components
   - Retain 95% of variance while reducing computational complexity
   - Achieve ~87% compression ratio

### 4. **Model Training**
   Three algorithms are implemented and trained:
   
   **Logistic Regression**
   - Simple, fast, interpretable baseline model
   
   **Support Vector Machine (SVM)**
   - Powerful non-linear classifier with RBF kernel
   - Best performing model in this project
   
   **Random Forest**
   - Ensemble method using 100 decision trees
   - Provides feature importance analysis

### 5. **Model Evaluation**
   - Compare accuracies across all models
   - Generate detailed classification reports
   - Visualize confusion matrices
   - Analyze feature importance

## 📈 Key Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~87% |
| **SVM** | **~92%** ✅ |
| Random Forest | ~88% |

## 🔍 Key Insights

1. **PCA Effectiveness**: Successfully reduced dimensionality while preserving 95% of variance information
2. **Best Model**: SVM achieved the highest accuracy (~92%) on the Fashion MNIST dataset
3. **Model Performance**: All three models showed strong performance (>85% accuracy)
4. **Computational Efficiency**: PCA reduced training time significantly without sacrificing accuracy

## 📁 Project Structure

```
fashion-minist/
├── README.md                          # This file
├── fashion_mnist_classification.ipynb  # Main notebook with full implementation
├── fashion-mnist_train.csv            # Training data (60,000 samples)
└── fashion-mnist_test.csv             # Test data (10,000 samples)
```

## 🛠️ Technologies Used

- **Python 3.10+**
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization

## 💡 Features

- ✅ Complete data preprocessing pipeline
- ✅ Dimensionality reduction with PCA
- ✅ Multiple model implementations and comparison
- ✅ Comprehensive evaluation metrics
- ✅ Visualization of results (confusion matrices, feature importance)
- ✅ Detailed classification reports

## 🚀 How to Run

1. Ensure you have Python 3.10+ installed
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Place the CSV files in your working directory
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook fashion_mnist_classification.ipynb
   ```

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🎓 Learning Outcomes

This project demonstrates:
- Data preprocessing and normalization techniques
- Dimensionality reduction methods (PCA)
- Implementation of multiple classification algorithms
- Model evaluation and comparison strategies
- Data visualization for model interpretation
- Feature importance analysis

## 💬 Recommendations for Improvement

1. **Hyperparameter Tuning**: Use GridSearchCV/RandomizedSearchCV for optimal parameters
2. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
3. **Deep Learning**: Try CNN models (like LeNet or ResNet) for potentially better results
4. **Data Augmentation**: Apply transformations (rotation, shift) to expand training data
5. **Class Balancing**: Check and handle any class imbalance if present

## 📞 Contact & Support

For questions or suggestions about this project, feel free to reach out!

---

**Author**: Gannakhaled22  
**Date**: May 2026  
**License**: MIT

