# Machine Learning Algorithm Selection Cheat Sheet

## Quick Decision Tree

### 1. What Type of Problem?

#### **Supervised Learning** (You have labeled data)
- **Regression** → Predicting continuous values (prices, temperatures, etc.)
- **Classification** → Predicting categories (spam/not spam, cat/dog, etc.)

#### **Unsupervised Learning** (No labels)
- **Clustering** → Grouping similar data points
- **Dimensionality Reduction** → Reducing features while keeping information
- **Anomaly Detection** → Finding outliers

#### **Reinforcement Learning** (Agent learning through rewards)
- Game AI, robotics, optimization problems

---

## Supervised Learning: Regression

### When to Use Each Algorithm

| Algorithm | Best For | Pros | Cons | Data Size |
|-----------|----------|------|------|-----------|
| **Linear Regression** | Simple relationships, interpretability needed | Fast, interpretable, low variance | Assumes linearity | Small-Large |
| **Ridge/Lasso Regression** | Many features, need regularization | Handles multicollinearity, prevents overfitting | Still assumes linearity | Small-Large |
| **Decision Tree Regressor** | Non-linear relationships, mixed data types | No feature scaling needed, handles non-linearity | High variance, overfits easily | Small-Medium |
| **Random Forest Regressor** | Complex non-linear, robust predictions | Reduces overfitting, handles outliers well | Slower, less interpretable | Medium-Large |
| **Gradient Boosting (XGBoost/LightGBM)** | Competition-winning performance | Often best accuracy, handles missing data | Can overfit, requires tuning | Medium-Large |
| **Support Vector Regression (SVR)** | Small datasets with non-linear patterns | Effective in high dimensions | Slow on large datasets, hard to interpret | Small-Medium |
| **Neural Networks** | Very complex patterns, large datasets | Can model any function | Needs lots of data, black box, slow to train | Large-Huge |

**Quick Pick:**
- **< 1,000 samples** → Linear/Ridge/Lasso or SVR
- **1,000-100,000 samples** → Random Forest or XGBoost
- **> 100,000 samples** → Neural Networks or LightGBM

---

## Supervised Learning: Classification

### Binary Classification (2 classes)

| Algorithm | Best For | Pros | Cons | Data Size |
|-----------|----------|------|------|-----------|
| **Logistic Regression** | Baseline, interpretability | Fast, probabilistic outputs, interpretable | Linear decision boundary | Small-Large |
| **Naive Bayes** | Text classification, real-time prediction | Very fast, works with small data | Assumes feature independence | Small-Medium |
| **Decision Tree** | Quick interpretable model | Visual rules, no scaling needed | Overfits easily | Small-Medium |
| **Random Forest** | Robust general purpose | Handles non-linearity, reduces overfitting | Slower prediction | Medium-Large |
| **XGBoost/LightGBM** | When accuracy matters most | Best performance usually | Needs tuning, can overfit | Medium-Large |
| **SVM** | Clear margin between classes | Effective in high dimensions | Slow on large data | Small-Medium |
| **Neural Networks** | Complex patterns, images, text | Can learn any pattern | Needs lots of data, slow | Large-Huge |

### Multi-class Classification (3+ classes)

- **One-vs-Rest**: Logistic Regression, SVM
- **Native Multi-class**: Random Forest, XGBoost, Neural Networks, Naive Bayes
- **Softmax Regression**: When you need probabilities for each class

**Quick Pick:**
- **Tabular data, medium size** → XGBoost
- **Text data** → Naive Bayes or Neural Networks (BERT, etc.)
- **Image data** → Convolutional Neural Networks (CNNs)
- **Need interpretability** → Logistic Regression or Decision Tree

---

## Unsupervised Learning: Clustering

| Algorithm | Best For | Pros | Cons | Cluster Shape |
|-----------|----------|------|------|---------------|
| **K-Means** | Well-separated spherical clusters | Fast, simple, scalable | Need to specify K, sensitive to outliers | Spherical |
| **DBSCAN** | Arbitrary shapes, don't know K | Finds outliers, no K needed | Struggles with varying densities | Any shape |
| **Hierarchical Clustering** | Need cluster relationships/dendrogram | No K needed, shows hierarchy | Slow on large data | Any shape |
| **Gaussian Mixture Models (GMM)** | Overlapping clusters, soft assignment | Probabilistic, elliptical clusters | Slower than K-Means | Elliptical |
| **Mean Shift** | Don't know number of clusters | No K needed, robust to outliers | Very slow | Any shape |

**Quick Pick:**
- **Know number of clusters** → K-Means
- **Don't know K, have noise** → DBSCAN
- **Small dataset, want hierarchy** → Hierarchical Clustering

---

## Dimensionality Reduction

| Algorithm | Best For | Pros | Cons | Linear/Non-linear |
|-----------|----------|------|------|-------------------|
| **PCA** | Visualization, preprocessing | Fast, interpretable | Linear only | Linear |
| **t-SNE** | Visualization (2D/3D) | Great for visualizing clusters | Very slow, only for visualization | Non-linear |
| **UMAP** | Faster t-SNE alternative | Faster than t-SNE, preserves global structure | Newer, less tested | Non-linear |
| **Autoencoders** | Deep learning preprocessing | Can learn complex patterns | Needs lots of data | Non-linear |
| **LDA** | Classification with dimensionality reduction | Supervised, maximizes class separation | Linear, needs labels | Linear |

**Quick Pick:**
- **Preprocessing for ML** → PCA
- **Just visualizing data** → t-SNE or UMAP
- **Large datasets** → UMAP

---

## Special Cases

### Time Series Forecasting
- **ARIMA/SARIMA** → Traditional statistical approach
- **LSTM/GRU** → Complex patterns, long sequences
- **Prophet** → Business time series with seasonality
- **XGBoost with lag features** → Often surprisingly effective

### Anomaly Detection
- **Isolation Forest** → Fast, general purpose
- **One-Class SVM** → Small datasets
- **Autoencoders** → Complex patterns in high dimensions
- **DBSCAN** → Spatial outliers

### Imbalanced Classification
- **Use class weights** with any algorithm
- **SMOTE** → Oversample minority class
- **Random Forest/XGBoost** → Naturally handle imbalance better
- **Focus on precision/recall** not just accuracy

---

## Selection Flowchart

```
Start
  ↓
Do you have labeled data?
  ├─ YES → Supervised Learning
  │   ↓
  │   Predicting numbers or categories?
  │   ├─ Numbers → Regression
  │   │   ├─ < 1K samples → Linear/Ridge/Lasso
  │   │   ├─ 1K-100K samples → Random Forest or XGBoost
  │   │   └─ > 100K samples → Neural Networks or XGBoost
  │   │
  │   └─ Categories → Classification
  │       ├─ Text data → Naive Bayes or Transformers
  │       ├─ Image data → CNN
  │       ├─ Tabular data → XGBoost or Random Forest
  │       └─ Need interpretability → Logistic Regression
  │
  └─ NO → Unsupervised Learning
      ↓
      What's your goal?
      ├─ Group similar items → Clustering
      │   ├─ Know # of groups → K-Means
      │   └─ Don't know # of groups → DBSCAN
      │
      ├─ Reduce features → Dimensionality Reduction
      │   ├─ For visualization → t-SNE or UMAP
      │   └─ For preprocessing → PCA
      │
      └─ Find unusual items → Anomaly Detection
          └─ Isolation Forest
```

---

## Pro Tips

### 1. **Always Start Simple**
   - Begin with Logistic Regression or Linear Regression
   - Establishes a baseline
   - Shows if complex models are worth it

### 2. **Data Size Matters**
   - **< 1,000 samples**: Simple models (Linear, Logistic, Naive Bayes)
   - **1,000-100,000**: Tree-based (Random Forest, XGBoost)
   - **> 100,000**: Neural Networks become viable

### 3. **Feature Engineering > Algorithm Choice**
   - Good features with simple models often beat complex models with raw features

### 4. **Interpretability vs Performance Trade-off**
   - Most interpretable: Linear/Logistic Regression, Decision Trees
   - Least interpretable: Neural Networks, XGBoost
   - Middle ground: Random Forest (feature importance)

### 5. **Common Winning Combinations**
   - **Kaggle/Competitions**: XGBoost, LightGBM, Neural Networks, Ensemble
   - **Production/Business**: Logistic Regression, Random Forest (balanced performance/interpretability)
   - **Research**: Neural Networks, SVMs, Gaussian Processes

### 6. **Industry-Specific Defaults**
   - **Finance**: Logistic Regression, XGBoost (need interpretability)
   - **E-commerce**: Collaborative Filtering, XGBoost
   - **Healthcare**: Logistic Regression (interpretability required)
   - **Computer Vision**: CNNs (ResNet, EfficientNet)
   - **NLP**: Transformers (BERT, GPT)

---

## When to Ensemble

Combine multiple models when:
- Individual models have different strengths
- You need maximum accuracy (competitions)
- You have computational resources

**Common Ensemble Methods:**
- **Voting**: Majority vote (classification) or average (regression)
- **Stacking**: Train a meta-model on predictions
- **Boosting**: XGBoost, LightGBM (already ensemble methods)
- **Bagging**: Random Forest (already an ensemble)

---

## Final Decision Matrix

| Priority | Algorithm Choice |
|----------|------------------|
| **Speed** | Naive Bayes, Linear/Logistic Regression |
| **Accuracy** | XGBoost, Neural Networks, Ensemble |
| **Interpretability** | Linear/Logistic Regression, Decision Trees |
| **Small Data** | Naive Bayes, Linear Models, SVMs |
| **Large Data** | LightGBM, Neural Networks |
| **High Dimensions** | PCA + any model, Neural Networks |
| **Non-linearity** | Random Forest, XGBoost, Neural Networks |
| **Robustness** | Random Forest, XGBoost |
| **Real-time Prediction** | Naive Bayes, Linear Models |
| **No Feature Scaling** | Tree-based models (Random Forest, XGBoost) |

---

**Remember**: The best algorithm is often the one you understand well enough to tune properly!
