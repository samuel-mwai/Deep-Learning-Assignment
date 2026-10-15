# 🧠 Deep Learning for Resume Screening and ATS Hiring Prediction

## 📌 Project Overview

This project applies **Deep Learning and Artificial Neural Networks (ANNs)** to a resume screening and applicant tracking system (ATS) dataset to predict whether a job applicant is selected or rejected.

Using Python and TensorFlow/Keras, the project develops a neural network classification model that learns patterns from candidate-related information and predicts recruitment outcomes.

The workflow covers data quality assessment, exploratory data analysis (EDA), statistical analysis, feature preprocessing, neural network development, model training, validation, and evaluation.

The project explores how deep learning can be applied to recruitment analytics to support data-driven candidate screening.

## 🎯 Project Objectives

- Explore a large resume screening dataset containing 100,000 candidate records.
- Investigate patterns in candidate characteristics and recruitment outcomes.
- Perform data quality checks and exploratory data analysis.
- Prepare numerical and categorical features for neural network training.
- Build a feedforward artificial neural network using TensorFlow/Keras.
- Train the model using validation data and early stopping.
- Evaluate predictions using classification metrics.
- Investigate the potential of deep learning in automated resume screening.

## 📊 Dataset

The project uses the **AI Resume Screening Dataset**, obtained from Kaggle.

The notebook loads the dataset from:

`ai_resume_screening_dataset.csv`

The dataset contains 100,000 candidate records and includes information such as:

- Candidate age and years of experience
- Education and field of study
- Current job title
- Previous companies
- Technical skills and programming languages
- Databases and cloud platforms
- Internship and leadership experience
- Certifications and professional profiles
- Recruitment selection outcome

### Target variable

`selected`

The target represents the recruitment outcome:

- **Selected** — The candidate was selected.
- **Rejected** — The candidate was rejected.

This is a **binary classification problem**.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | Exploratory data analysis |
| SciPy | Statistical hypothesis testing |
| Scikit-learn | Data preprocessing, splitting, and evaluation |
| TensorFlow/Keras | Neural network construction and training |
| Jupyter Notebook | Experimentation and development |

## 🔄 Project Workflow

### 1. Data Loading and Quality Assessment

The notebook loads the dataset into a Pandas DataFrame and performs initial data checks, including:

- Dataset dimensions and data types
- Missing-value counts
- Duplicate-record checks
- Descriptive statistics
- Unique values in categorical columns

These checks help establish the structure and quality of the dataset before modelling.

### 2. Exploratory Data Analysis (EDA)

EDA is used to explore the relationships between candidate characteristics and recruitment outcomes.

The notebook includes:

- Distribution of selected and rejected candidates
- Relationship between age and years of experience
- Selection outcomes by years of experience
- Selection rates by education level
- Correlation analysis of numerical variables

### 3. Statistical Analysis

The notebook applies the **Chi-square test of independence** to investigate associations between categorical variables and the recruitment outcome.

The analysis uses a significance level of 0.05 to determine whether there is sufficient statistical evidence of an association.

Statistical significance alone does not establish causation or prove that a feature is useful for prediction.

### 4. Feature Preparation

The machine learning workflow uses a preprocessing pipeline built with Scikit-learn.

**Numerical features**
- Standardized using `StandardScaler`.

**Categorical features**
- Encoded using `OneHotEncoder`.
- Configured to ignore previously unseen categories during transformation.

A `ColumnTransformer` combines both preprocessing pipelines.

The data is split into training, validation, and testing sets using stratified sampling:

- 70% training data
- 15% validation data
- 15% testing data

The split uses `random_state=42` to support reproducibility.

### 5. Deep Learning Model Architecture

The notebook builds a feedforward neural network using TensorFlow/Keras.

The architecture contains:

| Layer | Configuration |
|---|---|
| Input layer | Number of preprocessed features |
| Hidden layer 1 | 64 neurons, ReLU activation |
| Hidden layer 2 | 32 neurons, ReLU activation |
| Hidden layer 3 | 32 neurons, ReLU activation |
| Output layer | 3 neurons, Softmax activation in the notebook |

The hidden layers learn patterns from the input features, while the output layer produces class probabilities.

**Architecture note:** The target variable contains two classes, but the current notebook defines three output neurons. For a two-class problem, the output layer should be adjusted to match the intended binary classification setup.

### 6. Model Training

The neural network is compiled using:

- **Optimizer:** Adam
- **Loss function:** Sparse categorical cross-entropy
- **Metric:** Accuracy

The training configuration includes:

- Maximum of 200 epochs
- Batch size of 32
- Validation data for monitoring generalization
- Early stopping based on validation loss
- Patience of 5 epochs
- Restoration of the best model weights

Early stopping helps prevent unnecessary training when validation performance stops improving.

### 7. Model Evaluation

The trained model generates predictions on the test set. The predicted class is obtained using `argmax`, and performance is evaluated using Scikit-learn's classification report.

The evaluation metrics include:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positive cases that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

The notebook reports 100% test accuracy, precision, recall, and F1-scores for both classes.

**Important validation note:** These results should not be treated as verified model performance yet. The current notebook leaves `selected` in the feature DataFrame, which can leak the target into the model's inputs. The reported perfect scores are therefore not reliable evidence of generalization. The input features must be corrected and the model retrained before reporting final performance.

## 🚀 Getting Started

### Prerequisites

Install Python and the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn tensorflow jupyter
```

### Installation

Clone the repository:

```bash
git clone https://github.com/samuel-mwai/Deep-Learning-Assignment.git
```

Navigate to the project directory:

```bash
cd Deep-Learning-Assignment
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and run the cells in sequence.

**Dataset requirement:** Download the AI Resume Screening Dataset from Kaggle and update the notebook's input path to point to the local CSV file. The original notebook uses a Kaggle-specific file path.

## 🔬 Future Improvements

- Remove the target column from all input features to prevent data leakage.
- Align the output layer and loss function with the binary target.
- Retrain the model and report results from a leakage-free test set.
- Compare the neural network with baseline models such as Logistic Regression and Random Forest.
- Evaluate precision, recall, F1-score, and a confusion matrix.
- Investigate class imbalance and the reliability of the dataset's labels.
- Apply cross-validation or repeated experiments where appropriate.
- Save the fitted preprocessing pipeline and trained neural network for reproducible inference.
- Explore a simple application for demonstrating candidate outcome predictions.
- Assess fairness, privacy, and potential bias before considering any real recruitment use.

## ⚖️ Ethical Considerations

Recruitment models can reproduce or amplify biases present in historical data. Candidate information must be handled responsibly, and model predictions should not be treated as the sole basis for hiring decisions.

Any real-world application would require appropriate privacy safeguards, fairness testing, human oversight, and validation on representative data.

## 👨‍💻 Author

**Samuel Mwai**

Data Science & Analytics | Machine Learning | Deep Learning | Python

GitHub: [YOUR-GITHUB-PROFILE](https://github.com/samuel-mwai)

## 📄 License

No license has been specified. Add an appropriate license if you intend to permit reuse or redistribution of this project.

---

*This project demonstrates the development of a neural network classification workflow for resume screening, from data exploration and preprocessing to model training and evaluation.*
