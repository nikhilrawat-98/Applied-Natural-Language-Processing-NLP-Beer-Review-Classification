# Applied Natural Language Processing (NLP): Beer Review Classification

This project focuses on classifying beer reviews from the BeerAdvocate website into three rating categories: "okay," "good," and "excellent." By utilizing various machine learning classifiers and natural language processing (NLP) techniques, we aim to predict the review ratings based solely on the review text. The models were trained and evaluated using Term Frequency-Inverse Document Frequency (TF-IDF) vectorization techniques, and the best performing model (Support Vector Machine) was used for final predictions.

## Project Overview

- **Objective**: To build a predictive model capable of classifying beer reviews into one of three rating categories: "okay" (rating: 3.5-4), "good" (rating: 4-4.5), or "excellent" (rating: 4.5-5).
- **Key Focus**:
  - Exploratory data analysis to understand the distribution and key features of the reviews.
  - Use of TF-IDF for feature extraction from review text.
  - Comparison of various classifiers (Logistic Regression, Naive Bayes, Support Vector Machine, and Random Forest) for accuracy and performance.
  - Deployment of the best model (SVM) for predicting ratings on unseen reviews.
- **Tech Stack**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn, TF-IDF Vectorization

## Dataset

The dataset consists of user-generated beer reviews, each associated with a rating score. The task involves predicting one of three rating categories based on the review text.

### Dataset Features:
- `label`: Rating category (0 for "okay", 1 for "good", 2 for "excellent").
- `text`: Text of the beer review.
- **Train Set**: Contains 21,057 labeled reviews for training the model.
- **Test Set**: Contains 8,943 unlabeled reviews used for prediction.

### File Structure:
- **`Code.ipynb`**: Jupyter notebook containing the code for data preprocessing, model training, and evaluation.
- **`SVM predictions submission.csv`**: The final SVM model predictions on the test dataset.
- **`Report.pdf`**: Detailed report discussing the methodology, results, and future improvements.
- **`requirements.txt`**: Python dependencies required to run the analysis.

## Methodology

### 1. Exploratory Data Analysis (EDA)
- **Class Distribution**: Analyzed the distribution of the three rating categories to ensure no significant class imbalance.
- **Word Clouds**: Generated word clouds for each rating category to visualize the most common words and phrases in reviews.
- **Review Length Distribution**: Examined the distribution of review lengths across different rating categories to identify any correlations between review length and rating.

### 2. Data Preprocessing
- **Text Cleaning**: Removed stop words, punctuation, and applied lemmatization to normalize the text.
- **Tokenization**: Split the text into individual words (tokens) for easier analysis.
- **TF-IDF Vectorization**: Transformed the review text into numerical features using the TF-IDF technique. TF-IDF captures the importance of a word relative to its frequency across the entire document corpus.

### 3. Machine Learning Models
- **Logistic Regression**: A simple, interpretable classifier known for its efficiency with high-dimensional data.
- **Naive Bayes**: Effective for text classification due to its strong assumption of feature independence.
- **Support Vector Machine (SVM)**: A robust classifier that performed best in our analysis, handling high-dimensional text data effectively.
- **Random Forest**: Evaluated but did not perform as well as SVM in this particular task due to the high-dimensional nature of the data.

## Results

The Support Vector Machine (SVM) outperformed other classifiers in terms of accuracy, precision, recall, and F1 score. Below are the performance metrics for the best-performing models:

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 59.90%   | 59.19%    | 60.01% | 59.46%   |
| Naive Bayes           | 58.36%   | 57.60%    | 58.48% | 57.82%   |
| Support Vector Machine| 60.76%   | 60.29%    | 60.85% | 60.50%   |

SVM showed the best overall performance and was chosen for the final predictions on the test dataset.

## Future Improvements

- **Advanced NLP Models**: Future work could involve experimenting with transformer-based models like BERT or GPT to capture deeper semantic relationships within the text.
- **Sentiment Analysis**: Incorporating sentiment analysis as an additional feature could improve classification by providing more context to the review text.
- **Inclusion of Metadata**: Adding user-specific information (e.g., user ratings, review date) may help improve the overall accuracy of the model.
- **Transfer Learning**: Using pre-trained models such as BERT or GPT could further enhance the performance of the classifier.

## Usage

The Jupyter notebook demonstrates the following:
1. **Data Preprocessing**: Cleaning and transforming the text into numerical features using TF-IDF.
2. **Model Training and Evaluation**: Training various classifiers (Logistic Regression, Naive Bayes, SVM) and evaluating their performance.
3. **Prediction**: Using the best-performing model (SVM) to predict ratings on unseen reviews.

## Installation

To run the project locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Beer-Review-Classification-NLP.git
    cd Beer-Review-Classification-NLP
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook:
    ```bash
    jupyter notebook
    ```

## Acknowledgements

This project was developed as part of the MSc in Business Analytics at Bayes Business School, under the module **Applied Natural Language Processing (NLP)**.

## License

MIT License
