# Automated Customer Feedback Analyzer

![Demo GIF of the application](/gif/demo.gif)

This project is an end-to-end NLP application that analyzes customer reviews [(Amazon Fine Food Reviews - Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download) to provide actionable business insights. It automatically discovers key topics, analyzes sentiment, and generates concise summaries from large volumes of text data.

---

### Problem Statement
Businesses often receive thousands of customer reviews, making it impossible to manually read and categorize them all. This leads to missed opportunities for improvement and a poor understanding of customer sentiment. This tool automates the feedback analysis pipeline to solve that problem.

---

### Features
- **Automated Topic Modeling:** Uses BERTopic to discover and visualize the main themes in customer feedback without needing pre-labeled data.
- **Abstractive Summarization:** Leverages a T5 Transformer model from Hugging Face to generate human-like summaries for reviews within each topic.
- **Sentiment Analysis:** Performs sentiment analysis on reviews for each topic to quantify customer opinion.
- **Interactive Dashboard:** A user-friendly interface built with Streamlit that allows for easy exploration of topics and insights.

---

### Tech Stack
- **Core Libraries:** Python, Pandas
- **NLP:** spaCy, Hugging Face Transformers, BERTopic
- **Web App & Visualization:** Streamlit, Plotly
- **Environment:** Git, Virtual Environments

---

### Setup and Execution

Follow these steps to set up and run the project locally.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/chhuang216/customer-feedback-analyzer.git](https://github.com/chhuang216/customer-feedback-analyzer.git)
    cd customer-feedback-analyzer
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on macOS/Linux
    source venv/bin/activate

    # or

    # Activate on Windows
    # venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy Model**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Run the Application**
    The first time you run the app, it will train and save a topic model. Subsequent runs will be much faster.
    ```bash
    streamlit run app.py
    ```

---

### Project Structure
customer-feedback-analyzer/
├── data/
│   └── reviews_sample.csv
├── models/
│   └── (Saved BERTopic model will appear here)
├── .gitignore
├── app.py
├── analyzer.py
├── README.md
└── requirements.txt