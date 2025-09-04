import pandas as pd
import spacy
from bertopic import BERTopic
from transformers import pipeline
import plotly.express as px
import warnings
import os # Import the os module

# Suppress FutureWarning from Hugging Face Transformers
warnings.simplefilter(action='ignore', category=FutureWarning)

class FeedbackAnalyzer:
    """
    A class to handle the core NLP tasks for customer feedback analysis.
    Supports saving and loading the BERTopic model.
    """

    def __init__(self, data_path, num_samples=1000):
        """Initializes the analyzer by loading models and data."""
        print("Loading NLP models...")
        self.nlp = spacy.load("en_core_web_sm")
        self.summarizer = pipeline("summarization", model="t5-small")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        print("Models loaded successfully.")
        
        self.model_path = "models/bertopic_model" # Define path for saving/loading
        self.topic_model = None # Will be loaded or trained later
        self.df = self.load_and_sample_data(data_path, num_samples)
        self.topics = None

    def load_and_sample_data(self, path, n_samples):
        """Loads data from a CSV, validates it, and takes a random sample."""
        df = pd.read_csv(path)
        if 'Text' not in df.columns:
            raise ValueError("CSV must have a 'Text' column.")
        df = df[['Text']].dropna().copy()
        return df.sample(n=min(len(df), n_samples), random_state=42)

    def preprocess_text(self, text):
        """Lemmatizes and removes stopwords from a single text string."""
        doc = self.nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

    def perform_topic_modeling(self):
        """
        Loads a pre-trained topic model if available. 
        Otherwise, trains a new model and saves it.
        """
        print("Preprocessing text for topic modeling...")
        preprocessed_texts = self.df['Text'].apply(self.preprocess_text).reset_index(drop=True) # <-- ADD THIS

        # Check if a model is already saved
        if os.path.exists(self.model_path):
            print(f"Loading existing topic model from {self.model_path}...")
            self.topic_model = BERTopic.load(self.model_path)
        else:
            print("No existing model found. Training a new BERTopic model...")
            # Create a new model instance for training
            self.topic_model = BERTopic(verbose=False, min_topic_size=10)
            
            # Use .fit_transform() here since it's the first time training
            self.topics, _ = self.topic_model.fit_transform(preprocessed_texts)
            
            # Create the models directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"Saving new model to {self.model_path}...")
            self.topic_model.save(self.model_path)
            self.df['Topic'] = self.topics
            print("Topic modeling complete.")
            return # Exit function after training and saving

        # This part runs ONLY if the model was loaded
        print("Applying topic model to the current data sample...")
        self.topics, _ = self.topic_model.transform(preprocessed_texts)
        self.df['Topic'] = self.topics
        print("Topic modeling complete.")

    def get_topic_info(self):
        """Returns a DataFrame with information about each identified topic."""
        return self.topic_model.get_topic_info()

    def get_reviews_for_topic(self, topic_id):
        """Returns all review texts belonging to a specific topic ID."""
        return self.df[self.df['Topic'] == topic_id]['Text'].tolist()

    def summarize_topic(self, topic_id):
        """Generates a concise summary for all reviews within a given topic."""
        reviews_text = " ".join(self.get_reviews_for_topic(topic_id))
        
        max_chars = 1500  
        if not reviews_text.strip():
            return "Not enough text to generate a summary."

        summary = self.summarizer(reviews_text[:max_chars], max_length=120, min_length=30, do_sample=False)
        return summary[0]['summary_text']

    def analyze_sentiment_for_topic(self, topic_id):
        """Analyzes sentiment for a topic and returns a Plotly pie chart."""
        reviews = self.get_reviews_for_topic(topic_id)
        if not reviews:
            return None

        sentiments = self.sentiment_analyzer(reviews, truncation=True)
        sentiment_df = pd.DataFrame(sentiments)
        sentiment_counts = sentiment_df['label'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        fig = px.pie(sentiment_counts, names='Sentiment', values='Count', 
                     title=f'Sentiment Distribution',
                     color='Sentiment',
                     color_discrete_map={'POSITIVE': '#1f77b4', 'NEGATIVE': '#d62728'})
        return fig