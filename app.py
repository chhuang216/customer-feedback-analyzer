import streamlit as st
import pandas as pd
from analyzer import FeedbackAnalyzer

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="📊",
    layout="wide"
)

# --- Caching ---
# Cache the analyzer instance to prevent reloading models on each interaction.
@st.cache_resource
def get_analyzer():
    data_path = 'data/samples.csv' 
    return FeedbackAnalyzer(data_path=data_path, num_samples=1000)

# Cache the topic modeling results to avoid re-computation.
@st.cache_data
def run_topic_modeling(_analyzer):
    _analyzer.perform_topic_modeling()
    return _analyzer.get_topic_info()

# --- Main Application UI ---
st.title("📊 Automated Customer Feedback Analyzer")
st.markdown("""
This tool uses NLP to automatically analyze customer reviews. It discovers key topics, 
analyzes sentiment, and generates summaries to provide actionable insights.
""")

# --- Load Data and Run Analysis ---
with st.spinner('Initializing Analyzer and processing data... This may take a moment.'):
    analyzer = get_analyzer()
    topic_info_df = run_topic_modeling(analyzer)

st.success("Analysis complete! Select a topic from the sidebar to explore.")
st.divider()

# --- Sidebar for Topic Selection ---
with st.sidebar:
    st.header("Explore Topics")
    # Exclude the outlier topic (-1)
    selectable_topics = topic_info_df[topic_info_df.Topic != -1].copy()
    
    if selectable_topics.empty:
        st.warning("No topics were identified. Please try a different dataset.")
    else:
        selectable_topics['TopicName'] = selectable_topics['Name'].apply(lambda x: f"Topic {x.split('_')[0]}: {x.split('_')[1].capitalize()}")
        
        selected_topic_name = st.selectbox(
            "Choose a topic:",
            options=selectable_topics['TopicName']
        )
        
        # Extract topic ID from the user-friendly name
        selected_topic_id = int(selected_topic_name.split(":")[0].replace("Topic ", ""))

# --- Main Panel to Display Results ---
if 'selected_topic_id' in locals():
    st.header(f"Insights for: {selected_topic_name}")

    # Fetch details for the selected topic
    summary = analyzer.summarize_topic(selected_topic_id)
    sentiment_fig = analyzer.analyze_sentiment_for_topic(selected_topic_id)
    sample_reviews = analyzer.get_reviews_for_topic(selected_topic_id)[:5]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Automated Summary")
        st.info(summary)

        st.subheader("💬 Sample Reviews")
        for review in sample_reviews:
            st.markdown(f"- *{review[:150]}...*")

    with col2:
        st.subheader("😊 Sentiment")
        if sentiment_fig:
            st.plotly_chart(sentiment_fig, use_container_width=True)
        else:
            st.write("Could not generate sentiment chart for this topic.")
else:
    st.info("Awaiting topic selection from the sidebar.")