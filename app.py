import streamlit as st
from time import sleep
import pandas as pd
from transformers import pipeline
import json
import spacy
import spacy_streamlit
import plotly.express as px
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="NextGen NLP Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress .st-progress-bar {
        background-color: #1f77b4;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    .stButton button {
        border-radius: 20px;
        padding: 0.5rem 2rem;
    }
    div.stButton > button:hover {
        background-color: #1f77b4;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            st.error("PDF support requires additional dependencies")
            return ""
        else:
            st.warning("Please upload a text file")
            return ""
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""

def simple_tokenize(text):
    sentences = re.split('[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\b\w+\b', text.lower())
    return sentences, words

def create_word_cloud(text):
    try:
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                     'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                     'that', 'the', 'to', 'was', 'were', 'will', 'with'}
        words = re.findall(r'\b\w+\b', text.lower())
        words = [word for word in words if word not in stop_words]
        word_freq = Counter(words)
        return word_freq
    except Exception as e:
        st.error(f"Error in word cloud generation: {str(e)}")
        return Counter()

def text_statistics(text):
    try:
        if not text:
            return {
                'Sentences': 0,
                'Words': 0,
                'Characters': 0,
                'Average Word Length': 0,
                'Average Sentence Length': 0
            }
        
        sentences, words = simple_tokenize(text)
        char_count = len(text)
        
        return {
            'Sentences': len(sentences),
            'Words': len(words),
            'Characters': char_count,
            'Average Word Length': round(sum(len(word) for word in words) / len(words), 2) if words else 0,
            'Average Sentence Length': round(len(words) / len(sentences), 2) if sentences else 0
        }
    except Exception as e:
        st.error(f"Error calculating text statistics: {str(e)}")
        return {
            'Sentences': 0,
            'Words': 0,
            'Characters': 0,
            'Average Word Length': 0,
            'Average Sentence Length': 0
        }

def loading_spinner():
    with st.spinner('🔄 Processing your request... Please wait'):
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            sleep(0.01)

@st.cache_resource
def load_models():
    models = {}
    try:
        models['summarizer'] = pipeline('summarization')
        models['sentiment'] = pipeline('sentiment-analysis')
        models['qa'] = pipeline('question-answering')
        models['text_gen'] = pipeline('text-generation')
        models['nlp'] = spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
    return models

def main():
    try:
        models = load_models()
        
        with st.sidebar:
            st.image("logo2.webp", caption="NextGen NLP Hub")
            st.title("Navigation")
            choice = st.radio(
                "Choose a Feature",
                ["Home", "Text Analysis Dashboard", "Summarizer", "NER Analysis", 
                 "Sentiment Analysis", "Q&A System", "Text Completion"]
            )
            
            st.markdown("---")
            st.markdown("### About")
            st.info("NextGen NLP Hub: Your all-in-one platform for advanced natural language processing tasks.")

        if choice == "Home":
            st.title("🤖 Welcome to NextGen NLP Hub")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                ### Transform the way you work with text using AI
                
                NextGen NLP Hub offers state-of-the-art natural language processing capabilities:
                
                - 📊 **Text Analysis Dashboard**: Comprehensive text statistics and visualizations
                - 📝 **Advanced Summarization**: Generate concise summaries of long texts
                - 🎯 **Named Entity Recognition**: Identify and classify key entities
                - 🎭 **Sentiment Analysis**: Understand emotional tone and sentiment
                - ❓ **Question Answering**: Get precise answers from context
                - ✨ **Text Completion**: Generate human-like text continuations
                """)

            with col2:
                st.image("image2.webp", caption="NLP Capabilities")

        elif choice == "Text Analysis Dashboard":
            st.title("📊 Text Analysis Dashboard")
            uploaded_file = st.file_uploader("Upload a text file for analysis", type=['txt'])
            text_input = st.text_area("Or enter your text for analysis", height=200)
            
            analysis_text = read_uploaded_file(uploaded_file) if uploaded_file else text_input
            
            if analysis_text:
                loading_spinner()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Text Statistics")
                    stats = text_statistics(analysis_text)
                    for key, value in stats.items():
                        st.metric(key, value)
                
                with col2:
                    st.markdown("### 🔤 Word Frequency")
                    word_freq = create_word_cloud(analysis_text)
                    if word_freq:
                        fig = px.bar(
                            x=list(word_freq.keys())[:10],
                            y=list(word_freq.values())[:10],
                            title="Top 10 Most Common Words"
                        )
                        st.plotly_chart(fig)

        elif choice == "Summarizer":
            st.title("📝 Advanced Text Summarizer")
            uploaded_file = st.file_uploader("Upload a text file to summarize", type=['txt'])
            text_input = st.text_area("Or enter the text to summarize", height=200)
            
            summarize_text = read_uploaded_file(uploaded_file) if uploaded_file else text_input
            
            if summarize_text:
                try:
                    summary = models['summarizer'](summarize_text, 
                                                 min_length=30,
                                                 max_length=150)[0]['summary_text']
                    st.success("Summary generated successfully!")
                    st.markdown(f"### Summary\n{summary}")
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")

        elif choice == "NER Analysis":
            st.title("🎯 Named Entity Recognition")
            uploaded_file = st.file_uploader("Upload a text file for entity recognition", type=['txt'])
            text_input = st.text_area("Or enter text for entity recognition", height=200)
            
            ner_text = read_uploaded_file(uploaded_file) if uploaded_file else text_input
            
            if ner_text:
                try:
                    doc = models['nlp'](ner_text)
                    spacy_streamlit.visualize_ner(
                        doc,
                        labels=models['nlp'].get_pipe("ner").labels,
                        show_table=True,
                        title="Identified Entities"
                    )
                except Exception as e:
                    st.error(f"Error in NER analysis: {str(e)}")

        elif choice == "Sentiment Analysis":
            st.title("🎭 Sentiment Analysis")
            uploaded_file = st.file_uploader("Upload a text file for sentiment analysis", type=['txt'])
            text_input = st.text_area("Or enter text for sentiment analysis", height=200)
            
            sentiment_text = read_uploaded_file(uploaded_file) if uploaded_file else text_input
            
            if sentiment_text:
                try:
                    result = models['sentiment'](sentiment_text)[0]
                    sentiment = result['label']
                    score = result['score']
                    
                    if sentiment == "POSITIVE":
                        st.success(f"😊 Positive Sentiment (Confidence: {score:.2%})")
                    elif sentiment == "NEGATIVE":
                        st.error(f"😔 Negative Sentiment (Confidence: {score:.2%})")
                    else:
                        st.info(f"😐 Neutral Sentiment (Confidence: {score:.2%})")
                except Exception as e:
                    st.error(f"Error in sentiment analysis: {str(e)}")

        elif choice == "Q&A System":
            st.title("❓ Question Answering System")
            uploaded_file = st.file_uploader("Upload a text file containing the context", type=['txt'])
            context = st.text_area("Or enter the context", height=200)
            
            context_text = read_uploaded_file(uploaded_file) if uploaded_file else context
            question = st.text_input("Ask your question")
            
            if context_text and question:
                try:
                    result = models['qa'](question=question, context=context_text)
                    st.success("Answer found!")
                    st.markdown(f"### Answer\n{result['answer']}")
                except Exception as e:
                    st.error(f"Error in Q&A system: {str(e)}")

        elif choice == "Text Completion":
            st.title("✨ AI Text Completion")
            uploaded_file = st.file_uploader("Upload a text file to complete", type=['txt'])
            text_input = st.text_area("Or enter the text to complete", height=150)
            
            completion_text = read_uploaded_file(uploaded_file) if uploaded_file else text_input
            
            col1, col2 = st.columns(2)
            with col1:
                max_length = st.slider("Maximum length", 50, 500, 100)
            with col2:
                temperature = st.slider("Temperature (creativity)", 0.1, 1.0, 0.7)
            
            if st.button("Generate Completion"):
                if completion_text:
                    loading_spinner()
                    try:
                        completion = models['text_gen'](
                            completion_text,
                            max_length=max_length,
                            temperature=temperature
                        )[0]['generated_text']
                        
                        st.markdown("### Generated Text")
                        st.write(completion)
                        
                        st.markdown("### Original vs Generated")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Text:**")
                            st.markdown(completion_text)
                        with col2:
                            st.markdown("**Generated Continuation:**")
                            st.markdown(completion[len(completion_text):])
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == '__main__':
    main()