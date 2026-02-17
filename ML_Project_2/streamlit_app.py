"""
Sentiment Analysis of Internship Feedback - Streamlit App
==========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import string
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Set page config
st.set_page_config(
    page_title="Internship Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    h1 {color: #2c3e50; padding-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_sample_feedback(n_samples=500):
    """Generate synthetic internship feedback data"""
    
    positive_feedback = [
        "Great learning experience, mentors were very supportive",
        "Excellent work environment and challenging projects",
        "Amazing team culture, learned a lot about industry practices",
        "The training program was comprehensive and well-structured",
        "Got hands-on experience with cutting-edge technologies",
        "Mentorship was outstanding, helped me grow professionally",
        "Work-life balance was good, flexible working hours",
        "Real-world projects helped me apply my academic knowledge",
        "Team members were friendly and always willing to help",
        "The internship exceeded my expectations in every way"
    ]
    
    negative_feedback = [
        "Lack of proper guidance from mentors",
        "The work was monotonous and didn't match job description",
        "Poor communication from management team",
        "No structured training program provided",
        "Limited learning opportunities, mostly menial tasks",
        "Supervisors were unapproachable and unhelpful",
        "Unrealistic expectations with tight deadlines",
        "Poor work-life balance, excessive working hours",
        "Inadequate resources and outdated technology",
        "Felt undervalued and work was not appreciated"
    ]
    
    neutral_feedback = [
        "The internship was okay, nothing special",
        "Some projects were interesting, others were not",
        "Average experience overall",
        "Met basic expectations but nothing more"
    ]
    
    data = []
    
    for _ in range(int(n_samples * 0.4)):
        feedback = np.random.choice(positive_feedback) + " " + np.random.choice(positive_feedback)
        data.append({'feedback': feedback, 'sentiment': 'positive', 'rating': np.random.randint(4, 6)})
    
    for _ in range(int(n_samples * 0.35)):
        feedback = np.random.choice(negative_feedback) + " " + np.random.choice(negative_feedback)
        data.append({'feedback': feedback, 'sentiment': 'negative', 'rating': np.random.randint(1, 3)})
    
    for _ in range(int(n_samples * 0.25)):
        feedback = np.random.choice(neutral_feedback)
        data.append({'feedback': feedback, 'sentiment': 'neutral', 'rating': 3})
    
    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)

def preprocess_text(text):
    """Clean and preprocess feedback text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

class LogisticRegressionSentimentAnalyzer:
    """Sentiment analyzer using Logistic Regression"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        
    def train(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)
        return self
    
    def predict(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vec)
    
    def predict_proba(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.model.predict_proba(X_test_vec)
    
    def get_top_features(self, n=15):
        feature_names = self.vectorizer.get_feature_names_out()
        results = {}
        for idx, sentiment in enumerate(self.model.classes_):
            coef = self.model.coef_[idx]
            top_indices = coef.argsort()[-n:][::-1]
            top_features = [(feature_names[i], coef[i]) for i in top_indices]
            results[sentiment] = top_features
        return results

@st.cache_data
def load_or_generate_data(use_sample=True, uploaded_file=None):
    if use_sample or uploaded_file is None:
        df = generate_sample_feedback(n_samples=500)
    else:
        df = pd.read_csv(uploaded_file)
    df['feedback_clean'] = df['feedback'].apply(preprocess_text)
    return df

@st.cache_resource
def train_model(df):
    X = df['feedback_clean']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegressionSentimentAnalyzer()
    model.train(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("📊 Sentiment Analysis of Internship Feedback")
    st.markdown("### Analyze intern feedback to improve internship programs")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    data_source = st.sidebar.radio("Data Source:", ["Use Sample Data", "Upload CSV File"])
    
    uploaded_file = None
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    
    if data_source == "Upload CSV File" and uploaded_file is None:
        st.info("👆 Please upload a CSV file or switch to sample data")
        return
    
    with st.spinner("Loading data..."):
        df = load_or_generate_data(use_sample=(data_source == "Use Sample Data"), uploaded_file=uploaded_file)
    
    with st.spinner("Training model..."):
        model, X_train, X_test, y_train, y_test = train_model(df)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Analyze", "📊 Insights", "💡 Recommendations"])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.header("Overview Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        sentiment_counts = df['sentiment'].value_counts()
        
        with col1:
            st.metric("Total Feedback", len(df))
        with col2:
            positive_pct = (sentiment_counts.get('positive', 0) / len(df)) * 100
            st.metric("Positive", f"{positive_pct:.1f}%")
        with col3:
            negative_pct = (sentiment_counts.get('negative', 0) / len(df)) * 100
            st.metric("Negative", f"{negative_pct:.1f}%")
        with col4:
            st.metric("Model Accuracy", f"{accuracy:.1%}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            fig = px.pie(
                values=sentiment_counts.values, 
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={
                    'positive': '#27ae60', 
                    'negative': '#e74c3c', 
                    'neutral': '#95a5a6'
                },
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Rating Distribution")
            if 'rating' in df.columns:
                rating_counts = df['rating'].value_counts().sort_index()
                fig = px.bar(
                    x=rating_counts.index, 
                    y=rating_counts.values, 
                    labels={'x': 'Rating', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=model.model.classes_)
        fig = px.imshow(
            cm, 
            labels=dict(x="Predicted", y="Actual"), 
            x=model.model.classes_, 
            y=model.model.classes_,
            color_continuous_scale='Blues', 
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: ANALYZE
    with tab2:
        st.header("🔍 Analyze New Feedback")
        
        user_feedback = st.text_area(
            "Enter feedback:", 
            height=150, 
            placeholder="Example: The mentorship was great..."
        )
        
        if st.button("🔎 Analyze Sentiment", type="primary"):
            if user_feedback:
                clean_feedback = preprocess_text(user_feedback)
                prediction = model.predict([clean_feedback])[0]
                probabilities = model.predict_proba([clean_feedback])[0]
                
                sentiment_colors = {
                    'positive': '#27ae60', 
                    'negative': '#e74c3c', 
                    'neutral': '#95a5a6'
                }
                sentiment_emojis = {
                    'positive': '😊', 
                    'negative': '😞', 
                    'neutral': '😐'
                }
                
                # Fixed the f-string formatting here
                color = sentiment_colors[prediction]
                emoji = sentiment_emojis[prediction]
                confidence = max(probabilities)
                
                st.markdown(f"""
                    <div style='text-align: center; padding: 2rem; 
                                background-color: {color}20; 
                                border-radius: 10px; border: 2px solid {color}'>
                        <h1>{emoji}</h1>
                        <h2 style='color: {color}'>{prediction.upper()}</h2>
                        <p style='font-size: 1.2rem'>Confidence: {confidence:.1%}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Confidence Breakdown")
                classes = model.model.classes_
                prob_dict = dict(zip(classes, probabilities))
                
                for sentiment in ['positive', 'neutral', 'negative']:
                    prob = prob_dict[sentiment]
                    st.markdown(f"**{sentiment.capitalize()}**")
                    st.progress(float(prob))
                    st.caption(f"{prob:.1%}")
            else:
                st.warning("Please enter feedback to analyze")
    
    # TAB 3: INSIGHTS
    with tab3:
        st.header("📊 Detailed Insights")
        st.subheader("🔑 Key Words by Sentiment")
        
        top_features = model.get_top_features(n=15)
        sentiment_colors = {
            'positive': '#27ae60', 
            'negative': '#e74c3c', 
            'neutral': '#95a5a6'
        }
        
        col1, col2, col3 = st.columns(3)
        
        for col, sentiment in zip([col1, col2, col3], ['positive', 'negative', 'neutral']):
            with col:
                st.markdown(f"### {sentiment.capitalize()}")
                if sentiment in top_features:
                    features = top_features[sentiment]
                    words = [f[0] for f in features]
                    scores = [f[1] for f in features]
                    
                    fig = go.Figure(go.Bar(
                        x=scores, 
                        y=words, 
                        orientation='h',
                        marker=dict(color=sentiment_colors[sentiment])
                    ))
                    fig.update_layout(
                        height=500, 
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("☁️ Word Clouds")
        
        col1, col2, col3 = st.columns(3)
        
        for col, sentiment in zip([col1, col2, col3], ['positive', 'negative', 'neutral']):
            with col:
                st.markdown(f"### {sentiment.capitalize()}")
                sentiment_text = ' '.join(df[df['sentiment'] == sentiment]['feedback_clean'].values)
                
                if sentiment_text.strip():
                    colormap = 'Greens' if sentiment == 'positive' else ('Reds' if sentiment == 'negative' else 'Greys')
                    
                    wordcloud = WordCloud(
                        width=400, 
                        height=300, 
                        background_color='white',
                        colormap=colormap
                    ).generate(sentiment_text)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
    
    # TAB 4: RECOMMENDATIONS
    with tab4:
        st.header("💡 Actionable Recommendations")
        
        sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
        positive_pct = sentiment_dist.get('positive', 0)
        negative_pct = sentiment_dist.get('negative', 0)
        neutral_pct = sentiment_dist.get('neutral', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Strengths")
            positive_feedback = df[df['sentiment'] == 'positive']['feedback']
            strengths = {
                'Mentorship': ['mentor', 'guidance', 'support'],
                'Learning': ['learn', 'training', 'experience'],
                'Work Environment': ['culture', 'team', 'friendly'],
                'Projects': ['project', 'hands-on', 'challenging']
            }
            
            for area, keywords in strengths.items():
                count = positive_feedback.str.contains('|'.join(keywords), case=False).sum()
                if count > 0:
                    st.success(f"**{area}**: Mentioned {count} times")
        
        with col2:
            st.markdown("### ⚠️ Areas for Improvement")
            negative_feedback = df[df['sentiment'] == 'negative']['feedback']
            concerns = {
                'Mentorship Quality': ['lack', 'unapproachable'],
                'Work Structure': ['monotonous', 'disorganized'],
                'Communication': ['poor communication', 'limited'],
                'Work-Life Balance': ['excessive', 'unrealistic']
            }
            
            for area, keywords in concerns.items():
                count = negative_feedback.str.contains('|'.join(keywords), case=False).sum()
                if count > 0:
                    st.warning(f"**{area}**: Mentioned {count} times")
        
        st.markdown("---")
        st.subheader("📋 Priority Actions")
        
        if negative_pct > 30:
            st.error("🚨 HIGH PRIORITY: Negative sentiment > 30%")
            st.markdown("""
            **Actions:**
            1. Conduct mentor training
            2. Establish onboarding process
            3. Implement feedback mechanisms
            """)
        
        if neutral_pct > 25:
            st.info("📌 ENGAGE: High neutral sentiment")
            st.markdown("""
            **Actions:**
            1. Assign challenging projects
            2. Increase mentorship time
            3. Foster team integration
            """)
        
        if positive_pct > 50:
            st.success("🌟 LEVERAGE: Strong positive feedback")
            st.markdown("""
            **Capitalize:**
            1. Use in recruitment marketing
            2. Build alumni network
            3. Showcase success stories
            """)

if __name__ == "__main__":
    main()