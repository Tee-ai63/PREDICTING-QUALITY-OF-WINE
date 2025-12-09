# wine_quality_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A4A4A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #8B0000;
        margin: 1rem 0;
    }
    .feature-importance {
        background-color: #E8F4F8;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stSlider > div > div > div {
        color: #8B0000;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained model and scaler"""
    try:
        with open('wine_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please run train_and_save_model.py first.")
        st.stop()

def engineer_features(input_dict):
    """Add engineered features to the input"""
    df = pd.DataFrame([input_dict])
    
    # Create engineered features
    df['acid_balance'] = df['fixed acidity'] / (df['pH'] + 1e-6)
    df['alcohol_acidity_ratio'] = df['alcohol'] / (df['fixed acidity'] + 1e-6)
    df['free_so2_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-6)
    
    return df

def predict_quality(input_features):
    """Make prediction using the loaded model"""
    model, scaler, feature_names = load_models()
    
    # Ensure input has all features in correct order
    input_df = pd.DataFrame([input_features])
    input_df = input_df[feature_names]
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Clip prediction to realistic range
    prediction = np.clip(prediction, 3, 8)
    
    return prediction

def main():
    # Title and description
    st.markdown('<h1 class="main-header">🍷 Wine Quality Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    This app predicts wine quality based on chemical properties using machine learning. 
    The model was trained on the WineQT dataset and can predict quality scores from 3 (poor) to 8 (excellent).
    """)
    
    # Load models
    model, scaler, feature_names = load_models()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Prediction", 
        "📊 Feature Importance", 
        "📈 Data Insights", 
        "ℹ️ About"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Enter Wine Chemical Properties</h2>', unsafe_allow_html=True)
        
        # Create two columns for input sliders
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Properties")
            fixed_acidity = st.slider(
                "Fixed Acidity (g/L)",
                min_value=4.0,
                max_value=16.0,
                value=7.0,
                step=0.1,
                help="Non-volatile acids that do not evaporate"
            )
            
            volatile_acidity = st.slider(
                "Volatile Acidity (g/L)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.01,
                help="Amount of acetic acid - high values cause vinegar taste"
            )
            
            citric_acid = st.slider(
                "Citric Acid (g/L)",
                min_value=0.0,
                max_value=1.5,
                value=0.3,
                step=0.01,
                help="Adds freshness and flavor"
            )
            
            residual_sugar = st.slider(
                "Residual Sugar (g/L)",
                min_value=0.5,
                max_value=30.0,
                value=2.0,
                step=0.1,
                help="Amount of sugar remaining after fermentation"
            )
            
            chlorides = st.slider(
                "Chlorides (g/L)",
                min_value=0.01,
                max_value=0.8,
                value=0.08,
                step=0.001,
                help="Amount of salt"
            )
            
        with col2:
            st.markdown("#### Advanced Properties")
            free_sulfur_dioxide = st.slider(
                "Free SO₂ (mg/L)",
                min_value=1.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
                help="Free form of SO₂ - prevents microbial growth"
            )
            
            total_sulfur_dioxide = st.slider(
                "Total SO₂ (mg/L)",
                min_value=5.0,
                max_value=300.0,
                value=45.0,
                step=1.0,
                help="Free + bound forms of SO₂"
            )
            
            density = st.slider(
                "Density (g/cm³)",
                min_value=0.98,
                max_value=1.05,
                value=0.997,
                step=0.001,
                help="Density of wine"
            )
            
            pH = st.slider(
                "pH",
                min_value=2.7,
                max_value=4.0,
                value=3.2,
                step=0.01,
                help="Acidity level (lower = more acidic)"
            )
            
            sulphates = st.slider(
                "Sulphates (g/L)",
                min_value=0.3,
                max_value=2.0,
                value=0.6,
                step=0.01,
                help="Additives for preservation"
            )
            
            alcohol = st.slider(
                "Alcohol (%)",
                min_value=8.0,
                max_value=15.0,
                value=10.5,
                step=0.1,
                help="Alcohol content by volume"
            )
        
        # Create input dictionary
        input_features = {
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol
        }
        
        # Add engineered features
        engineered_df = engineer_features(input_features)
        
        # Create prediction button
        if st.button("Predict Wine Quality", type="primary", use_container_width=True):
            with st.spinner("Analyzing wine properties..."):
                # Make prediction
                prediction = predict_quality(engineered_df.iloc[0])
                
                # Display prediction with styling
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                # Quality score display
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.metric(
                        label="Predicted Quality Score",
                        value=f"{prediction:.1f}",
                        delta=None
                    )
                
                with col_b:
                    # Quality interpretation
                    if prediction >= 7.5:
                        quality_text = "Excellent"
                        quality_color = "green"
                        emoji = "🏆"
                    elif prediction >= 6.5:
                        quality_text = "Very Good"
                        quality_color = "blue"
                        emoji = "⭐"
                    elif prediction >= 5.5:
                        quality_text = "Good"
                        quality_color = "orange"
                        emoji = "👍"
                    elif prediction >= 4.5:
                        quality_text = "Average"
                        quality_color = "yellow"
                        emoji = "⚖️"
                    else:
                        quality_text = "Below Average"
                        quality_color = "red"
                        emoji = "⚠️"
                    
                    st.markdown(f"""
                    **Quality Assessment:** <span style='color:{quality_color}; font-weight:bold'>{quality_text} {emoji}</span>
                    
                    *Score interpretation: 3 (poor) to 8 (excellent)*
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Quality scale visualization
                st.markdown("#### Quality Scale")
                quality_values = [3, 4, 5, 6, 7, 8]
                quality_labels = ["Poor", "Below Avg", "Average", "Good", "Very Good", "Excellent"]
                
                # Create a visual scale
                scale_html = """
                <div style='background: linear-gradient(to right, #FF6B6B, #FFE66D, #4ECDC4); 
                padding: 10px; border-radius: 5px; margin: 10px 0; position: relative;'>
                """
                
                for i, (val, label) in enumerate(zip(quality_values, quality_labels)):
                    position = (i / (len(quality_values)-1)) * 100
                    scale_html += f"""
                    <div style='position: absolute; left: {position}%; transform: translateX(-50%); 
                    text-align: center; font-weight: bold;'>
                        <div>{val}</div>
                        <div style='font-size: 0.8em;'>{label}</div>
                    </div>
                    """
                
                # Add prediction marker
                pred_position = ((prediction - 3) / 5) * 100
                scale_html += f"""
                <div style='position: absolute; left: {pred_position}%; transform: translateX(-50%); 
                top: -20px; color: #8B0000; font-weight: bold;'>
                    ↓<br>Prediction
                </div>
                """
                
                scale_html += "</div>"
                st.markdown(scale_html, unsafe_allow_html=True)
                
                # Show feature importance for this prediction
                st.markdown("#### Key Influencing Factors")
                if hasattr(model, 'feature_importances_'):
                    # Get feature importance for context
                    feat_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(5)
                    
                    for _, row in feat_importance.iterrows():
                        st.markdown(f"""
                        <div class="feature-importance">
                            **{row['Feature']}**: {row['Importance']*100:.1f}% importance
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Feature Importance Analysis</h2>', unsafe_allow_html=True)
        
        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            feat_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Create two columns for visualization
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = feat_importance.head(10)
                bars = ax.barh(top_features['Feature'], top_features['Importance'])
                ax.set_xlabel('Importance Score')
                ax.set_title('Top 10 Most Important Features')
                ax.invert_yaxis()
                
                # Color bars based on importance
                for i, bar in enumerate(bars):
                    bar.set_color(plt.cm.RdYlBu(i/len(bars)))
                
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### How Features Affect Quality")
                st.markdown("""
                **🔴 Negative Impact:**
                - Volatile Acidity (vinegar taste)
                - Chlorides (salty taste)
                
                **🟢 Positive Impact:**
                - Alcohol (body & complexity)
                - Sulphates (preservation)
                - Citric Acid (freshness)
                
                **⚖️ Engineered Features:**
                - Acid Balance
                - Alcohol-Acidity Ratio
                """)
                
                # Show top features table
                st.markdown("#### Top 5 Features")
                st.dataframe(
                    feat_importance.head(5)[['Feature', 'Importance']].round(4),
                    use_container_width=True,
                    hide_index=True
                )
    
    with tab3:
        st.markdown('<h2 class="sub-header">Data Insights</h2>', unsafe_allow_html=True)
        
        # Load sample data for visualization
        @st.cache_data
        def load_sample_data():
            df = pd.read_csv('WineQT.csv')
            if 'Id' in df.columns:
                df = df.drop('Id', axis=1)
            return df
        
        try:
            wine_df = load_sample_data()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Quality distribution
                st.markdown("#### Quality Score Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                quality_counts = wine_df['quality'].value_counts().sort_index()
                ax.bar(quality_counts.index, quality_counts.values, color='steelblue')
                ax.set_xlabel('Quality Score')
                ax.set_ylabel('Number of Wines')
                ax.set_xticks(range(3, 9))
                st.pyplot(fig)
            
            with col2:
                # Correlation with alcohol
                st.markdown("#### Alcohol vs Quality")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(wine_df['alcohol'], wine_df['quality'], alpha=0.5)
                ax.set_xlabel('Alcohol (%)')
                ax.set_ylabel('Quality Score')
                ax.set_title('Higher alcohol generally means better quality')
                st.pyplot(fig)
            
            # Show correlation heatmap
            st.markdown("#### Feature Correlation")
            if st.checkbox("Show correlation heatmap"):
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = wine_df.corr()
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                          center=0, square=True, ax=ax)
                st.pyplot(fig)
                
        except Exception as e:
            st.info("Upload WineQT.csv to see data visualizations")
    
    with tab4:
        st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🍷 Wine Quality Prediction System
        
        **How it works:**
        1. The model analyzes 11 chemical properties of wine
        2. It adds 3 engineered features based on wine chemistry
        3. A Random Forest model predicts quality (3-8 scale)
        4. Results are presented with interpretations
        
        **Model Performance:**
        - Mean Absolute Error: ~0.45 points
        - 67% of predictions within ±0.5 points
        - R² Score: ~0.45 (explains 45% of variance)
        
        **Data Source:**
        - WineQT dataset from Kaggle
        - 1,158 wine samples
        - 11 chemical measurements per wine
        
        **Technical Details:**
        - Model: Random Forest Regressor
        - Framework: Scikit-learn
        - Interface: Streamlit
        - Features: 11 original + 3 engineered
        
        **Limitations:**
        - Predicts based on chemistry only
        - Doesn't consider grape variety or region
        - Quality scores have subjective elements
        """)
        
        # Add download link for sample data
        st.markdown("---")
        st.markdown("### 📥 Get Started")
        st.markdown("""
        1. Download [WineQT.csv](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
        2. Place it in the same folder as this app
        3. Run the app with: `streamlit run wine_quality_app.py`
        """)

if __name__ == "__main__":
    main()