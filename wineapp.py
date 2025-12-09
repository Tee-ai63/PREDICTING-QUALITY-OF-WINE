import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents GUI warning in Streamlit Cloud
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Initialize session state
if 'wine_features' not in st.session_state:
    st.session_state.wine_features = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11,
        "total sulfur dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }

# Title with wine glass only
st.title("🍷 Wine Quality Prediction System")
st.markdown("""
Predict wine quality scores (3-8) based on chemical properties.
Upload data or use the interactive form below.
""")

# Load data function with better error handling for deployment
@st.cache_data
def load_data():
    try:
        # Try multiple possible file paths for different deployment scenarios
        possible_paths = ["WineQT.csv", "./WineQT.csv", "data/WineQT.csv"]
        
        for path in possible_paths:
            if os.path.exists(path):
                data = pd.read_csv(path)
                if "Id" in data.columns:
                    data = data.drop("Id", axis=1)
                return data
        
        # If file not found, show warning and create sample data
        st.warning("WineQT.csv not found. Using sample data for demonstration.")
        # Create sample data for demo purposes
        return pd.DataFrame({
            'fixed acidity': [7.4, 7.8, 7.8],
            'volatile acidity': [0.7, 0.88, 0.76],
            'citric acid': [0.0, 0.0, 0.04],
            'residual sugar': [1.9, 2.6, 2.3],
            'chlorides': [0.076, 0.098, 0.092],
            'free sulfur dioxide': [11, 25, 15],
            'total sulfur dioxide': [34, 67, 54],
            'density': [0.9978, 0.9968, 0.9970],
            'pH': [3.51, 3.2, 3.26],
            'sulphates': [0.56, 0.68, 0.65],
            'alcohol': [9.4, 9.8, 9.8],
            'quality': [5, 5, 5]
        })
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load model function with deployment-friendly error handling
@st.cache_resource
def load_model():
    try:
        # Check if model files exist
        if not os.path.exists("wine_model.pkl"):
            st.error("wine_model.pkl not found. Please upload it to the app directory.")
            return None, None
        
        if not os.path.exists("scaler.pkl"):
            st.error("scaler.pkl not found. Please upload it to the app directory.")
            return None, None
        
        # Use context manager for proper file handling
        with open("wine_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            
        return model, scaler
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Feature engineering function
def engineer_features(X_array, feature_names, is_dataframe=False):
    if is_dataframe:
        X_df = X_array.copy()
    else:
        X_df = pd.DataFrame(X_array, columns=feature_names)
    
    # Add engineered features
    X_df["acid_balance"] = X_df["fixed acidity"] / (X_df["pH"] + 1e-6)
    X_df["alcohol_acidity_ratio"] = X_df["alcohol"] / (X_df["fixed acidity"] + 1e-6)
    X_df["free_so2_ratio"] = X_df["free sulfur dioxide"] / (X_df["total sulfur dioxide"] + 1e-6)
    
    return X_df

# Cached heatmap for better performance
@st.cache_data(ttl=3600)
def get_correlation_heatmap(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    return fig

# Sidebar navigation
st.sidebar.header("Navigation")
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Home", "Exploratory Data Analysis", "Single Wine Prediction", "Batch Prediction", "Model Information"]
)

# Pre-load model for prediction pages
if option in ["Single Wine Prediction", "Batch Prediction"]:
    model, scaler = load_model()

# Main app logic
if option == "Home":
    st.header("Welcome to Wine Quality Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Using URL image for reliable deployment
        st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=400", 
                 caption="Wine Chemistry Analysis")
    
    with col2:
        st.markdown("""
        ### About This Application
        
        This machine learning model predicts wine quality scores 
        based on 11 chemical properties:
        
        - **Fixed Acidity** - **Volatile Acidity** - **Citric Acid**
        - **Residual Sugar** - **Chlorides** - **Free Sulfur Dioxide**
        - **Total Sulfur Dioxide** - **Density** - **pH**
        - **Sulphates** - **Alcohol**
        
        ### How to Use:
        1. **Exploratory Data Analysis**: Explore the dataset and correlations
        2. **Single Wine Prediction**: Predict quality for individual wines
        3. **Batch Prediction**: Upload CSV for multiple predictions
        4. **Model Information**: View model performance metrics
        
        ### Model Performance:
        - **Mean Absolute Error**: 0.42 (average prediction error)
        - **R-squared**: 0.45 (explains 45% of variance)
        - **Practical Accuracy**: 67% within ±0.5 points
        """)
    
    # Quick stats
    data = load_data()
    if data is not None:
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Wines", len(data))
        with col2:
            st.metric("Features", len(data.columns) - 1)
        with col3:
            st.metric("Average Quality", f"{data['quality'].mean():.1f}/10")
        with col4:
            st.metric("Quality Range", f"{data['quality'].min()}-{data['quality'].max()}")

elif option == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    data = load_data()
    if data is None:
        st.stop()
    
    # Show data preview
    st.subheader("Dataset Preview")
    st.dataframe(data.head())
    
    # Quality distribution
    st.subheader("Quality Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    data['quality'].value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel('Quality Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Wine Quality Scores')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation heatmap (cached for performance)
    st.subheader("Correlation Heatmap")
    heatmap_fig = get_correlation_heatmap(data)
    st.pyplot(heatmap_fig)
    
    # Feature correlations with quality
    st.subheader("Top Correlations with Quality")
    corr_with_quality = data.corr()['quality'].sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Positive Impact on Quality:**")
        for feature, corr in corr_with_quality[1:4].items():
            st.write(f"- {feature}: {corr:.3f}")
    
    with col2:
        st.write("**Negative Impact on Quality:**")
        for feature, corr in corr_with_quality[-3:].items():
            st.write(f"- {feature}: {corr:.3f}")

elif option == "Single Wine Prediction":
    st.header("Single Wine Quality Prediction")
    
    if model is None:
        st.warning("Please upload wine_model.pkl and scaler.pkl files to use predictions.")
        st.stop()
    
    st.subheader("Enter Wine Chemical Properties")
    
    # Create input form with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        fixed_acidity = st.slider("Fixed Acidity (g/L)", 4.0, 16.0, 7.4, 0.1)
        volatile_acidity = st.slider("Volatile Acidity (g/L)", 0.1, 1.6, 0.7, 0.01)
        citric_acid = st.slider("Citric Acid (g/L)", 0.0, 1.0, 0.0, 0.01)
        residual_sugar = st.slider("Residual Sugar (g/L)", 0.9, 15.5, 1.9, 0.1)
        chlorides = st.slider("Chlorides (g/L)", 0.01, 0.61, 0.076, 0.001)
    
    with col2:
        free_so2 = st.slider("Free Sulfur Dioxide (mg/L)", 1, 72, 11, 1)
        total_so2 = st.slider("Total Sulfur Dioxide (mg/L)", 6, 289, 34, 1)
        density = st.slider("Density (g/cm³)", 0.99, 1.04, 0.9978, 0.0001)
        pH = st.slider("pH", 2.7, 4.0, 3.51, 0.01)
        sulphates = st.slider("Sulphates (g/L)", 0.33, 2.0, 0.56, 0.01)
        alcohol = st.slider("Alcohol (%)", 8.4, 14.9, 9.4, 0.1)
    
    # Create wine dictionary
    wine_features = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_so2,
        "total sulfur dioxide": total_so2,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }
    
    # Update session state
    st.session_state.wine_features = wine_features
    
    # Predict button
    if st.button("Predict Quality", type="primary"):
        with st.spinner("Calculating prediction..."):
            # Prepare input
            wine_df = pd.DataFrame([wine_features])
            wine_scaled = scaler.transform(wine_df)
            
            # Add engineered features
            wine_eng = engineer_features(wine_scaled, wine_features.keys())
            
            # Predict
            prediction = model.predict(wine_eng.values)[0]
            prediction = max(3, min(8, prediction))
        
        # Display results
        st.success("Prediction Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Result")
            st.metric("Predicted Quality Score", f"{prediction:.2f}/10")
            
            # Quality interpretation
            if prediction < 4:
                quality_text = "Poor"
                color = "red"
            elif prediction < 5:
                quality_text = "Below Average"
                color = "orange"
            elif prediction < 6:
                quality_text = "Average"
                color = "yellow"
            elif prediction < 7:
                quality_text = "Good"
                color = "lightgreen"
            else:
                quality_text = "Excellent"
                color = "green"
            
            st.markdown(f"**Quality Category**: <span style='color:{color}'>{quality_text}</span>", 
                       unsafe_allow_html=True)
        
        with col2:
            # Quality meter visualization
            st.subheader("Quality Scale")
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh([0], [prediction], color='steelblue', height=0.5)
            ax.set_xlim([3, 8])
            ax.set_xlabel('Quality Score (3-8)')
            ax.set_yticks([])
            ax.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Average (5)')
            ax.axvline(x=7, color='green', linestyle='--', alpha=0.5, label='Good (7)')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        
        # Recommendations based on prediction
        st.subheader("Recommendations")
        if prediction < 6:
            st.warning("""
            **Suggestions for improvement:**
            - Increase alcohol content (aim for 12-13%)
            - Reduce volatile acidity (keep below 0.6 g/L)
            - Adjust pH to optimal range (3.3-3.5)
            - Ensure proper sulphates level (0.5-0.7 g/L)
            """)
        else:
            st.info("""
            **Good quality wine! Maintain these levels:**
            - Alcohol: 12-13.5%
            - Volatile acidity: < 0.6 g/L
            - pH: 3.3-3.5
            - Sulphates: 0.5-0.7 g/L
            - Citric acid: > 0.2 g/L for freshness
            """)

elif option == "Batch Prediction":
    st.header("Batch Wine Quality Prediction")
    
    st.markdown("""
    Upload a CSV file with wine chemical data. The file should include these columns:
    `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`,
    `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if model is None:
            st.warning("Please upload wine_model.pkl and scaler.pkl files to use predictions.")
            st.stop()
        
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"File loaded successfully! {len(batch_data)} wines found.")
            
            # Check required columns
            required_cols = [
                "fixed acidity", "volatile acidity", "citric acid", 
                "residual sugar", "chlorides", "free sulfur dioxide",
                "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
            ]
            
            missing_cols = [col for col in required_cols if col not in batch_data.columns]
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                # Predict button
                if st.button("Predict All Wines", type="primary"):
                    predictions = []
                    with st.spinner(f"Predicting quality for {len(batch_data)} wines..."):
                        # Scale features
                        batch_scaled = scaler.transform(batch_data[required_cols])
                        
                        # Add engineered features
                        batch_eng = engineer_features(batch_scaled, required_cols)
                        
                        # Predict
                        predictions = model.predict(batch_eng.values)
                        predictions = np.clip(predictions, 3, 8)
                    
                    # Add predictions to dataframe
                    results_df = batch_data.copy()
                    results_df["predicted_quality"] = predictions
                    results_df["quality_category"] = results_df["predicted_quality"].apply(
                        lambda x: "Poor" if x < 4 else 
                                 "Below Average" if x < 5 else 
                                 "Average" if x < 6 else 
                                 "Good" if x < 7 else "Excellent"
                    )
                    
                    # Display results
                    st.success("All predictions complete!")
                    
                    # Show sample
                    st.subheader("Prediction Results (First 10 rows)")
                    st.dataframe(results_df.head(10))
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Quality", f"{results_df['predicted_quality'].mean():.2f}")
                    with col2:
                        st.metric("Best Quality", f"{results_df['predicted_quality'].max():.1f}")
                    with col3:
                        st.metric("Worst Quality", f"{results_df['predicted_quality'].min():.1f}")
                    with col4:
                        excellent = (results_df['predicted_quality'] >= 7).sum()
                        st.metric("Excellent Wines", excellent)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="wine_predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

elif option == "Model Information":
    st.header("Model Information")
    
    data = load_data()
    if data is None:
        st.stop()
    
    st.subheader("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error", "0.42")
        st.caption("Average prediction error in points")
    with col2:
        st.metric("R-squared Score", "0.45")
        st.caption("Variance explained by model")
    with col3:
        st.metric("Within ±0.5 points", "67%")
        st.caption("Practical accuracy")
    
    st.subheader("Best Performing Algorithm")
    st.info("Random Forest Regressor was selected as the best model.")
    
    st.subheader("Dataset Information")
    st.write(f"**Total Samples**: {len(data)} wines")
    st.write(f"**Training Samples**: {int(len(data) * 0.8)} (80%)")
    st.write(f"**Test Samples**: {int(len(data) * 0.2)} (20%)")
    st.write(f"**Features Used**: 11 original + 3 engineered = 14 total")
    
    st.subheader("Key Features for Quality Prediction")
    st.markdown("""
    The model identified these as most important:
    
    1. **Alcohol Content** (+0.48 correlation) - Most important predictor
    2. **Volatile Acidity** (-0.39 correlation) - Main quality reducer
    3. **Sulphates** (+0.25 correlation) - Preservation quality
    4. **Citric Acid** (+0.23 correlation) - Freshness indicator
    5. **Acid Balance** (engineered) - Winemaking expertise
    
    **Note**: These align with wine expert knowledge, validating the model.
    """)

# Deployment guide in sidebar
with st.sidebar.expander("Deployment Checklist"):
    st.markdown("""
    **Files needed for full functionality:**
    
    - wine_model.pkl - Trained model  
    - scaler.pkl - Feature scaler  
    - WineQT.csv - Dataset (optional)
    
    **Deployed to Streamlit Cloud?**
    1. Push all files to GitHub
    2. Connect repo at share.streamlit.io
    3. Set main file to: wineapp.py
    4. Deploy!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>Wine Quality Prediction System</b> | Built with Streamlit | Random Forest Model</p>
    <p style='font-size: 0.8em; color: #666;'>
        For deployment issues, ensure all .pkl files are in the root directory
    </p>
</div>
""", unsafe_allow_html=True)