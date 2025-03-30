# Information aboutimport streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# Page setup
st.set_page_config(
    page_title="ML Model Trainer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1e7dd;
        color: #0f5132;
    }
    .info-box {
        padding: 0.75rem;
        border-radius: 0.5rem;
        background-color: #cff4fc;
        color: #055160;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 4px 4px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# App title with custom styling
st.markdown('<p class="main-header">Machine Learning Model Trainer</p>', unsafe_allow_html=True)
st.markdown("""
Train and evaluate machine learning models on various datasets with ease.
This tool allows you to experiment with different models and visualize their performance.
""")

# Add a separator
st.markdown("---")

# Initialize session state for persistence between reruns
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None

# Sidebar for data selection
with st.sidebar:
    st.header("1. Select Data")
    
    data_source = st.radio(
        "Choose a data source:",
        ["Seaborn Dataset", "Upload CSV"]
    )
    
    if data_source == "Seaborn Dataset":
        datasets = ["iris", "tips", "diamonds", "titanic", "penguins"]
        selected_dataset = st.selectbox("Select dataset:", datasets)
        
        if st.button("Load Dataset"):
            try:
                st.session_state.data = sns.load_dataset(selected_dataset)
                st.success(f"Loaded {selected_dataset} dataset!")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    else:
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
        if uploaded_file is not None:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success("CSV file uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

# Main content
if st.session_state.data is not None:
    # Show data preview
    st.markdown('<p class="sub-header">Dataset Preview</p>', unsafe_allow_html=True)
    
    # Display dataset summary statistics alongside the preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(st.session_state.data.head(), use_container_width=True)
    
    with col2:
        st.write("**Dataset Summary:**")
        st.write(f"• Rows: {st.session_state.data.shape[0]:,}")
        st.write(f"• Columns: {st.session_state.data.shape[1]}")
        
        # Count numeric and categorical columns
        num_cols = len(st.session_state.data.select_dtypes(include=['int64', 'float64']).columns)
        cat_cols = len(st.session_state.data.select_dtypes(include=['object', 'category']).columns)
        
        st.write(f"• Numeric columns: {num_cols}")
        st.write(f"• Categorical columns: {cat_cols}")
        
        # Check for missing values
        missing = st.session_state.data.isna().sum().sum()
        if missing > 0:
            st.write(f"• Missing values: {missing:,} ⚠️")
        else:
            st.write("• Missing values: 0 ✓")
    
    # Add a separator
    st.markdown("---")
    
    # Two columns layout for model configuration
    st.markdown('<p class="sub-header">Model Configuration</p>', unsafe_allow_html=True)
    
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        with st.container(border=True):
            st.markdown("**2. Features & Target Selection**")
            
            # Select target variable
            target_variable = st.selectbox(
                "Select target variable:",
                options=st.session_state.data.columns
            )
            
            # Identify numeric and categorical columns
            numeric_cols = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = st.session_state.data.select_dtypes(include=['object', 'category']).columns
            
            # Problem type
            problem_type = st.radio(
                "Select problem type:",
                ["Regression", "Classification"],
                help="Choose regression for predicting continuous values, classification for categories."
            )
            
            # Feature selection
            st.write("**Select Features:**")
            
            # Numeric features
            if len(numeric_cols) > 0:
                selected_numeric = st.multiselect(
                    "Numeric features:",
                    [col for col in numeric_cols if col != target_variable],
                    default=[col for col in numeric_cols if col != target_variable][:min(3, len(numeric_cols))]
                )
            else:
                selected_numeric = []
                st.info("No numeric features available in this dataset.")
            
            # Categorical features
            if len(categorical_cols) > 0:
                selected_categorical = st.multiselect(
                    "Categorical features:",
                    [col for col in categorical_cols if col != target_variable],
                    default=[col for col in categorical_cols if col != target_variable][:min(2, len(categorical_cols))]
                )
            else:
                selected_categorical = []
                st.info("No categorical features available in this dataset.")
    
    with model_col2:
        with st.container(border=True):
            st.markdown("**3. Model Parameters**")
            
            # Test size with a nice gauge-like visual
            test_size = st.slider(
                "Test set size (%):",
                min_value=10,
                max_value=50,
                value=20,
                help="Percentage of data to use for testing the model"
            )
            
            st.write(f"Train-Test Split: {100-test_size}% / {test_size}%")
            
            # Model selection
            if problem_type == "Regression":
                model_choice = st.selectbox(
                    "Select regression model:",
                    ["Linear Regression", "Random Forest Regressor"]
                )
                
                if model_choice == "Random Forest Regressor":
                    n_estimators = st.slider("Number of trees:", 10, 200, 100)
                    max_depth = st.slider("Maximum depth:", 1, 30, 10)
            else:
                model_choice = st.selectbox(
                    "Select classification model:",
                    ["Logistic Regression", "Random Forest Classifier"]
                )
                
                if model_choice == "Random Forest Classifier":
                    n_estimators = st.slider("Number of trees:", 10, 200, 100)
                    max_depth = st.slider("Maximum depth:", 1, 30, 10)
                else:
                    C = st.slider("Regularization strength:", 0.1, 10.0, 1.0, 
                                help="Lower values increase regularization")
            
            # Train button with color and icon
            train_button = st.button("Train Model 🚀", type="primary", use_container_width=True)
            if not selected_numeric and not selected_categorical:
                st.error("Please select at least one feature")
            else:
                try:
                    # Prepare data
                    features = selected_numeric + selected_categorical
                    X = st.session_state.data[features].copy()
                    y = st.session_state.data[target_variable].copy()
                    
                    # Check for missing values
                    if X.isna().any().any():
                        st.warning("Your data contains missing values. Attempting to handle them...")
                        # Show which columns have missing values
                        missing_cols = X.columns[X.isna().any()].tolist()
                        st.write(f"Columns with missing values: {', '.join(missing_cols)}")
                        
                        # Simple imputation strategy - fill with mean/mode
                        for col in X.columns:
                            if X[col].dtype.kind in 'ifc':  # numeric columns
                                X[col] = X[col].fillna(X[col].mean())
                            else:  # categorical columns
                                X[col] = X[col].fillna(X[col].mode()[0])
                    
                    if y.isna().any():
                        st.error("The target variable contains missing values. Please select a different target variable or clean your data.")
                        st.stop()
                    
                    # Handle categorical features with one-hot encoding
                    if selected_categorical:
                        X = pd.get_dummies(X, columns=selected_categorical, drop_first=True)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )
                    
                    # Save for later evaluation
                    st.session_state.y_test = y_test
                    
                    # Select and train model
                    with st.spinner("Training model..."):
                        if model_choice == "Linear Regression":
                            model = LinearRegression()
                            st.session_state.model_type = "regression"
                        elif model_choice == "Random Forest Regressor":
                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=42
                            )
                            st.session_state.model_type = "regression"
                        elif model_choice == "Logistic Regression":
                            model = LogisticRegression(
                                C=C,
                                max_iter=1000,
                                random_state=42
                            )
                            st.session_state.model_type = "classification"
                        else:  # Random Forest Classifier
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=42
                            )
                            st.session_state.model_type = "classification"
                        
                        # Fit and predict
                        model.fit(X_train, y_train)
                        st.session_state.model = model
                        st.session_state.predictions = model.predict(X_test)
                        st.session_state.X_test = X_test
                        st.session_state.feature_names = X.columns
                    
                    st.success("Model trained successfully!")
                
                except Exception as e:
                    error_msg = str(e)
                    
                    # Handle common error cases with user-friendly messages
                    if "Input y contains NaN" in error_msg:
                        st.error("The target variable contains missing values. Please select a different target variable or clean your data.")
                    elif "Input X contains NaN" in error_msg:
                        st.error("Your data contains missing values. Try using the 'Data Preprocessing' options to handle them.")
                    elif "shape of X and y differ" in error_msg:
                        st.error("The number of samples in features and target are not the same. This might be caused by filtering or missing values.")
                    elif "Solver newton-cg" in error_msg and "multinomial" in error_msg:
                        st.error("For multi-class classification with Logistic Regression, try using more data or simplifying your model parameters.")
                    elif "Expected 2D array, got 1D array instead" in error_msg:
                        st.error("There's an issue with the shape of your data. Try selecting different features.")
                    elif "invalid value encountered" in error_msg:
                        st.error("Mathematical error encountered, possibly due to extreme values or divisions by zero.")
                    else:
                        # For unexpected errors, show the original message
                        st.error(f"Error during training: {error_msg}")
    
    # Display results (only if model has been trained)
    if st.session_state.model is not None:
        st.markdown('<p class="sub-header">Model Results</p>', unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "📈 Visualizations", "🔍 Feature Importance"])
        
                with tab1:
            st.subheader("Model Performance")
            
            # Calculate metrics based on model type
            if st.session_state.model_type == "regression":
                mse = mean_squared_error(st.session_state.y_test, st.session_state.predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(st.session_state.y_test, st.session_state.predictions)
                
                # Display metrics in a nice format with icons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Mean Squared Error**")
                    st.markdown(f"<h1 style='text-align: center; color: #1e88e5;'>{mse:.4f}</h1>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Root MSE**")
                    st.markdown(f"<h1 style='text-align: center; color: #1e88e5;'>{rmse:.4f}</h1>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("**R² Score**")
                    st.markdown(f"<h1 style='text-align: center; color: #1e88e5;'>{r2:.4f}</h1>", unsafe_allow_html=True)
                
                # Add explanation of metrics
                with st.expander("What do these metrics mean?"):
                    st.markdown("""
                    - **Mean Squared Error (MSE)**: Average of squared differences between predicted and actual values. Lower is better.
                    - **Root Mean Squared Error (RMSE)**: Square root of MSE, in the same units as the target variable. Lower is better.
                    - **R² Score**: Proportion of variance explained by the model, ranges from 0 to 1. Higher is better.
                    """)
            
            else:  # classification
                accuracy = accuracy_score(st.session_state.y_test, st.session_state.predictions)
                
                # Display metrics with a gauge-like visualization
                st.markdown("**Accuracy**")
                st.markdown(f"<h1 style='text-align: center; color: #1e88e5;'>{accuracy:.4f}</h1>", unsafe_allow_html=True)
                st.progress(float(accuracy))
                
                # Confusion matrix as a nicely formatted heatmap
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state.y_test, st.session_state.predictions)
                
                # Create a dataframe for better display
                if hasattr(st.session_state.y_test, 'unique'):
                    labels = sorted(st.session_state.y_test.unique())
                    cm_df = pd.DataFrame(cm, 
                                        index=[f'Actual: {i}' for i in labels],
                                        columns=[f'Predicted: {i}' for i in labels])
                else:
                    cm_df = pd.DataFrame(cm)
                
                st.dataframe(cm_df, use_container_width=True)
                
                # Add explanation
                with st.expander("What do these metrics mean?"):
                    st.markdown("""
                    - **Accuracy**: Proportion of correct predictions out of all predictions. Higher is better.
                    - **Confusion Matrix**: Shows correct and incorrect predictions for each class. 
                      The diagonal represents correct predictions, while off-diagonal values are errors.
                    """)
            
            # Add model export functionality
            st.subheader("Export Model")
            
            import pickle
            import base64
            
            # Create a download button for the model
            def get_model_download_link(model, model_name):
                """Generate a download link for the trained model"""
                # Serialize the model
                model_pkl = pickle.dumps(model)
                # Encode to base64
                b64 = base64.b64encode(model_pkl).decode()
                # Generate download link
                href = f'<a href="data:file/pickle;base64,{b64}" download="{model_name}.pkl">Download {model_name} Model</a>'
                return href
            
            model_name = model_choice.replace(" ", "_").lower()
            st.markdown(get_model_download_link(st.session_state.model, model_name), unsafe_allow_html=True)
            
            # Add instructions for using the downloaded model
            with st.expander("How to use the downloaded model"):
                st.markdown("""
                To use this model in your own Python code:
                ```python
                import pickle
                
                # Load the model
                with open('model_filename.pkl', 'rb') as f:
                    model = pickle.load(f)
                    
                # Make predictions
                predictions = model.predict(X_new)
                ```
                
                Make sure your new data has the same features and preprocessing as the training data.
                """)
                
        
        with tab2:
            st.subheader("Visualizations")
            
            if st.session_state.model_type == "regression":
                # Create two columns for visualizations
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Actual vs Predicted
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(st.session_state.y_test, st.session_state.predictions, 
                              alpha=0.6, edgecolor='k', color='#1e88e5')
                    ax.plot(
                        [st.session_state.y_test.min(), st.session_state.y_test.max()],
                        [st.session_state.y_test.min(), st.session_state.y_test.max()],
                        'r--', linewidth=2
                    )
                    ax.set_xlabel("Actual Values", fontsize=12)
                    ax.set_ylabel("Predicted Values", fontsize=12)
                    ax.set_title("Actual vs Predicted Values", fontsize=14)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with viz_col2:
                    # Residual distribution
                    residuals = st.session_state.y_test - st.session_state.predictions
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(residuals, kde=True, ax=ax, color='#1e88e5')
                    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
                    ax.set_xlabel("Residual Value", fontsize=12)
                    ax.set_ylabel("Frequency", fontsize=12)
                    ax.set_title("Residual Distribution", fontsize=14)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            else:  # classification
                # Confusion Matrix heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(st.session_state.y_test, st.session_state.predictions)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
                ax.set_xlabel("Predicted Labels", fontsize=12)
                ax.set_ylabel("True Labels", fontsize=12)
                ax.set_title("Confusion Matrix", fontsize=14)
                st.pyplot(fig)
        
        with tab3:
            st.subheader("Feature Importance")
            
            # Extract feature importance if available
            if hasattr(st.session_state.model, 'feature_importances_'):
                # For tree-based models
                feature_imp = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': st.session_state.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Display as table and bar chart
                st.dataframe(feature_imp, use_container_width=True, hide_index=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax, palette='viridis')
                ax.set_title("Feature Importance", fontsize=14)
                ax.set_xlabel("Importance", fontsize=12)
                ax.grid(True, axis='x', alpha=0.3)
                st.pyplot(fig)
                
            elif hasattr(st.session_state.model, 'coef_'):
                # For linear models
                if st.session_state.model_type == "regression":
                    coeffs = st.session_state.model.coef_
                else:
                    # Handle multi-class case
                    coeffs = st.session_state.model.coef_[0] if st.session_state.model.coef_.ndim > 1 else st.session_state.model.coef_
                
                feature_imp = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Coefficient': coeffs
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                st.dataframe(feature_imp, use_container_width=True, hide_index=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                # Use a diverging color palette for coefficients
                bars = sns.barplot(x='Coefficient', y='Feature', data=feature_imp, ax=ax)
                
                # Color positive and negative bars differently
                for i, bar in enumerate(bars.patches):
                    if feature_imp.iloc[i]['Coefficient'] < 0:
                        bar.set_facecolor('#ff7f7f')  # Light red for negative
                    else:
                        bar.set_facecolor('#7fbf7f')  # Light green for positive
                
                ax.set_title("Feature Coefficients", fontsize=14)
                ax.set_xlabel("Coefficient Value", fontsize=12)
                ax.axvline(x=0, color='gray', linestyle='--')
                ax.grid(True, axis='x', alpha=0.3)
                st.pyplot(fig)
            
            else:
                st.info("Feature importance not available for this model type.")
else:
    # Welcome screen when no data is loaded
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("# 🤖 Welcome to ML Model Trainer")
    st.markdown("</div>", unsafe_allow_html=True)

    # Centered content with better styling
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 20px;">
            <p style="font-size: 18px;">👈 Please select a dataset from the sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Information about the app with better styling
        st.markdown("""
        ## What can you do with this app?
        
        This interactive tool allows you to experiment with machine learning models:
        
        1. **Choose your data** - Select from built-in datasets or upload your own CSV file
        2. **Configure your model** - Select features and target variables
        3. **Train the model** - Choose between regression and classification models
        4. **Analyze results** - Visualize model performance and explore predictions
        
        ## Available Models
        """)
        
        # Models displayed in a more visual way
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.markdown("""
            <div style="padding: 15px; border-radius: 5px; background-color: #e3f2fd; height: 150px;">
                <h3>Regression Models</h3>
                <ul>
                    <li>Linear Regression</li>
                    <li>Random Forest Regressor</li>
                </ul>
                <p><em>For predicting continuous values</em></p>
            </div>
            """, unsafe_allow_html=True)
            
        with model_col2:
            st.markdown("""
            <div style="padding: 15px; border-radius: 5px; background-color: #e8f5e9; height: 150px;">
                <h3>Classification Models</h3>
                <ul>
                    <li>Logistic Regression</li>
                    <li>Random Forest Classifier</li>
                </ul>
                <p><em>For predicting categories</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Example workflow with visual design
        st.markdown("## Sample Workflow")
        
        st.markdown("""
        <div style="display: flex; justify-content: space-between; text-align: center; margin-top: 20px;">
            <div style="flex: 1; padding: 10px;">
                <div style="background-color: #f1f8e9; border-radius: 50%; width: 50px; height: 50px; display: flex; justify-content: center; align-items: center; margin: 0 auto;">1</div>
                <p>Select Data</p>
            </div>
            <div style="flex: 1; padding: 10px;">
                <div style="background-color: #fff8e1; border-radius: 50%; width: 50px; height: 50px; display: flex; justify-content: center; align-items: center; margin: 0 auto;">2</div>
                <p>Configure Model</p>
            </div>
            <div style="flex: 1; padding: 10px;">
                <div style="background-color: #e0f7fa; border-radius: 50%; width: 50px; height: 50px; display: flex; justify-content: center; align-items: center; margin: 0 auto;">3</div>
                <p>Train Model</p>
            </div>
            <div style="flex: 1; padding: 10px;">
                <div style="background-color: #f3e5f5; border-radius: 50%; width: 50px; height: 50px; display: flex; justify-content: center; align-items: center; margin: 0 auto;">4</div>
                <p>Analyze Results</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
