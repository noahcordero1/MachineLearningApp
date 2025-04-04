import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime

# Page setup with custom theme and wider layout
st.set_page_config(
    page_title="ML Model Trainer Pro", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance UI
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px; 
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 0px 20px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #cee4fd;
        font-weight: bold;
    }
    div.stButton > button:hover {border-color: #9AD3FF;}
    div.stButton > button:active {border-color: #9AD3FF;}
    div[data-testid="stForm"] {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border: 1px solid #eee;
    }
    .stAlert {border-radius: 5px;}
    [data-testid="collapsedControl"] {
        display: flex;
        justify-content: space-between;
        border-radius: 5px;
    }
    [data-testid="stMetricValue"] {font-size: 2rem !important;}
    .css-j7qwjs {border-radius: 5px;}
    .css-1544g2n {padding: 0rem 1rem 1rem;}
</style>
""", unsafe_allow_html=True)

# App title
st.title("🤖 ML Model Trainer Pro")
st.markdown("Train, evaluate, and deploy machine learning models with ease")

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
if 'training_time' not in st.session_state:
    st.session_state.training_time = None
if 'preprocessing_options' not in st.session_state:
    st.session_state.preprocessing_options = {}
if 'model_history' not in st.session_state:
    st.session_state.model_history = []
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'correlation_data' not in st.session_state:
    st.session_state.correlation_data = None

# Define callback functions
def reset_app():
    """Reset all session state variables"""
    # Keep model_history but clear everything else
    model_history = st.session_state.get('model_history', [])
    
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Restore model history
    st.session_state.model_history = model_history
    
    # Re-initialize required session state variables with defaults
    st.session_state.data = None
    st.session_state.model = None
    st.session_state.predictions = None
    st.session_state.y_test = None
    st.session_state.model_type = None
    st.session_state.X_test = None
    st.session_state.feature_names = None
    st.session_state.correlation_data = None
    st.session_state.training_time = None
    st.session_state.preprocessing_options = {}

def add_to_history():
    """Save current model to history"""
    if st.session_state.model is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract important metrics
        if st.session_state.model_type == "regression":
            mse = mean_squared_error(st.session_state.y_test, st.session_state.predictions)
            r2 = r2_score(st.session_state.y_test, st.session_state.predictions)
            metric = f"R² = {r2:.4f}, MSE = {mse:.4f}"
        else:
            accuracy = accuracy_score(st.session_state.y_test, st.session_state.predictions)
            metric = f"Accuracy = {accuracy:.4f}"
        
        # Create entry
        model_entry = {
            "timestamp": timestamp,
            "model_name": st.session_state.model_choice,
            "metrics": metric,
            "training_time": st.session_state.training_time,
            "model": st.session_state.model,
            "predictions": st.session_state.predictions,
            "y_test": st.session_state.y_test,
            "model_type": st.session_state.model_type,
            "feature_names": st.session_state.feature_names
        }
        
        st.session_state.model_history.append(model_entry)

# Sidebar for data selection and high-level actions
with st.sidebar:
    st.header("🛠️ Toolbox")
    
    # Data source selection
    st.subheader("1. Select Data")
    
    data_source = st.radio(
        "Choose a data source:",
        ["Seaborn Dataset", "Upload CSV"],
        key="data_source"
    )
    
    if data_source == "Seaborn Dataset":
        # Get all available seaborn datasets
        datasets = ["iris", "tips", "diamonds", "titanic", "penguins", "planets", "mpg", "car_crashes"]
        selected_dataset = st.selectbox("Select dataset:", datasets)
        
        # Dataset details expander
        with st.expander("Dataset Details"):
            dataset_info = {
                "iris": "Classic iris flower dataset (150 samples, 4 features, 3 classes)",
                "tips": "Restaurant tipping data (244 samples, customer info & meal prices)",
                "diamonds": "Diamond prices and characteristics (53,940 samples)",
                "titanic": "Titanic passenger survival data (891 samples)",
                "penguins": "Measurements of Antarctic penguins (344 samples)",
                "planets": "Data on planets discovered around other stars",
                "mpg": "Auto fuel efficiency data (398 samples)",
                "car_crashes": "Car crash statistics by US state"
            }
            st.info(dataset_info.get(selected_dataset, "Dataset information not available"))
        
        if st.button("Load Dataset", type="primary"):
            with st.spinner("Loading dataset..."):
                try:
                    st.session_state.data = sns.load_dataset(selected_dataset)
                    st.session_state.correlation_data = None  # Reset correlation data
                    st.success(f"Loaded {selected_dataset} dataset with {len(st.session_state.data)} samples!")
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
    else:
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
        if uploaded_file is not None:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.session_state.correlation_data = None  # Reset correlation data
                st.success(f"CSV file uploaded successfully! ({len(st.session_state.data)} rows)")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    # App reset button
    st.divider()
    if st.button("🔄 Reset App"):
        reset_app()
        st.rerun()
    
    # Model history section
    st.divider()
    st.subheader("Model History")
    if len(st.session_state.model_history) > 0:
        for i, entry in enumerate(reversed(st.session_state.model_history)):
            with st.expander(f"{entry['model_name']} - {entry['timestamp']}"):
                st.write(f"**Metrics:** {entry['metrics']}")
                st.write(f"**Training Time:** {entry['training_time']:.2f}s")
                
                # Load this model
                if st.button("Restore This Model", key=f"restore_{i}"):
                    st.session_state.model = entry['model']
                    st.session_state.predictions = entry['predictions']
                    st.session_state.y_test = entry['y_test']
                    st.session_state.model_type = entry['model_type']
                    st.session_state.feature_names = entry['feature_names']
                    st.session_state.model_choice = entry['model_name']
                    st.experimental_rerun()
    else:
        st.info("No models trained yet")

# Main content
if st.session_state.data is not None:
    # Show data info in a collapsible section
    with st.expander("Dataset Overview", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.dataframe(st.session_state.data.head(), height=250)
        
        with col2:
            st.write("**Dataset Shape:**", st.session_state.data.shape)
            st.write("**Data Types:**")
            dtypes_df = pd.DataFrame({'Type': st.session_state.data.dtypes})
            st.dataframe(dtypes_df, height=200)
        
        with col3:
            st.write("**Missing Values:**")
            missing_df = pd.DataFrame({'Missing': st.session_state.data.isnull().sum()})
            missing_df['%'] = (missing_df['Missing'] / len(st.session_state.data) * 100).round(2)
            st.dataframe(missing_df, height=200)
    
    # Data exploration and correlation matrix
    with st.expander("Data Exploration", expanded=False):
        if st.button("Generate Correlation Matrix"):
            # Compute correlation only for numeric columns
            numeric_data = st.session_state.data.select_dtypes(include=['int64', 'float64'])
            if not numeric_data.empty:
                st.session_state.correlation_data = numeric_data.corr()
                
        if st.session_state.correlation_data is not None:
            st.subheader("Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(st.session_state.correlation_data, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
            st.pyplot(fig)
    
    # Two columns layout for configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("2. Configure Model")
        
        # Target variable selection
        target_variable = st.selectbox(
            "Select target variable:",
            options=st.session_state.data.columns
        )
        
        # Identify numeric and categorical columns
        numeric_cols = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = st.session_state.data.select_dtypes(include=['object', 'category']).columns
        
        # Determine problem type based on target variable
        if target_variable in numeric_cols:
            default_problem = "Regression"
        else:
            default_problem = "Classification"
        
        # Problem type selection with default suggestion
        problem_type = st.radio(
            "Select problem type:",
            ["Regression", "Classification"],
            index=0 if default_problem == "Regression" else 1,
            help="Regression predicts continuous values, Classification predicts categories."
        )
        
        # Feature selection with guidance
        st.subheader("Select Features")
        
        # Numeric features with selectability check
        available_numeric = [col for col in numeric_cols if col != target_variable]
        if available_numeric:
            recommended_numeric = min(3, len(available_numeric))
            selected_numeric = st.multiselect(
                "Numeric features:",
                available_numeric,
                default=available_numeric[:recommended_numeric],
                help="Select numeric features to include in your model"
            )
        else:
            st.info("No numeric features available")
            selected_numeric = []
        
        # Categorical features 
        available_categorical = [col for col in categorical_cols if col != target_variable]
        if available_categorical:
            recommended_categorical = min(2, len(available_categorical))
            selected_categorical = st.multiselect(
                "Categorical features:",
                available_categorical,
                default=available_categorical[:recommended_categorical],
                help="Select categorical features to include in your model"
            )
        else:
            selected_categorical = []
        
        # Data preprocessing options
        st.subheader("Data Preprocessing")
        
        # Missing value handling
        st.session_state.preprocessing_options['handle_missing'] = st.checkbox(
            "Handle missing values automatically", 
            value=True,
            help="Impute missing numeric values with mean and categorical with mode"
        )
        
        # Feature scaling
        st.session_state.preprocessing_options['scaling'] = st.checkbox(
            "Apply feature scaling",
            value=False,
            help="Standardize numeric features to have zero mean and unit variance"
        )
    
    with col2:
        st.header("3. Model Parameters")
        
        # Test size
        test_size = st.slider(
            "Test size (%):",
            min_value=10,
            max_value=50,
            value=20,
            help="Percentage of data to use for testing"
        )
        
        # Random state for reproducibility
        random_state = st.number_input(
            "Random seed:",
            min_value=0,
            value=42,
            help="Random seed for reproducible results"
        )
        
        # Model selection based on problem type
        if problem_type == "Regression":
            model_choice = st.selectbox(
                "Select regression model:",
                ["Linear Regression", "Random Forest Regressor"]
            )
            
            # Model-specific parameters
            if model_choice == "Random Forest Regressor":
                st.subheader("Model Hyperparameters")
                n_estimators = st.slider("Number of trees:", 10, 200, 100)
                max_depth = st.slider("Maximum depth:", 1, 30, 10)
                min_samples_split = st.slider("Minimum samples to split:", 2, 10, 2)
                min_samples_leaf = st.slider("Minimum samples per leaf:", 1, 10, 1)
        else:
            model_choice = st.selectbox(
                "Select classification model:",
                ["Logistic Regression", "Random Forest Classifier"]
            )
            
            # Model-specific parameters
            if model_choice == "Random Forest Classifier":
                st.subheader("Model Hyperparameters")
                n_estimators = st.slider("Number of trees:", 10, 200, 100)
                max_depth = st.slider("Maximum depth:", 1, 30, 10)
                min_samples_split = st.slider("Minimum samples to split:", 2, 10, 2)
                class_weight = st.radio("Class weights:", ["None", "Balanced"], index=0)
                class_weight = None if class_weight == "None" else "balanced"
            else:
                st.subheader("Model Hyperparameters")
                C = st.slider("Regularization strength (C):", 0.1, 10.0, 1.0, 0.1)
                penalty = st.radio("Penalty type:", ["l2", "l1", "elasticnet", "none"], index=0)
                max_iter = st.slider("Maximum iterations:", 100, 2000, 1000, 100)
                class_weight = st.radio("Class weights:", ["None", "Balanced"], index=0)
                class_weight = None if class_weight == "None" else "balanced"
        
        # Train button with visual enhancements
        st.markdown("")  # Add space
        train_col1, train_col2 = st.columns([3, 1])
        
        with train_col1:
            train_button = st.button(
                "🚀 Train Model",
                type="primary",
                use_container_width=True,
                help="Click to train the model with selected parameters"
            )
        
        with train_col2:
            add_history = st.checkbox("Save to history", value=True)
    
    # Training process
    if train_button:
        if not selected_numeric and not selected_categorical:
            st.error("Please select at least one feature")
        else:
            try:
                # Start timer
                start_time = time.time()
                
                # Prepare data
                features = selected_numeric + selected_categorical
                X = st.session_state.data[features].copy()
                y = st.session_state.data[target_variable].copy()
                
                # Data preprocessing
                with st.spinner("Preprocessing data..."):
                    # Handle missing values
                    if st.session_state.preprocessing_options['handle_missing'] and X.isna().any().any():
                        st.info("Handling missing values...")
                        # Handle missing numeric values
                        for col in selected_numeric:
                            if X[col].isna().any():
                                X[col] = X[col].fillna(X[col].mean())
                        
                        # Handle missing categorical values
                        for col in selected_categorical:
                            if X[col].isna().any():
                                X[col] = X[col].fillna(X[col].mode()[0])
                    
                    # Handle missing target values
                    if y.isna().any():
                        st.error("The target variable contains missing values. Please select a different target variable or clean your data.")
                        st.stop()
                    
                    # Handle categorical features with one-hot encoding
                    if selected_categorical:
                        X = pd.get_dummies(X, columns=selected_categorical, drop_first=True)
                    
                    # Apply scaling if selected
                    if st.session_state.preprocessing_options['scaling'] and selected_numeric:
                        scaler = StandardScaler()
                        X[selected_numeric] = scaler.fit_transform(X[selected_numeric])
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=random_state
                    )
                    
                    # Save for later evaluation
                    st.session_state.y_test = y_test
                    st.session_state.X_test = X_test
                    st.session_state.feature_names = X.columns
                    st.session_state.model_choice = model_choice
                
                # Select and train model
                with st.spinner("Training model..."):
                    progress_bar = st.progress(0)
                    
                    # Initialize model based on selection
                    if model_choice == "Linear Regression":
                        model = LinearRegression()
                        st.session_state.model_type = "regression"
                    elif model_choice == "Random Forest Regressor":
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=random_state
                        )
                        st.session_state.model_type = "regression"
                    elif model_choice == "Logistic Regression":
                        model = LogisticRegression(
                            C=C,
                            penalty=penalty if penalty != "none" else None,
                            max_iter=max_iter,
                            class_weight=class_weight,
                            random_state=random_state
                        )
                        st.session_state.model_type = "classification"
                    else:  # Random Forest Classifier
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            class_weight=class_weight,
                            random_state=random_state
                        )
                        st.session_state.model_type = "classification"
                    
                    # Update progress
                    progress_bar.progress(30)
                    
                    # Fit and predict
                    model.fit(X_train, y_train)
                    progress_bar.progress(70)
                    
                    st.session_state.model = model
                    st.session_state.predictions = model.predict(X_test)
                    
                    # Update progress
                    progress_bar.progress(100)
                    
                    # Record training time
                    st.session_state.training_time = time.time() - start_time
                    
                    # Add to history if checkbox selected
                    if add_history:
                        add_to_history()
                
                st.success(f"Model trained successfully in {st.session_state.training_time:.2f} seconds!")
            
            except Exception as e:
                error_msg = str(e)
                
                # Handle common error cases with user-friendly messages
                if "Input y contains NaN" in error_msg:
                    st.error("The target variable contains missing values. Please select a different target variable or clean your data.")
                elif "Input X contains NaN" in error_msg:
                    st.error("Your data contains missing values. Enable the 'Handle missing values automatically' option to fix this.")
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
        st.header("4. Model Results")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Metrics", "📈 Visualizations", "🔍 Feature Importance", "🔮 Make Predictions"])
        
        with tab1:
            st.subheader("Model Performance")
            
            # Calculate metrics based on model type
            if st.session_state.model_type == "regression":
                mse = mean_squared_error(st.session_state.y_test, st.session_state.predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(st.session_state.y_test, st.session_state.predictions)
                
                # Display metrics in a nice format with columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Squared Error", f"{mse:.4f}")
                col2.metric("Root MSE", f"{rmse:.4f}")
                col3.metric("R² Score", f"{r2:.4f}")
                col4.metric("Training Time", f"{st.session_state.training_time:.2f}s")
                
                st.write("### Interpretation")
                interp_col1, interp_col2 = st.columns(2)
                
                with interp_col1:
                    st.info(f"""
                    **R² Score**: {r2:.4f}
                    - Values range from 0 to 1 (higher is better)
                    - Represents how well the model explains the variance in the target
                    - Values >0.7 generally indicate a good fit
                    """)
                
                with interp_col2:
                    st.info(f"""
                    **RMSE**: {rmse:.4f}
                    - In the same units as the target variable
                    - Lower values indicate better predictions
                    - Useful for understanding prediction error magnitude
                    """)
            
            else:  # classification
                accuracy = accuracy_score(st.session_state.y_test, st.session_state.predictions)
                
                # Display metrics with columns
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("Training Time", f"{st.session_state.training_time:.2f}s")
                
                # Generate classification report
                report = classification_report(st.session_state.y_test, st.session_state.predictions, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                st.write("### Classification Report")
                st.dataframe(report_df.style.format("{:.4f}"))
                
                # Confusion matrix as a dataframe
                st.write("### Confusion Matrix")
                cm = confusion_matrix(st.session_state.y_test, st.session_state.predictions)
                # Get unique classes
                unique_classes = np.unique(np.concatenate([st.session_state.y_test, st.session_state.predictions]))
                cm_df = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)
                cm_df.index.name = 'Actual'
                cm_df.columns.name = 'Predicted'
                st.dataframe(cm_df.style.background_gradient(cmap='Blues'))
        
        with tab2:
            st.subheader("Visualizations")
            
            if st.session_state.model_type == "regression":
                # Actual vs Predicted Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(st.session_state.y_test, st.session_state.predictions, alpha=0.5)
                ax.plot(
                    [st.session_state.y_test.min(), st.session_state.y_test.max()],
                    [st.session_state.y_test.min(), st.session_state.y_test.max()],
                    'r--'
                )
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted Values")
                # Add 45-degree line
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Residual Plot
                residuals = st.session_state.y_test - st.session_state.predictions
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Residual scatterplot
                axes[0].scatter(st.session_state.predictions, residuals, alpha=0.5)
                axes[0].axhline(y=0, color='r', linestyle='--')
                axes[0].set_xlabel("Predicted Values")
                axes[0].set_ylabel("Residuals")
                axes[0].set_title("Residual Plot")
                axes[0].grid(True, linestyle='--', alpha=0.7)
                
                # Residual distribution
                sns.histplot(residuals, kde=True, ax=axes[1])
                axes[1].axvline(x=0, color='r', linestyle='--')
                axes[1].set_xlabel("Residual Value")
                axes[1].set_ylabel("Frequency")
                axes[1].set_title("Residual Distribution")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Interpretation
                st.info("""
                **Interpreting Residual Plots:**
                - Points should be randomly scattered around the zero line
                - No obvious patterns should be visible
                - Histogram should be roughly bell-shaped and centered at zero
                - Patterns may indicate model assumptions are violated
                """)
            
            else:  # classification
                # Confusion Matrix heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                cm = confusion_matrix(st.session_state.y_test, st.session_state.predictions)
                
                # Get unique classes for better labeling
                unique_classes = np.unique(np.concatenate([st.session_state.y_test, st.session_state.predictions]))
                
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    ax=ax,
                    xticklabels=unique_classes,
                    yticklabels=unique_classes
                )
                ax.set_xlabel("Predicted Labels")
                ax.set_ylabel("True Labels")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
                
                # Interpretation
                st.info("""
                **Interpreting the Confusion Matrix:**
                - Diagonal elements represent correct predictions
                - Off-diagonal elements are errors
                - Rows represent the true classes
                - Columns represent the predicted classes
                """)
        
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
                st.dataframe(feature_imp)
                
                fig, ax = plt.subplots(figsize=(12, min(10, max(6, len(feature_imp) * 0.3))))
                sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
                ax.set_title("Feature Importance")
                ax.grid(True, axis='x', linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Top 3 features
                top_features = feature_imp.head(3)['Feature'].tolist()
                st.success(f"Top 3 most important features: {', '.join(top_features)}")
                
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
                
                st.dataframe(feature_imp)
                
                fig, ax = plt.subplots(figsize=(12, min(10, max(6, len(feature_imp) * 0.3))))
                # Use a diverging colormap to highlight positive and negative coefficients
                colors = ['red' if c < 0 else 'blue' for c in feature_imp['Coefficient']]
                sns.barplot(x='Coefficient', y='Feature', data=feature_imp, palette=colors, ax=ax)
                ax.set_title("Feature Coefficients")
                ax.axvline(x=0, color='gray', linestyle='--')
                ax.grid(True, axis='x', linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Interpretation of coefficients
                st.info("""
                **Interpreting Coefficients:**
                - The magnitude (absolute value) indicates feature importance
                - Sign (+ or -) shows the direction of influence:
                  - Positive: Increases target value
                  - Negative: Decreases target value
                - For classification, higher absolute values indicate stronger influence on class assignment
                """)
            
            else:
                st.info("Feature importance not available for this model type.")
                
        with tab4:
            st.subheader("Make Predictions")
            st.write("Use this tool to predict outcomes using your trained model")
            
            # Create form for inputting prediction values
            st.write("### Enter feature values")
            
            prediction_inputs = {}
            
            # Create input fields based on the features used in training
            if not hasattr(st.session_state, 'feature_names') or st.session_state.feature_names is None:
                st.warning("No features available. Please train a model first.")
            else:
                # Get original feature names (before one-hot encoding)
                original_features = selected_numeric + selected_categorical
                
                # Create two columns for inputs
                col1, col2 = st.columns(2)
                
                # Create numeric inputs
                with col1:
                    st.write("**Numeric Features**")
                    for feature in selected_numeric:
                        # Get min, max and mean from training data
                        feature_min = float(st.session_state.data[feature].min())
                        feature_max = float(st.session_state.data[feature].max())
                        feature_mean = float(st.session_state.data[feature].mean())
                        
                        # Create slider with these values
                        prediction_inputs[feature] = st.slider(
                            f"{feature}:",
                            min_value=feature_min,
                            max_value=feature_max,
                            value=feature_mean,
                            step=(feature_max - feature_min) / 100
                        )
                
                # Create categorical inputs
                with col2:
                    st.write("**Categorical Features**")
                    for feature in selected_categorical:
                        # Get unique values for this feature
                        unique_values = st.session_state.data[feature].dropna().unique().tolist()
                        
                        # Create selectbox
                        prediction_inputs[feature] = st.selectbox(
                            f"{feature}:",
                            options=unique_values,
                            index=0
                        )
                
                # Make prediction button
                if st.button("Make Prediction", type="primary"):
                    try:
                        # Create a dataframe with the input values
                        input_df = pd.DataFrame([prediction_inputs])
                        
                        # Apply the same preprocessing steps as during training
                        if selected_categorical:
                            input_df = pd.get_dummies(input_df, columns=selected_categorical, drop_first=True)
                        
                        # Add missing columns that were in training data but not in input data
                        for col in st.session_state.feature_names:
                            if col not in input_df.columns:
                                input_df[col] = 0
                        
                        # Ensure column order matches training data
                        input_df = input_df[st.session_state.feature_names]
                        
                        # Apply scaling if it was applied during training
                        if st.session_state.preprocessing_options.get('scaling', False) and selected_numeric:
                            # We would need the scaler from training, but we don't have it
                            # This is a simplified approach for demo purposes
                            st.warning("Scaling was applied during training, but can't be precisely replicated for predictions.")
                        
                        # Make prediction
                        prediction = st.session_state.model.predict(input_df)
                        
                        # Display the prediction
                        st.success("### Prediction Results")
                        
                        if st.session_state.model_type == "regression":
                            st.metric("Predicted Value", f"{prediction[0]:.4f}")
                        else:
                            # For classification, show the predicted class
                            st.metric("Predicted Class", prediction[0])
                            
                            # If the model has predict_proba method, show probabilities
                            if hasattr(st.session_state.model, 'predict_proba'):
                                proba = st.session_state.model.predict_proba(input_df)
                                proba_df = pd.DataFrame(
                                    proba[0], 
                                    index=st.session_state.model.classes_, 
                                    columns=['Probability']
                                ).sort_values('Probability', ascending=False)
                                
                                st.write("### Prediction Probabilities")
                                
                                # Create a bar chart of probabilities
                                fig, ax = plt.subplots(figsize=(8, min(8, max(3, len(proba_df) * 0.5))))
                                sns.barplot(x='Probability', y=proba_df.index, data=proba_df.reset_index(), ax=ax)
                                ax.set_xlim(0, 1)
                                ax.set_title("Class Probabilities")
                                ax.grid(True, axis='x', linestyle='--', alpha=0.7)
                                st.pyplot(fig)
                                
                                st.dataframe(proba_df.style.format("{:.4f}").bar(subset=['Probability'], color='#5fba7d'))
                    
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.info("This error might be due to missing features or different processing steps between training and prediction.")
            
    # Model Download Section
    if st.session_state.model is not None:
        st.divider()
        st.header("5. Download Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Download the trained model for use in your own applications")
            
            # Serialize the model
            model_pickle = pickle.dumps(st.session_state.model)
            
            # Get model type for filename
            model_name = st.session_state.model_choice.replace(" ", "_").lower()
            
            # Create download button
            st.download_button(
                label="Download Model",
                data=model_pickle,
                file_name=f"{model_name}_model.pkl",
                mime="application/octet-stream",
                help="Download the trained model as a pickle file"
            )
            
            st.warning("Note: Only open pickle files from trusted sources.")
        
        with col2:
            # Create model info for users
            feature_info = {
                "Features": list(st.session_state.feature_names),
                "Target": target_variable,
                "Model Type": st.session_state.model_type,
                "Model": st.session_state.model_choice,
                "Training Time": f"{st.session_state.training_time:.2f} seconds",
                "Date Trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Convert to formatted text for download
            feature_info_str = "\n".join([f"{k}: {v}" for k, v in feature_info.items()])
            
            # Download button for model info
            st.download_button(
                label="Download Model Info",
                data=feature_info_str,
                file_name=f"{model_name}_info.txt",
                mime="text/plain",
                help="Information about features used in this model"
            )
            
            st.info("Download this information to know how to use your model correctly.")
        
        # Usage instructions
        with st.expander("How to use the downloaded model"):
            st.markdown("""
            ### Python code to load and use this model
            
            ```python
            import pickle
            import pandas as pd
            
            # Load the model
            with open('your_downloaded_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Prepare your new data with the same features
            # Make sure to apply the same preprocessing steps:
            # - Handle missing values
            # - One-hot encode categorical features
            # Include all required features in the same order
            
            # Example:
            X_new = pd.DataFrame({
                'feature1': [...],
                'feature2': [...],
                # Include all features used during training
            })
            
            # Make predictions
            predictions = model.predict(X_new)
            ```
            
            **Important:** Your new data must have the same structure as the data used for training.
            """)
            
            # Export sample prediction code
            if st.session_state.feature_names is not None:
                sample_code = f"""
# Sample code for using this specific model
import pickle
import pandas as pd
import numpy as np

# Load the model
with open('{model_name}_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a sample input with the exact features used
sample_data = {{
"""
                # Add sample values for each feature
                for feature in st.session_state.feature_names:
                    sample_code += f"    '{feature}': [0],  # Replace with your value\n"
                
                sample_code += """}}

# Create DataFrame
X_new = pd.DataFrame(sample_data)

# Make prediction
prediction = model.predict(X_new)
print(f"Prediction: {prediction[0]}")
"""
                
                st.download_button(
                    label="Download Sample Usage Code",
                    data=sample_code,
                    file_name=f"{model_name}_sample_usage.py",
                    mime="text/plain"
                )

else:
    # Instructions when no data is loaded
    st.info("👈 Please select a dataset from the sidebar to get started.")
    
    # Information about the app
    st.markdown("""
    ## About this ML Model Trainer
    
    This application allows you to:
    
    1. **Select Data**: Choose from sample datasets or upload your own CSV
    2. **Configure Models**: Choose between regression and classification models
    3. **Tune Parameters**: Configure model hyperparameters for optimal performance
    4. **Visualize Results**: View performance metrics, predictions, and feature importance
    5. **Make Predictions**: Use your trained model to make predictions on new data
    6. **Download Models**: Save your trained models for external use
    
    ### Available Models
    
    **Regression:**
    - Linear Regression: Simple, interpretable model for linear relationships
    - Random Forest Regressor: Powerful ensemble model for complex patterns
    
    **Classification:**
    - Logistic Regression: Probabilistic classification with good interpretability
    - Random Forest Classifier: Ensemble learning model with high accuracy
    
    ### Data Preprocessing
    
    The app offers automatic handling of:
    - Missing values
    - Categorical features (one-hot encoding)
    - Feature scaling
    
    ### Usage Tips
    
    - Begin by selecting a dataset from the sidebar
    - For best results, choose features that are likely to influence the target variable
    - Keep track of different models with the history feature
    - Download your best model for use in other applications
    """)
    
    # Display sample images
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Example Regression Result")
        plt.figure(figsize=(8, 6))
        plt.scatter(np.random.normal(0, 1, 100), np.random.normal(0, 1, 100) + np.random.normal(0, 0.5, 100), alpha=0.6)
        plt.plot([-2, 2], [-2, 2], 'r--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Sample Regression Plot")
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(plt)
    
    with col2:
        st.markdown("### Example Feature Importance")
        plt.figure(figsize=(8, 6))
        features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
        importances = [0.35, 0.25, 0.20, 0.15, 0.05]
        plt.barh(features, importances)
        plt.xlabel("Importance")
        plt.title("Sample Feature Importance")
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        st.pyplot(plt)
