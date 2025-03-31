import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# Page setup
st.set_page_config(page_title="ML Model Trainer", layout="wide")

# App title
st.title("Machine Learning Model Trainer")
st.write("Train and evaluate ML models on various datasets")

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
    st.header("Dataset Preview")
    st.dataframe(st.session_state.data.head())
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("2. Configure Model")
        
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
            ["Regression", "Classification"]
        )
        
        # Feature selection
        st.subheader("Select Features")
        
        # Numeric features
        selected_numeric = st.multiselect(
            "Numeric features:",
            [col for col in numeric_cols if col != target_variable],
            default=[col for col in numeric_cols if col != target_variable][:min(3, len(numeric_cols))]
        )
        
        # Categorical features
        selected_categorical = st.multiselect(
            "Categorical features:",
            [col for col in categorical_cols if col != target_variable],
            default=[col for col in categorical_cols if col != target_variable][:min(2, len(categorical_cols))]
        )
    
    with col2:
        st.header("3. Model Parameters")
        
        # Test size
        test_size = st.slider(
            "Test size (%):",
            min_value=10,
            max_value=50,
            value=20
        )
        
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
                C = st.slider("Regularization strength:", 0.1, 10.0, 1.0)
        
        # Train button
        if st.button("Train Model"):
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
                        
                        # Store additional info for download
                        st.session_state.selected_numeric = selected_numeric
                        st.session_state.selected_categorical = selected_categorical
                        st.session_state.model_choice = model_choice
                    
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
        st.header("4. Model Results")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Visualizations", "Feature Importance"])
        
        with tab1:
            st.subheader("Model Performance")
            
            # Calculate metrics based on model type
            if st.session_state.model_type == "regression":
                mse = mean_squared_error(st.session_state.y_test, st.session_state.predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(st.session_state.y_test, st.session_state.predictions)
                
                # Display metrics in a nice format
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Squared Error", f"{mse:.4f}")
                col2.metric("Root MSE", f"{rmse:.4f}")
                col3.metric("R² Score", f"{r2:.4f}")
            
            else:  # classification
                accuracy = accuracy_score(st.session_state.y_test, st.session_state.predictions)
                
                # Display metrics
                st.metric("Accuracy", f"{accuracy:.4f}")
                
                # Confusion matrix as a dataframe
                cm = confusion_matrix(st.session_state.y_test, st.session_state.predictions)
                st.subheader("Confusion Matrix")
                st.dataframe(pd.DataFrame(cm))
        
        with tab2:
            st.subheader("Visualizations")
            
            if st.session_state.model_type == "regression":
                # Actual vs Predicted
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(st.session_state.y_test, st.session_state.predictions, alpha=0.5)
                ax.plot(
                    [st.session_state.y_test.min(), st.session_state.y_test.max()],
                    [st.session_state.y_test.min(), st.session_state.y_test.max()],
                    'r--'
                )
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted Values")
                st.pyplot(fig)
                
                # Residual distribution
                residuals = st.session_state.y_test - st.session_state.predictions
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(residuals, kde=True, ax=ax)
                ax.axvline(x=0, color='r', linestyle='--')
                ax.set_xlabel("Residual Value")
                ax.set_ylabel("Frequency")
                ax.set_title("Residual Distribution")
                st.pyplot(fig)
            
            else:  # classification
                # Confusion Matrix heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(st.session_state.y_test, st.session_state.predictions)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted Labels")
                ax.set_ylabel("True Labels")
                ax.set_title("Confusion Matrix")
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
                st.dataframe(feature_imp)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
                ax.set_title("Feature Importance")
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
                
                st.dataframe(feature_imp)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Coefficient', y='Feature', data=feature_imp, ax=ax)
                ax.set_title("Feature Coefficients")
                st.pyplot(fig)
            
            else:
                st.info("Feature importance not available for this model type.")
                
        # Model Download Section (NEW)
        st.divider()
        st.subheader("5. Download Trained Model")
        
        st.write("You can download the trained model for use in your own applications.")
        
        col1, col2 = st.columns(2)
        
        with col1:
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
                "Model": st.session_state.model_choice
            }
            
            # Convert to JSON for download
            feature_info_str = str(feature_info)
            
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
            # - Include all required features in the same order
            
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
else:
    # Instructions when no data is loaded
    st.info("👈 Please select a dataset from the sidebar to get started.")
    
    # Information about the app
    st.markdown("""
    ## About this app
    
    This application allows you to:
    
    1. Select from sample datasets or upload your own CSV
    2. Choose between regression and classification models
    3. Configure model parameters
    4. View performance metrics and visualizations
    5. Download trained models for external use
    
    ### Available Models
    
    **Regression:**
    - Linear Regression
    - Random Forest Regressor
    
    **Classification:**
    - Logistic Regression
    - Random Forest Classifier
    """)
