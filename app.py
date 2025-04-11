import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
import base64
import io  # for StringIO

# Additional packages for LLM Q&A
import PyPDF2
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Ensure TensorFlow logs are minimized
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set the page configuration immediately after imports
st.set_page_config(page_title="ML & AI Explorer", layout="wide")

# ------------------------------
# Custom CSS for additional styling
# ------------------------------
st.markdown(
    """
    <style>
    /* Example Dark Theme Colors */
    body {
        background-color: #1A1A1A !important; /* dark background */
        color: #F5F5F5 !important;           /* light text */
    }
    .block-container {
        background-color: #2E2E2E !important;
        border-radius: 8px;
        padding: 2rem;
        color: #F5F5F5 !important;
    }
    /* Force sidebar darker background */
    .css-1d391kg {
        background: linear-gradient(180deg, #333333, #1A1A1A);
    }
    /* Sidebar text color */
    .css-1lcbmhc {
        color: #F5F5F5 !important;
        font-size: 1.2rem;
    }
    /* Buttons */
    div.stButton > button {
        background-color: #444444 !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #666666 !important;
    }
    /* Tables / Dataframe text color fix */
    .css-12oz5g7, .css-17eq0hr, .css-1sbuyqj {
        color: #F5F5F5 !important;
        background-color: #2E2E2E !important;
    }
    /* Heading overrides */
    h1, h2, h3, h4, h5, h6 {
        color: #F5F5F5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.title("Machine Learning & AI Explorer Dashboard")

# Sidebar navigation for sub-tasks
task = st.sidebar.radio(
    "Select Task",
    ["Regression", "Clustering", "Neural Network", "LLM Q&A"]
)




# a) Regression Problem - Linear Regression 
if task == "Regression":
    st.header("📈 Regression Explorer")
    st.write(
        "Upload your regression dataset, select target and feature columns, then view model performance and make custom predictions."
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="regression")
    if st.button("Clear Uploaded Dataset"):
        st.experimental_rerun()

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(data.head())
        if st.checkbox("Show dataset summary"):
            st.write(data.describe())
        if st.checkbox("Show dataset info"):
            buf = io.StringIO()
            data.info(buf=buf)
            info_text = buf.getvalue()
            lines = info_text.splitlines()
            shape_line = [line.strip() for line in lines if "RangeIndex:" in line]
            shape_str = shape_line[0] if shape_line else ""
            mem_line = [line.strip() for line in lines if "memory usage:" in line.lower()]
            mem_str = mem_line[0] if mem_line else ""
            import re
            col_pattern = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\d+)\s+non-null\s+(\S+)')
            col_details = []
            parsing_cols = False
            for line in lines:
                if "#   Column" in line:
                    parsing_cols = True
                    continue
                if parsing_cols:
                    if line.strip() == "" or "memory usage:" in line.lower():
                        break
                    match = col_pattern.match(line)
                    if match:
                        idx, col_name, non_null, dtype_ = match.groups()
                        col_details.append({
                            "Index": idx,
                            "Column Name": col_name,
                            "Non-Null Count": non_null,
                            "DType": dtype_
                        })
            st.markdown("**Dataset Info**")
            st.markdown(f"**Shape:** {shape_str}")
            st.markdown(f"**Memory Usage:** {mem_str}")
            if col_details:
                df_info = pd.DataFrame(col_details)
                st.dataframe(df_info)
            else:
                st.text(info_text)
        columns = data.columns.tolist()
        target_column = st.selectbox("Select the target column", columns)
        candidate_features = [col for col in columns if col != target_column]
        feature_columns = st.multiselect(
            label="Select feature columns",
            options=candidate_features,
            help="Tip: after choosing a feature, press Enter to confirm it. Then choose the next feature."
        )
        if not feature_columns:
            st.warning("Please select at least one feature column.")
            st.stop()
        if target_column not in data.columns:
            st.error("The specified target column does not exist in the dataset.")
            st.stop()
        st.markdown("### Missing Value Handling")
        missing_strategy = st.radio(
            "How do you want to handle missing values?",
            ("Drop rows with missing data in selected columns", 
             "Fill missing numeric with mean & categorical with mode", "Do nothing")
        )
        subset_data = data[feature_columns + [target_column]]
        if missing_strategy == "Drop rows with missing data in selected columns":
            subset_data = subset_data.dropna(subset=feature_columns + [target_column])
            st.success("Dropped rows with missing values from selected columns.")
        elif missing_strategy == "Fill missing numeric with mean & categorical with mode":
            for col in subset_data.columns:
                if subset_data[col].dtype.kind in ('i', 'f'):
                    subset_data[col].fillna(subset_data[col].mean(), inplace=True)
                else:
                    mode_val = subset_data[col].mode()[0] if not subset_data[col].mode().empty else ""
                    subset_data[col].fillna(mode_val, inplace=True)
            st.success("Filled missing numeric values with column mean, and categorical with mode.")
        X = subset_data[feature_columns]
        y = subset_data[target_column]
        if not pd.api.types.is_numeric_dtype(y):
            st.warning(
                f"Target column '{target_column}' is not numeric. Attempting conversion..."
            )
            y = pd.to_numeric(y, errors='coerce')
            if y.isna().all():
                st.error(
                    f"Failed to convert '{target_column}' to numeric. Choose a valid numeric column."
                )
                st.stop()
            else:
                valid_mask = ~y.isna()
                X = X[valid_mask]
                y = y[valid_mask]
                st.success(f"Converted '{target_column}' to numeric where possible.")
        X = pd.get_dummies(X)
        st.markdown("### Train-Test Split")
        test_size = st.slider("Select test set size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        st.markdown("### Model Training")
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.subheader("Model Performance")
        st.write(f"📉 **Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"📊 **R² Score:** {r2:.2f}")
        st.subheader("Predictions vs Actual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7, label="Predicted vs Actual")
        ymin, ymax = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([ymin, ymax], [ymin, ymax], 'r--', lw=2, label="Ideal Fit")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)
        if len(feature_columns) == 1:
            st.subheader("Regression Line (Single Feature)")
            relevant_cols = [col for col in X.columns if col.startswith(feature_columns[0])]
            if len(relevant_cols) == 1:
                fig2, ax2 = plt.subplots()
                ax2.scatter(X_test[relevant_cols[0]], y_test, color='blue', label="Actual")
                ax2.plot(X_test[relevant_cols[0]], y_pred, color='red', linewidth=2, label="Predicted")
                ax2.set_xlabel(feature_columns[0])
                ax2.set_ylabel(target_column)
                ax2.set_title("Linear Regression Line")
                ax2.legend()
                st.pyplot(fig2)
            else:
                st.info("Selected feature expanded to multiple one-hot columns; regression line not shown.")
        st.subheader("🔍 Make a Custom Prediction")
        with st.expander("Enter custom input for prediction"):
            st.write("Fill in values for each selected feature:")
            num_cols = 3
            cols = st.columns(num_cols)
            custom_input = {}
            for i, feature in enumerate(feature_columns):
                col_idx = i % num_cols
                with cols[col_idx]:
                    if pd.api.types.is_numeric_dtype(subset_data[feature]):
                        default_val = float(subset_data[feature].mean())
                        custom_input[feature] = st.number_input(label=f"{feature}:", value=default_val)
                    else:
                        mode_val = subset_data[feature].mode()
                        default_str = str(mode_val[0]) if not mode_val.empty else ""
                        custom_input[feature] = st.text_input(label=f"{feature}:", value=default_str)
            if st.button("Predict"):
                custom_df = pd.DataFrame([custom_input])
                custom_df = pd.get_dummies(custom_df)
                custom_df = custom_df.reindex(columns=X.columns, fill_value=0)
                custom_pred = model.predict(custom_df)[0]
                st.success(f"**Predicted {target_column}:** {custom_pred:.2f}")













#############################################
# b) Clustering
#############################################
elif task == "Clustering":
    st.header("🔍 Interactive K-Means Clustering Explorer")
    
    # ------------------------------
    # Step 1: Data Upload & Preview
    # ------------------------------
    st.markdown("## Step 1: Data Upload & Preview")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="clustering")
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin clustering.")
        st.stop()
    
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------
    # Step 2: Feature Selection
    # ------------------------------
    st.markdown("## Step 2: Feature Selection")
    all_features = df.columns.tolist()
    selected_features = st.multiselect(
        label="Select features for clustering",
        options=all_features,
        help="Choose at least 2 features. After selecting, press Enter to confirm your choices."
    )
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features for clustering.")
        st.stop()
    
    # ------------------------------
    # Step 3: Missing Value Handling
    # ------------------------------
    st.markdown("## Step 3: Missing Value Handling")
    if st.checkbox("Drop rows with missing values in selected features"):
        df = df.dropna(subset=selected_features)
        st.success("Rows with missing values dropped.")
    
    # ------------------------------
    # Step 4: Cluster Settings
    # ------------------------------
    st.markdown("## Step 4: Cluster Settings")
    num_clusters = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)
    
    # ------------------------------
    # Step 5: Visualization Options
    # ------------------------------
    st.markdown("## Step 5: Visualization Options")
    # Set visualization option based on number of features selected.
    if len(selected_features) == 2:
        vis_option = "2D (No reduction)"
    elif len(selected_features) == 3:
        vis_option = st.radio(
            "Choose visualization mode:",
            options=["2D (using PCA)", "3D Interactive"],
            index=1,
            help="For 3 features, choose '3D Interactive' for a 3D view or '2D (using PCA)' for a 2D projection."
        )
    else:
        vis_option = st.radio(
            "Select Dimensionality Reduction Method for 2D Visualization:",
            options=["PCA", "t-SNE"],
            index=0,
            help="For more than 3 features, choose a reduction method for a 2D plot."
        )
    
    # ------------------------------
    # Step 6: Clustering & Centroid Calculation
    # ------------------------------
    st.markdown("## Step 6: Clustering & Centroid Calculation")
    # Prepare the data for clustering.
    X = df[selected_features]
    # Convert categorical features to numeric via one-hot encoding.
    X = pd.get_dummies(X)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)
    
    # Compute centroids in the one-hot encoded feature space.
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
    centroids.index.name = "Cluster"
    st.markdown("#### Cluster Centroids")
    st.dataframe(centroids.style.format("{:.2f}"))
    
    # Optionally compute mean summaries for the originally selected numeric features.
    numeric_cols = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        cluster_summary = df.groupby("Cluster")[numeric_cols].mean().reset_index()
        st.markdown("#### Cluster Mean Summary")
        st.dataframe(cluster_summary.style.format("{:.2f}"))
    else:
        st.info("No numeric features available for computing cluster summary.")
    
    # ------------------------------
    # Step 7: Cluster Visualization
    # ------------------------------
    st.markdown("## Step 7: Cluster Visualization")
    if vis_option == "2D (No reduction)":
        # For exactly 2 selected features.
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df[selected_features[0]], df[selected_features[1]], 
                   c=df["Cluster"], cmap="viridis", alpha=0.7, label="Data Points")
        ax.scatter(centroids[selected_features[0]], centroids[selected_features[1]], 
                   c="red", marker="X", s=200, label="Centroids")
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        ax.set_title("2D K-Means Clustering with Centroids")
        ax.legend()
        st.pyplot(fig)
    elif vis_option == "3D Interactive":
        # For exactly 3 features, use Plotly for interactive 3D visualization.
        fig_3d = px.scatter_3d(
            df,
            x=selected_features[0],
            y=selected_features[1],
            z=selected_features[2],
            color="Cluster",
            title="3D K-Means Clustering",
            opacity=0.7
        )
        fig_3d.add_scatter3d(
            x=centroids[selected_features[0]],
            y=centroids[selected_features[1]],
            z=centroids[selected_features[2]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="x"),
            name="Centroids"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    elif vis_option == "PCA":
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        df["PC1"], df["PC2"] = components[:, 0], components[:, 1]
        centroid_components = pca.transform(centroids)
        centroids["PC1"], centroids["PC2"] = centroid_components[:, 0], centroid_components[:, 1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["PC1"], df["PC2"], c=df["Cluster"], cmap="viridis", alpha=0.7, label="Data Points")
        ax.scatter(centroids["PC1"], centroids["PC2"], c="red", marker="X", s=200, label="Centroids")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("PCA-based 2D Clustering with Centroids")
        ax.legend()
        st.pyplot(fig)
    elif vis_option == "t-SNE":
        tsne_perplexity = min(30, len(X) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
        tsne_components = tsne.fit_transform(X)
        df["tSNE1"], df["tSNE2"] = tsne_components[:, 0], tsne_components[:, 1]
        tsne_centroids = df.groupby("Cluster")[["tSNE1", "tSNE2"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["tSNE1"], df["tSNE2"], c=df["Cluster"], cmap="viridis", alpha=0.7, label="Data Points")
        ax.scatter(tsne_centroids["tSNE1"], tsne_centroids["tSNE2"], c="red", marker="X", s=200, label="Centroids")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_title("t-SNE based 2D Clustering with Centroids")
        ax.legend()
        st.pyplot(fig)
    
    # ------------------------------
    # Step 8: Download Clustered Dataset
    # ------------------------------
    st.markdown("## Step 8: Download Clustered Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="clustered_data.csv",
        mime="text/csv"
    )


#############################################
# c) Neural Network Classifier
#############################################
elif task == "Neural Network":
    st.header("🧠 Neural Network Classifier")
    st.markdown("""
    **Follow these steps to train your neural network classifier:**

    **Step 1:** Upload your CSV dataset.  
    **Step 2:** Select the target column and feature columns.  
    **Step 3:** Optionally drop rows with missing values.  
    **Step 4:** Data Preprocessing – Numeric features are standardized and categorical features are one-hot encoded.  
    **Step 5:** Configure hyperparameters and split the dataset.  
    **Step 6:** Click **Train Model** to train with live progress feedback.  
    **Step 7:** After training, view metrics and make custom predictions.
    """)

    # Initialize session state for training
    if "nn_trained" not in st.session_state:
        st.session_state.nn_trained = False

    #############################################
    # Step 1: Data Upload & Preview
    #############################################
    st.markdown("## Step 1: Data Upload & Preview")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="nn")
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin.")
        st.stop()
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    #############################################
    # Step 2: Target & Feature Selection
    #############################################
    st.markdown("## Step 2: Target & Feature Selection")
    all_columns = df.columns.tolist()
    target_column = st.selectbox("Select the target column", all_columns)
    feature_columns = st.multiselect(
        "Select feature columns (all columns except target)",
        options=[col for col in all_columns if col != target_column],
        help="After selecting a feature, press Enter to confirm."
    )
    if not feature_columns:
        st.warning("Please select at least one feature column.")
        st.stop()

    #############################################
    # Step 3: Missing Value Handling
    #############################################
    st.markdown("## Step 3: Missing Value Handling")
    if st.checkbox("Drop rows with missing values"):
        df = df.dropna(subset=feature_columns + [target_column])
        st.success("Dropped rows with missing values.")

    #############################################
    # Step 4: Data Preprocessing & Target Binning
    #############################################
    st.markdown("## Step 4: Data Preprocessing & Target Binning")
    # Extract target.
    y = df[target_column].values
    unique_target_vals = df[target_column].nunique()
    if unique_target_vals > 50:
        st.warning(f"Target column '{target_column}' has {unique_target_vals} unique values (appears continuous).")
        if st.checkbox("Apply automatic binning (10 bins) to target"):
            df[target_column] = pd.qcut(df[target_column], q=10, labels=False)
            st.info("Target has been binned into 10 classes.")
            y = df[target_column].values
        else:
            st.error("Target appears continuous. For classification, please apply binning or use the Regression module.")
            st.stop()
    # Encode non-numeric target values.
    if not np.issubdtype(y.dtype, np.number):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        le = None
    # Validate class labels are 0-indexed.
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    if not np.array_equal(np.sort(unique_classes), np.arange(num_classes)):
        st.error(f"Class labels must be integers from 0 to {num_classes - 1}. Found: {unique_classes}")
        st.stop()

    # Separate features into numeric and categorical.
    numeric_features = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_features = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(df[col])]
    st.write("**Numeric features:**", numeric_features)
    st.write("**Categorical features:**", categorical_features)

    # Process numeric features.
    from sklearn.preprocessing import StandardScaler
    if numeric_features:
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(df[numeric_features].astype(float))
    else:
        X_numeric = np.empty((df.shape[0], 0))
    # Process categorical features.
    if categorical_features:
        X_categorical_df = pd.get_dummies(df[categorical_features], drop_first=False)
        X_categorical = X_categorical_df.values
        cat_dummy_cols = X_categorical_df.columns.tolist()
    else:
        X_categorical = np.empty((df.shape[0], 0))
        cat_dummy_cols = []
    # Combine features.
    X = np.hstack([X_numeric, X_categorical])
    st.write("Final shape of feature matrix after preprocessing:", X.shape)

    #############################################
    # Step 5: Train-Test Split & Hyperparameter Settings
    #############################################
    st.markdown("## Step 5: Train-Test Split & Hyperparameter Settings")
    from sklearn.model_selection import train_test_split
    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    st.subheader("Configure Hyperparameters")
    epochs = st.slider("Epochs", 1, 100, 10)
    batch_size = st.slider("Batch Size", 8, 128, 32)
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0,
                                    value=0.001, step=0.0001, format="%.4f")

    #############################################
    # Step 6: Model Building & Training
    #############################################
    if st.button("Train Model"):
        st.markdown("## Step 6: Model Building & Training")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, name="Adam"),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        st.subheader("Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Epoch {epoch + 1} / {epochs}")

        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[StreamlitCallback()]
            )
            st.session_state.nn_trained = True
            st.session_state.nn_model = model
            st.session_state.nn_history = history
            st.session_state.nn_scaler = scaler
            st.session_state.nn_le = le
            st.session_state.nn_numeric_features = numeric_features
            st.session_state.nn_cat_dummy_cols = cat_dummy_cols
            st.session_state.nn_df = df  # store original DataFrame for prediction alignment
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"❌ Error during training: {e}")
            st.stop()

    if not st.session_state.get("nn_trained", False):
        st.info("Click the **Train Model** button to start training.")
    else:
        #############################################
        # Step 7: Training & Validation Metrics
        #############################################
        st.markdown("## Step 7: Training & Validation Metrics")
        history = st.session_state.nn_history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['loss'], label="Train Loss")
        ax1.plot(history.history['val_loss'], label="Validation Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax2.plot(history.history['accuracy'], label="Train Accuracy")
        ax2.plot(history.history['val_accuracy'], label="Validation Accuracy")
        ax2.set_title("Accuracy")
        ax2.legend()
        st.pyplot(fig)
        st.success(f"✅ Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")

        #############################################
        # Step 8: Custom Prediction
        #############################################
        st.markdown("## Step 8: Custom Prediction")
        st.write("Enter custom values for each original feature to get a prediction:")
        
        custom_numeric = {}
        custom_categorical = {}
        num_cols_ui = 3
        cols_ui = st.columns(num_cols_ui)
        # Create inputs for numeric features.
        for i, feat in enumerate(st.session_state.nn_numeric_features):
            with cols_ui[i % num_cols_ui]:
                default_val = float(st.session_state.nn_df[feat].mean())
                custom_numeric[feat] = st.number_input(f"Enter value for {feat}", value=default_val)
        # Create inputs for categorical features.
        for i, feat in enumerate(df.columns.difference(st.session_state.nn_numeric_features + [target_column])):
            # This includes categorical features that were one-hot encoded.
            if feat in st.session_state.nn_cat_dummy_cols:
                with cols_ui[i % num_cols_ui]:
                    default_val = st.session_state.nn_df[feat].mode()[0] if not st.session_state.nn_df[feat].mode().empty else ""
                    custom_categorical[feat] = st.text_input(f"Enter value for {feat}", value=str(default_val))
        
        if st.button("Predict"):
            try:
                # Process numeric inputs.
                if st.session_state.nn_numeric_features:
                    custom_num_df = pd.DataFrame([custom_numeric])
                    custom_num_scaled = st.session_state.nn_scaler.transform(custom_num_df.astype(float))
                else:
                    custom_num_scaled = np.empty((1, 0))
                # Process categorical inputs.
                if st.session_state.nn_cat_dummy_cols:
                    custom_cat_df = pd.DataFrame([custom_categorical])
                    custom_cat_dummies = pd.get_dummies(custom_cat_df, drop_first=False)
                    custom_cat_aligned = custom_cat_dummies.reindex(columns=st.session_state.nn_cat_dummy_cols, fill_value=0).values
                else:
                    custom_cat_aligned = np.empty((1, 0))
                full_input = np.hstack([custom_num_scaled, custom_cat_aligned])
                prediction = st.session_state.nn_model.predict(full_input)
                predicted_index = np.argmax(prediction, axis=1)[0]
                if st.session_state.nn_le is not None:
                    predicted_class = st.session_state.nn_le.inverse_transform([predicted_index])[0]
                else:
                    predicted_class = predicted_index
                st.success(f"🎯 Predicted {target_column}: {predicted_class}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")






#############################################
# d) LLM Q&A using Gemini AI / LLM RAG & Multimodal
#############################################
elif task == "LLM Q&A":
    st.header("💬 LLM Q&A using Gemini AI")
    st.markdown("""
    **LLM Approach Options:**
    - **Retrieval-Augmented Generation (RAG):** Extracts the most relevant text snippets from the selected dataset based on your query.
    - **Multimodal:** Uses the entire content as context along with your query.
    
    **Instructions:**
    1. Choose your LLM approach.
    2. Select a preloaded dataset.
    3. Preview the content (CSV is shown in tabular form; PDFs are available for download).
    4. Enter your question and receive a real-time answer.
    """)

    # Choose LLM Approach.
    llm_approach = st.radio("Select LLM Approach", options=["RAG", "Multimodal"], index=0)

    # Load environment variables and initialize Gemini API.
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
        st.stop()

    # Instantiate the Gemini API client.
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    # Use the gemini-2.0-flash model.
    model = client.models  # We'll call generate_content via this client

    # Use caching to avoid repeated heavy PDF processing.
    @st.cache_data(show_spinner=True)
    def extract_text_from_pdf(file_path):
        """Extract text from a PDF file for retrieval purposes."""
        try:
            import PyPDF2
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    @st.cache_data(show_spinner=True)
    def get_pdf_base64(file_path):
        """Read and encode a PDF file in base64 for download."""
        try:
            import base64
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            st.error(f"Error encoding PDF: {e}")
            return ""
    
    # For PDFs, instead of trying to embed them which might be slow, we offer a download option.
    def offer_pdf_download(file_path):
        """Offer a download button for the PDF and inform the user."""
        base64_pdf = get_pdf_base64(file_path)
        file_name = os.path.basename(file_path)
        st.download_button("Download PDF", data=base64_pdf, file_name=file_name, mime="application/pdf")
        st.info("Download the PDF and view it with your local PDF reader or browser.")

    # Preloaded datasets (update file paths as needed)
    datasets = {
        "Ghana Election Results (CSV)": "/Users/nosei-opoku/Desktop/MyProjects/AI_EXAM/Ghana_Election_Result.csv",
        "2025 Budget Statement (PDF)": "/Users/nosei-opoku/Desktop/MyProjects/AI_EXAM/2025-Budget-Statement-and-Economic-Policy_v4.pdf",
        "Academic City Student Handbook (PDF)": "/Users/nosei-opoku/Desktop/MyProjects/AI_EXAM/handbook.pdf"
    }

    dataset_name = st.selectbox("Choose a dataset:", list(datasets.keys()))
    selected_path = datasets[dataset_name]
    content = ""
    df_csv = None

    # Load and preview content based on file extension.
    if selected_path.endswith(".csv"):
        df_csv = pd.read_csv(selected_path)
        st.subheader("📊 CSV Preview")
        st.dataframe(df_csv.head(10))
        # For retrieval purposes, convert the DataFrame to text.
        content = df_csv.to_string(index=False)
    elif selected_path.endswith(".pdf"):
        st.subheader("📄 PDF Preview")
        with st.spinner("Processing PDF... Please wait"):
            offer_pdf_download(selected_path)
            content = extract_text_from_pdf(selected_path)

    # -------------------------------------------------------------
    # Retrieval for RAG Approach
    # -------------------------------------------------------------
    if llm_approach == "RAG":
        st.markdown("### RAG Retrieval Settings")
        import re
        paragraphs = content.split("\n\n")
        num_passages = st.slider("Number of top passages to retrieve:", 1, 10, 3)
        query = st.text_input("Enter your question for RAG:")
        if query:
            query_terms = set(re.findall(r'\w+', query.lower()))
            ranked = []
            for p in paragraphs:
                p_terms = set(re.findall(r'\w+', p.lower()))
                score = len(query_terms.intersection(p_terms))
                ranked.append((score, p))
            ranked.sort(key=lambda x: x[0], reverse=True)
            retrieval_context = "\n\n".join([p for score, p in ranked[:num_passages]])
        else:
            retrieval_context = content
    else:
        query = st.text_input("Enter your question:")
        retrieval_context = content

    st.subheader("❓ Ask a Question")
    if st.button("Ask Gemini"):
        if retrieval_context and query:
            with st.spinner("Thinking..."):
                try:
                    response = model.generate_content(
                        model='gemini-2.0-flash',
                        contents=[retrieval_context, query]
                    )
                    st.success("🧠 Gemini’s Answer:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        else:
            st.warning("Please load content and enter a question.")



