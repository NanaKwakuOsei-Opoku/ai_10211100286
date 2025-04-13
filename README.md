### NANA KWAKU OSEI-OPOKU - 10211100286
# 🌐 ML & AI Explorer Dashboard

This repository hosts a Streamlit-based application showcasing multiple machine learning approaches—**Linear Regression**, **K-Means Clustering**, and **Neural Networks**—along with a powerful **LLM Q&A** feature using **Google’s Gemini AI** in **RAG mode**.

---

## 📂 Project Structure

```
ML_AI_Explorer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # List of dependencies
├── .env                    # Environment variables (ignored by Git)
├── data/                   # (Optional) Folder for local CSV/PDF data
└── ...                     # Additional logs or assets
```

### 🤖 LLM Implementation Diagram


```![Screenshot 2025-04-13 at 7 37 41 PM](https://github.com/user-attachments/assets/6fe1be64-9046-4415-a8c5-560628b2b72b)

---

## 🚀 How It Works

1. **Unified Dashboard Navigation**  
   - A left sidebar lets you switch between four tasks: **Regression**, **Clustering**, **Neural Network**, and **LLM Q&A**.
   - All tasks share a consistent interface, each with relevant configuration options.

2. **User-Friendly Data Interfaces**  
   - Upload CSV or PDF files.
   - Handle missing values (drop rows or fill using mean/mode).
   - View descriptive statistics and info.

3. **Model Training & Results Visualization**  
   - **Regression**: Train a linear model, visualize predictions vs. actual, and make custom predictions.  
   - **Clustering**: Perform K-Means clustering, visualize cluster centroids, and examine 2D/3D or reduced dimensionality plots.  
   - **Neural Network**: Train a basic classifier with live progress tracking (epochs), see validation metrics, and predict on custom inputs.

4. **LLM Q&A (RAG Mode)**  
   - Preloaded PDFs:  
     - **2025 Budget Statement**  
     - **Academic City Student Handbook**  
   - Extract text via PyPDF2, split into paragraphs, then retrieve top passages based on your query.  
   - Queries + retrieved passages are fed to **gemini-1.5-flash** for relevant responses.  

---

## 🎛 Usage Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Set Up Environment**:
   ```bash
   # Inside your .env file
   GEMINI_API_KEY=<Your Gemini API Key Here>
   ```
3. **Run Streamlit**:
   ```bash
   streamlit run app.py
   ```
4. **Navigate the App**:
   - **Regression** tab: Upload CSV, train & evaluate a linear regression model, and test custom inputs.
   - **Clustering** tab: Explore K-Means with interactive 2D/3D or PCA/t-SNE options.
   - **Neural Network** tab: Build a feed-forward classifier with live epoch-by-epoch feedback.
   - **LLM Q&A** tab: Load PDF data, retrieve relevant context passages, and query the Gemini AI for real-time answers.

---

## (d) Datasets & Models (LLM Q&A)

### Datasets
- **2025 Budget Statement (PDF)**: Comprehensive economic policy document.  
- **Academic City Student Handbook (PDF)**: Official rules & guidelines for students.

### Architecture (LLM Q&A)
1. **PDF Parsing & Caching** → Convert PDF pages to text once, store for quick retrieval.  
2. **Ranking & Retrieval** → Paragraphs are split and ranked based on shared words with the query.  
3. **Generative Model** → **gemini-1.5-flash** merges context + user query to provide concise answers.

### Methodology (LLM Q&A)
- **Retrieval-Augmented Generation (RAG)**: Ranks and inserts top-matching passages into the LLM prompt.  
- **Generative Response**: The AI uses the combined text and query to produce contextually relevant answers.

---

## 📞 Contact

- **Email**: nana.osei@acity.edu  
- **Phone**: 0550705822  

Feel free to reach out with questions or issues! Thank you for exploring this **ML & AI Explorer Dashboard**
