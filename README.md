# FinSort - Complete Prototype

This repo contains a full working prototype of FinSort: local transaction categorization with a configurable taxonomy.

## 📽 Demo Video
Watch the demonstration here:  
🔗 (https://drive.google.com/drive/folders/1g84kALKV_2iH49TKdJo7w-rkGQ5o2b55?usp=sharing)

## What is included
- `finsort/` package: cleaner, model training, inference, explain, config
- `data/`: train/test CSVs and full dataset
- `demo/`: interactive CLI demo and Streamlit web UI
- `train.py`, `evaluation.py`, `requirements.txt`
- `docs/Our Idea and Why It Matters.pdf` - Original project brief (see [Project Brief](#project-brief))

## Quickstart

### 1. Create virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```

This will:
- Load training data from `data/finsort_train.csv`
- Train a LogisticRegression model with TF-IDF vectorization
- Save the model to `finsort/model.pkl` and vectorizer to `finsort/vectorizer.pkl`
- Print classification report and Macro F1 score

### 4. Run CLI demo
```bash
python demo/demo.py
```

Interactive command-line interface for testing predictions.

### 5. Run Streamlit web UI
```bash
streamlit run demo/streamlit_app.py
```

Opens a web interface in your browser for easy transaction categorization.

### 6. Edit category mappings

Edit `finsort/config.json` to change how model tags map to final categories. No retraining needed!

Example:
```json
{
  "coffee_shop": "Food & Dining",
  "grocery": "Groceries",
  "ecommerce": "Shopping"
}
```

## Project Brief

The original project brief can be found at: `docs/Our_Idea_and_Why_It_Matters.pdf`

## Additional Commands

### Evaluate model performance
```bash
python evaluation.py
```

Runs evaluation on test data and prints classification report with confusion matrix.

### Run tests
```bash
pytest -q
```

Runs the unit test suite.

## Notes
- `finsort/config.json` maps model tags to final categories — edit without retraining.
- Feedback corrections from the UI are appended to `finsort/feedback.log`.
- Do not commit model binaries to GitHub; `.gitignore` excludes them by default.
