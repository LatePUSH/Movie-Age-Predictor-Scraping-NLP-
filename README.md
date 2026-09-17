# 🎬 Movie Age Predictor (Scraping & NLP)

This Language Engineering project (Université Paris 8) combines web data collection (scraping) and Natural Language Processing (NLP). The goal is to automatically predict the recommended age for a movie based solely on its synopsis.

## 📌 Project Architecture
The project is divided into two main steps:
1. **Corpus Creation (`scraper.py`)**: Extraction of 15,000 movies from an online cinema database, spread across three genres (Action, Romance, Horror) using `BeautifulSoup` and `requests`.
2. **Text Classification (`trainer.py`)**: Use of the pre-trained **DistilBERT** model from HuggingFace. The age labels are mapped to 8 numerical classes. The model is fine-tuned on 80% of the data and evaluated on the remaining 20%.

## 🚀 Installation & Prerequisites

Make sure you have Python 3.8+ installed with access to a GPU (CUDA) to speed up the training. 
Install the required dependencies:

> pip install -r requirements.txt

## 🧠 Usage

**1. Collect new data (Optional)**
If you want to generate your own JSON files from the web:

> python scraper.py

**2. Train and test the model**
Make sure you have the global `merged_movies.json` file containing the fusion of your data, then run:

> python trainer.py

The model and the tokenizer will be automatically saved in the `./saved_model` folder at the end of the 5 epochs.

## 📊 Results & Metrics
The model was evaluated by measuring Accuracy, Precision, Recall, and F1-Score. BERT's bidirectional approach captures the nuances of violence or language in a synopsis to classify it accurately. The detailed metrics and the system's limitations (synopsis length) are discussed in the `Rapport.pdf` included in this repository.
