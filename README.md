# Fake Review Detection

*Uncovering authenticity, one review at a time.*

## 📖 Overview

Fake reviews harm online trust and decision-making. This project uses NLP feature engineering and machine learning models to systematically detect computer-generated (fake) reviews from human-written ones.

We leverage Scikit-Learn models, NLTK/Spacy preprocessing, and track experiments using MLflow, deploying our final model via a Dockerized Streamlit app for live testing.

---

## 📂 Project Organization

```
├── LICENSE
├── Makefile           <- Commands like `make data`, `make train`
├── README.md          <- This file
├── data
│   ├── external       <- Third-party raw data
│   ├── interim        <- Transformed intermediate data
│   ├── processed      <- Final dataset for modeling
│   └── raw            <- Original immutable data dump
├── docs               <- Sphinx documentation
├── models             <- Trained models and predictions
├── notebooks          <- Jupyter notebooks for EDA and testing
├── references         <- Data dictionaries and manuals
├── reports            <- Generated analysis and figures
├── requirements.txt   <- Dependency list
├── setup.py           <- Pip installable project structure
├── src                <- Source code
│   ├── data           <- Data collection and generation
│   ├── features       <- Feature engineering scripts
│   ├── models         <- Training and prediction scripts
│   └── visualization  <- EDA and result visualizations
└── tox.ini            <- Testing configurations
```

---

## 🛠️ Installation

### Prerequisites

* Python 3.8+
* Docker (for deployment)
* Git

### Steps

```bash
git clone https://github.com/YogeshKumar-saini/Fake-Review-Detection.git
cd fake-review-detection

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🚀 Usage

### Preprocess Data

```bash
make data
```

### Train the Model

```bash
make train
```

### Explore Notebooks

```bash
jupyter notebook notebooks/
```

### Launch Streamlit App Locally

```bash
docker build -t fake-review-app .
docker run -p 8501:8501 fake-review-app
```

Access the app at [http://localhost:8501](http://localhost:8501).

---

## ☁️ Deployment on AWS

Deploy the Streamlit app on an AWS EC2 instance:

```bash
docker run -d -p 80:8501 yogesh1090/fake-review-app:latest
```


Access via: `http://13.53.99.130:8501/`

Ensure your EC2 security group allows inbound traffic on port 80.

---

## 📊 Experiment Tracking with MLflow

Track experiments and performance:

```bash
mlflow ui
```

Access the MLflow dashboard at [http://localhost:5000](http://localhost:5000).

---

## 🧩 Dataset

* Binary classification dataset of reviews labeled as `fake` or `real`.
* Source: Public datasets like Yelp, Amazon, or custom scraping.
* Includes review text, optional user metadata, and labels.

---

## 🛠️ Feature Engineering

* **Text preprocessing:** tokenization, stop-word removal, lemmatization.
* **Feature extraction:** TF-IDF vectors, sentiment scores, review length, readability scores, POS tagging.
* Potential advanced embeddings using BERT for future enhancement.

---

## 🤖 Model Details

* Models tested: Logistic Regression, Random Forest, SVM.
* Evaluation metrics: Accuracy, F1-score, ROC-AUC.
* Best model persisted via Pickle and tracked on MLflow.

---

## 📈 Results

* Achieved \~90% accuracy on validation data.
* ROC-AUC consistently above 0.92.
* Full experiment logs available on MLflow.

---

## 🖥️ Streamlit App Features

* Input a review to check if it is `fake` or `real` with confidence probability.
* Display dataset statistics and charts.
* Option to view EDA and model performance graphs.

---

## 🛠️ Technologies Used

* Python 3.8+
* Scikit-Learn
* NLTK / Spacy
* MLflow
* Streamlit
* Docker
* Pandas / NumPy

---

## 📚 Resources

* [MLflow Documentation](https://mlflow.org/)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
* [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
* [NLTK Book](https://www.nltk.org/book/)

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request.

For major changes, please open an issue to discuss your proposal first.

---

## 📜 License

This project is licensed under the MIT License.

---

## 📬 Contact

Maintained by Yogesh Saini.

For inquiries, please open an issue or email at [yksaini1090@gmail.com](mailto:yksaini1090@gmail.com).

---


> **“Uncovering authenticity, one review at a time.”**
