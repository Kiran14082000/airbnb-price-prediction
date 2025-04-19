# 🏠 Airbnb Price Prediction with Machine Learning

This project aims to predict Airbnb listing prices using various machine learning models including Linear Regression, Random Forest, and a Multi-Layer Perceptron (MLP) Neural Network. The notebook provides an end-to-end workflow from data preprocessing to model evaluation.

---

## 📁 Files

- `main_optimized.ipynb` – Complete Jupyter Notebook with:
  - EDA (Exploratory Data Analysis)
  - Feature Engineering
  - Model Training (Linear Regression, Random Forest, MLP)
  - Hyperparameter Tuning & Regularization
  - Evaluation Metrics (MSE, MAE, R²)

---

## 📊 Features Covered

- 🔍 Exploratory Data Analysis (EDA)
- 🧼 Data Cleaning & Handling Missing Values
- 🧠 Feature Selection with RFE
- 📈 Model Building:
  - Linear Regression
  - Random Forest Regressor
  - Deep Learning (MLP with Dropout, L2 Regularization, EarlyStopping)
- ⚙️ Hyperparameter Tuning (Grid/Random Search)
- 🧪 Evaluation using MSE, MAE, and R²
- 📉 Overfitting Handling using Dropout and EarlyStopping

---

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

---

## 🧠 Model Performance

| Model              | Train R² | Test R² | Train MSE | Test MSE |
|-------------------|----------|---------|-----------|----------|
| Linear Regression |   ~0.94  |  ~0.93  |    —      |    —     |
| Random Forest     |   ~0.98  |  ~0.96  |    —      |    —     |
| MLP Neural Net    |   ~0.94  |  ~0.94  |    —      |    —     |

*Note: Replace metrics with your actual values from the notebook.*

---

## 📦 Dataset

- **Source**: [Airbnb Open Data](https://www.kaggle.com/datasets)
- **Columns Used**: `price`, `room_type`, `number_of_reviews`, `location`, etc.
- **Preprocessing**:
  - Label encoding
  - Feature scaling
  - Outlier removal

---

## 📎 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Kiran14082000/airbnb-price-prediction.git
   cd airbnb-price-prediction
   ```

2. Open the notebook:
   ```bash
   jupyter notebook main_optimized.ipynb
   ```

3. Run cells in order for full execution.

---

## 🚀 Future Work

- Incorporate location-based features using geohashing
- Apply XGBoost or CatBoost for advanced regression
- Hyperparameter optimization with Optuna
- Deploy as a web app using Streamlit or Flask

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Kiran Gobi Manivannan**  
📧 kiranxgobi@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/kiran14082000) • [GitHub](https://github.com/Kiran14082000)
