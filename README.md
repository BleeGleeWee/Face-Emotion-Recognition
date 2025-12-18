
# 🧠 Image Classification with Keras & TensorFlow
## Deployed link - ✨Here
([https://emotion-recognition-ml.streamlit.app/](https://emotion-recognition-ml.streamlit.app/))

This repository contains a deep learning project built with **Keras** and **TensorFlow** for image classification.  
The model is trained using convolutional neural networks (CNNs) on a custom dataset and demonstrates the full workflow of preprocessing, model training, evaluation, and saving the trained model.

---

## 📌 Features
- Data preprocessing and augmentation  
- Convolutional Neural Network (CNN) model built with Keras  
- Training loop with validation monitoring  
- Checkpoint saving & early stopping  
- Accuracy and loss visualization  
- Easily extendable for other datasets  

---

## 📂 Project Structure
```

├── trainFERmodel.ipynb           # Main Jupyter Notebook for training
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Ignored files/folders
└── realtimedetection.py       # Dataset (ignored in git)

````

---

## ⚙️ Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/BleeGleeWee/Face-Emotion-Recognition.git
   cd Face-Emotion-Recognition


2. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

1. Place your dataset inside the `data/` folder.

   * Training and validation data should be structured as:

     ```
     data/
       train/
         class1/
         class2/
         ...
       val/
         class1/
         class2/
         ...
     ```

2. Open the Jupyter Notebook:

   ```bash
   jupyter notebook trainFERmodel.ipynb
   ```

3. Run all cells to train the model.

---

## 📊 Results

* Training accuracy: \~74%
* Validation accuracy: \~63%
* Final model saved in `saved_models/` (ignored in git by default).

> 📌 Training logs and accuracy/loss plots are included in the notebook.

---

## 🛠️ Technologies Used

* Python 3.x
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib, Seaborn (for visualization)
* Jupyter Notebook

---

## 🤝 Contributing

This project is open for contributions. Feel free to fork the repo, create a branch, and submit pull requests with improvements (e.g., better models, hyperparameter tuning, visualization).

---

## 📜 License

This project is for **educational and research purposes only**.
Feel free to use, modify, and share with attribution.

---

## 🙌 Acknowledgements

* TensorFlow/Keras community
* Open source datasets used for training
* Inspiration from various deep learning research papers and tutorials
