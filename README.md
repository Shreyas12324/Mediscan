# MediScan - Eye Disease Detection

This project is an eye disease detection system that uses a machine learning model to predict whether an eye image corresponds to one of the following conditions: Cataract, Diabetic Retinopathy, Glaucoma, or Normal.

## Features

-   Predicts four classes of eye conditions.
-   Simple web interface to upload an image and see the prediction.
-   Displays a history of predictions.

## Technologies Used

-   **Frontend:** Streamlit
-   **Backend:** Flask
-   **Machine Learning:** Scikit-learn

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/mediscan.git
    cd mediscan
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv mediscan_env
    source mediscan_env/bin/activate  # On Windows, use `mediscan_env\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Download the dataset:**
    The dataset is not included in this repository due to its size. You can download it from [Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification). Once downloaded, extract it to a `dataset` directory in the root of the project.

4.  **Train the model:**
    To train the SVM model, run the `week1_model_training.ipynb` notebook. This will generate the `svm_model.pkl` and `scaler.pkl` files.

5.  **Run the backend:**
    ```bash
    python backend.py
    ```

6.  **Run the frontend:**
    In a new terminal, run:
    ```bash
    streamlit run app.py
    ```

    The application will be available at `http://localhost:8501`.
