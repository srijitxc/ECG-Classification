# ECG Heartbeat Classification Using 1D CNN

A deep learning project that classifies ECG (Electrocardiogram) heartbeat signals into five heartbeat categories using a **1D Convolutional Neural Network (CNN)** built with TensorFlow and Keras.

## Project Overview

Electrocardiogram (ECG) signals record the electrical activity of the heart. This project uses heartbeat signal samples to train a neural network to distinguish between five heartbeat classes.

The notebook implements the following workflow:

1. Import the required Python libraries.
2. Download the MIT-BIH heartbeat dataset.
3. Load and inspect the training and testing data.
4. Visualize heartbeat class distribution and representative ECG signals.
5. Prepare the data for a 1D CNN.
6. Build, compile, and train the CNN model.
7. Evaluate predictions using a classification report and confusion matrix.

## Heartbeat Classes

The notebook uses five target classes:

| Label | Category |
|---:|---|
| 0 | Normal (N) |
| 1 | Supraventricular Premature (S) |
| 2 | Premature Ventricular Contraction (V) |
| 3 | Fusion Beat (F) |
| 4 | Unclassifiable Beat (Q) |

## Dataset

The notebook attempts to download the MIT-BIH heartbeat CSV files from the following repository:

- Training data: [mitbih_train.csv](https://raw.githubusercontent.com/shayanfazeli/heartbeat/master/mitbih_train.csv)
- Testing data: [mitbih_test.csv](https://raw.githubusercontent.com/shayanfazeli/heartbeat/master/mitbih_test.csv)
- Dataset repository: [shayanfazeli/heartbeat](https://github.com/shayanfazeli/heartbeat)

It first tries the `media.githubusercontent.com` mirror and then falls back to the `raw.githubusercontent.com` URLs.

Each row is expected to contain **187 signal values followed by one class label**. The notebook reads the CSV files without headers and separates the final column as the target.

### Synthetic fallback

If both download attempts fail, the notebook generates a small synthetic dataset:

- 500 training samples
- 100 testing samples
- 187 signal values per sample
- Five possible class labels

The synthetic data is generated from mathematical waveforms with added noise and variation. It is a fallback for running the notebook, **not a substitute for real MIT-BIH data or a clinically validated dataset**. Check which dataset was loaded before interpreting model performance.

## Model Architecture

The project uses a Sequential 1D CNN designed for one-dimensional ECG signal input.

| Layer | Configuration |
|---|---|
| Input | 187 time points, 1 channel |
| Conv1D | 64 filters, kernel size 6, ReLU |
| Batch Normalization | — |
| MaxPooling1D | Pool size 3, stride 2 |
| Conv1D | 128 filters, kernel size 3, ReLU |
| Batch Normalization | — |
| MaxPooling1D | Pool size 2, stride 2 |
| Flatten | — |
| Dense | 64 units, ReLU |
| Dropout | 0.3 |
| Output Dense | 5 units, Softmax |

### Training configuration

- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metric: Accuracy
- Epochs: 15
- Batch size: 32
- Input shape: `(187, 1)`

The integer labels are converted to categorical (one-hot) vectors before training.

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Requests
- TensorFlow / Keras
- Scikit-learn

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

Replace the placeholders with your GitHub repository URL and folder name.

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy requests tensorflow scikit-learn jupyter
```

### 3. Open Jupyter Notebook

```bash
jupyter notebook
```

Open the ECG notebook (`ecg (1).ipynb`) and run the cells in order.

The notebook downloads the dataset automatically when network access is available. Otherwise, it attempts to generate the synthetic fallback dataset.

## Project Structure

```text
ECG-Heartbeat-Classification/
│
├── ecg (1).ipynb       # Data processing, CNN training and evaluation
├── mitbih_train.csv   # Downloaded training data (generated at runtime)
├── mitbih_test.csv    # Downloaded testing data (generated at runtime)
└── README.md          # Project documentation
```

The CSV files are created in the notebook's current working directory when the notebook runs. They do not need to be committed if you prefer to download them at runtime.

## Visualizations and Evaluation

The notebook generates:

- Training dataset class-distribution plot
- ECG heartbeat signal comparison across the five categories
- Training and validation accuracy curves
- Training and validation loss curves
- Classification report with precision, recall, and F1-score
- Confusion matrix

These outputs help inspect class imbalance, training behavior, and classification performance.

## Important Notes and Limitations

- The notebook uses the test dataset as `validation_data` during model training and later evaluates predictions on that same dataset. For an unbiased final evaluation, split the training data into training and validation sets, and reserve the test set for final evaluation only.
- The synthetic fallback is small and artificially generated. Results obtained from it should not be interpreted as performance on real ECG recordings.
- The notebook does not implement a real-time ECG acquisition system or clinical diagnostic workflow.
- Model accuracy and other metric values are not stated here because they depend on the actual dataset and training run.

## Future Improvements

- Create a separate validation split from the training dataset.
- Add per-class performance analysis to better understand minority-class predictions.
- Save the trained model for later inference.
- Add a function to classify a new ECG sample.
- Experiment with class weighting or resampling to address class imbalance.
- Test the model on an independent dataset.

## Disclaimer

This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose, treat, or make clinical decisions about heart conditions.

## Author

**Srijit Dam**

If you found this project useful, consider giving the repository a ⭐ on GitHub.
