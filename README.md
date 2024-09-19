# Gorilla Vocalization Detection Using Machine Learning Models

## Project Overview

This repository contains the scripts, models, and results for the detection of gorilla vocalizations using machine learning approaches, including Independent Component Analysis (ICA), Wave-U-Net, and RNN-LSTM. The primary objective is to identify gorilla vocalizations (specifically "food songs") from a set of 450 audio recordings. The audio data has been extracted and analyzed using the methods described below.

The work demonstrates the application of signal processing and machine learning techniques to bioacoustic data, focusing on gorilla vocalizations in varying environments with background noise.

## Folder Structure

- **notebooks/**
  - **RNN_LSTM.ipynb**: Jupyter notebook for training, evaluating, and making predictions with the RNN-LSTM model. Follows a modular approach with separate steps for preprocessing, feature extraction, model training, evaluation, and prediction.
  - **ICA_on_a_single_audio.ipynb**: Jupyter notebook for applying the ICA model. Includes data loading, preprocessing, and evaluation.
  - **wave-u-net.ipynb**: Jupyter notebook for training and evaluating the Wave-U-Net model. Combines preprocessing, training, and evaluation.
  - **wave-unet-with-wav2vec2.ipynb**: Jupyter notebook for training and evaluating the Wave-U-Net model built on a pre-trained Wave2Vec2 model. Includes preprocessing, training, and evaluation.

## Getting Started

### Prerequisites

Ensure you have the following Python packages installed:

- TensorFlow
- NumPy
- pandas
- librosa
- matplotlib
- seaborn
- scikit-learn
- Jupyter

Install these packages using `pip`:

```bash
pip install tensorflow numpy pandas librosa matplotlib seaborn scikit-learn jupyter
```

### Data Preparation

1. **Download the Data**: Access the required data files from the following link:

   [Download Data](https://drive.google.com/drive/folders/1nrun9TjCE3X1Nt92NF10SfEwoc5klWTM?usp=drive_link)

   Place the data files in the `data/raw/` directory of your project.

2. **Data Format**: Ensure audio files are in `.wav` format and your Excel file is properly formatted.

### Running the Jupyter Notebooks

1. **Launch Jupyter Notebook**: Navigate to your project directory and start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. **Open and Run Notebooks**: In the Jupyter interface, open the desired notebook and execute the cells. You can run all cells in order or execute them individually.

   - **RNN_LSTM.ipynb**: Contains the RNN-LSTM model with a modular approach. Includes steps for preprocessing, feature extraction, model training, evaluation, and prediction.
   - **ICA_on_a_single_audio.ipynb**: Includes the ICA model pipeline. Handles data loading, preprocessing, and evaluation.
   - **wave-u-net.ipynb**: Covers the full pipeline for the Wave-U-Net model, including preprocessing, training, and evaluation.
   - **wave-unet-with-wav2vec2.ipynb**: Includes the full pipeline for the Wave-U-Net model with a pre-trained Wave2Vec2 feature extractor, covering preprocessing, training, and evaluation.

### Evaluation

The evaluation sections in each notebook generate:

- Learning curves showing accuracy and loss over epochs.
- A confusion matrix to visualize model performance.
- A classification report with precision, recall, and F1-score metrics.

### Example Output

After running a notebook, you will find:

- **Model checkpoints**: Saved model weights and architecture.
- **Plots**: Learning curves and confusion matrix plots.
- **Classification Report**: Metrics for evaluating model performance.
- **Predictions**: Output from the RNN-LSTM model, including predicted labels for new audio recordings.

## Contributing

Contributions are welcome! Submit pull requests or open issues to contribute to this project, such as adding new models or improving preprocessing functions.


## Contact

For any questions or comments, please reach out to [ayobamiomolusi@gmail.com].
