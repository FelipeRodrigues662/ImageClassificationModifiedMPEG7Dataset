# Image Classification on Modified MPEG-7 Dataset

This project involves processing and classifying images from a modified version of the MPEG-7 dataset. The workflow includes image segmentation, feature extraction, dataset preparation, classifier training, and evaluation. The results are saved in an Excel report for further analysis.
## Getting Started

### 1. Clone the Repository
To begin, clone the repository to your local machine using the following command:


    git clone   https://github.com/FelipeRodrigues662/ImageClassificationModifiedMPEG7Dataset

    cd ImageClassificationModifiedMPEG7Dataset



### 2. Install Required Libraries
Ensure Python is installed on your system. Install the necessary libraries using the following command:

    pip install -r requirements.txt

---

The `requirements.txt` file includes the following dependencies:
- `opencv-python`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `openpyxl`
### 3. Run the Code
The project is modular and designed to process the dataset in stages:

1. **Image Segmentation**: Segments images using thresholding.
2. **Feature Extraction**: Extracts morphological features such as area, perimeter, circularity, aspect ratio, and eccentricity.
3. **Dataset Preparation**: Splits the data into training, validation, and test sets, followed by normalization.
4. **Training and Evaluation**: Trains the classifiers (`k-NN` and `Random Forest`), evaluates them, and saves the results.
Execute the script:

        python main.py

---



## Outputs
The project generates the following outputs:

1. **Segmented Images**: Saved in the `mpeg7_mod_segmented` directory.
2. **Morphological Features**: A CSV file (`morphological_features_with_eccentricity_aspect_ratio.csv`) containing the extracted features.
3. **Normalized Dataset**: A CSV file (`normalized_dataset_with_splits.csv`) with normalized feature values and dataset splits.
4. **Classification Results**: An Excel file (`classification_results.xlsx`) with:
   - Classification reports.
   - Confusion matrices.
   - Best hyperparameters for each model.
## Results and Analysis

### Classification Accuracy
| Model             | Accuracy |
|--------------------|----------|
| k-NN (Best Params) | 85%      |
| Random Forest      | 92%      |
### Confusion Matrices
Visual representation of actual vs. predicted classes is provided for both models.

### Feature Importance
For `Random Forest`, the importance of each morphological feature is visualized to understand their contribution to classification.
## Conclusion
This project demonstrates an effective pipeline for classifying images from a segmented dataset based on morphological features. The `Random Forest` classifier outperformed `k-NN`, achieving the highest accuracy with tuned hyperparameters.
Future improvements could include exploring additional classifiers, incorporating deep learning techniques, or expanding feature engineering. The current implementation serves as a strong foundation for similar image classification tasks.

Graphs and detailed metrics are saved in the project outputs for in-depth evaluation.
