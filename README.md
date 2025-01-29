# A Performance Study of Classification Models for Motor Imagery EEG Data

## Introduction
Recent advancements in EEG technologies have made it more accessible to acquire EEG devices capable of recording brain wave data. Brain-computer interfaces (BCIs) use this data to operate devices through thought processes. However, raw EEG data requires effective preprocessing to ensure suitability for machine learning algorithms. This study compares various classification models to identify the most accurate algorithm for classifying EEG signals.

## Dataset Description
The dataset used in this study is the SCP dataset (5a) compiled by Kaya et al. (2018). It contains EEG recordings from 13 participants, with this analysis focusing on Subject A's session from April 5th, 2016. The dataset captures motor imagery tasks corresponding to single-finger flexion movements.

## Data Preprocessing
- The dataset was available in `.mat` format and loaded using `scipy`.
- EEG data was structured into (3595, 22, 1000) format.
- Preprocessing steps included:
  - Filtering using `MNE` library
  - Feature extraction via Power Spectral Density (PSD)
  - Label assignment for motor imagery tasks
  - Data balancing using Synthetic Minority Oversampling Technique (SMOTE)
  - Standardization for feature consistency

## Data Visualization
- **Raw EEG Signals**: Visualized across all 22 channels.
- **Power Spectral Density (PSD)**: Computed for extracting meaningful frequency components.
- **Average EEG Signal**: Analyzed across all trials and channels.

## Machine Learning Models
The following models were evaluated:
- **Random Forest** (92.35%)
- **Support Vector Machine (SVM)** (26.66%)
- **k-Nearest Neighbors (KNN)** (85.93%)
- **Logistic Regression** (44.28%)
- **Decision Tree** (80.40%)
- **Gradient Boosting** (81.91%)
- **CatBoost** (90.14%)
- **LightGBM** (78.59%)
- **XGBoost** (92.83%)

## Results and Discussion
- **XGBoost** achieved the highest accuracy (92.83%), followed closely by **Random Forest** (92.35%).
- Significant misclassification was observed for label 0, likely due to class imbalance.
- Future work may involve hyperparameter tuning and deep learning approaches to improve performance.

## Conclusion
This study identified **XGBoost** as the best-performing model for EEG classification. The results contribute to improving BCIs by enhancing classification accuracy. Future research should explore deep learning architectures and advanced sampling techniques to optimize performance further.

## References
1. Kaya, M., et al. (2018). A large electroencephalographic motor imagery dataset for EEG-BCIs. [Figshare](https://doi.org/10.6084/m9.figshare.c.3917698.v1).
2. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.


