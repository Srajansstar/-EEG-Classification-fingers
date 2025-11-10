# A Performance Study of Classical and Deep Learning Models for Motor Imagery EEG Data

## Introduction

Recent advancements in EEG technologies have made it more accessible to acquire EEG devices capable of recording brain wave data. Brain-computer interfaces (BCIs) use this data to operate devices through thought processes. However, raw EEG data requires effective preprocessing to ensure suitability for machine learning algorithms.

This study implements and compares three distinct classification approaches to identify the most accurate algorithm for classifying these EEG signals:

1.  **Classical Machine Learning models** (e.g., Random Forest, XGBoost) on Power Spectral Density (PSD) features.
2.  **Recurrent Neural Networks** (RNN, GRU, LSTM) on the same PSD features.
3.  **A hybrid CNN-LSTM model** on the raw, time-series EEG data, with a focus on comparing optimizer performance.

---

## Dataset Description

The dataset used in this study is from a larger BCI competition dataset, specifically subject 'A' from a 5-finger motor imagery task (`5F-SubjectA-160408-5St-SGLHand-HFREQ.mat`). It contains EEG recordings for tasks corresponding to single-finger flexion movements (thumb, index, middle, ring, pinkie) and a 'no raise' (rest) state, captured across 22 EEG channels.

---

## Methodology and Preprocessing

The `.mat` file was loaded using `scipy` and processed using the `mne` library to create an `EpochsArray`. Two distinct processing pipelines were established for the different modeling approaches.

### Pipeline 1: Feature-Based Analysis (Classical ML & RNNs)

This pipeline was used for the classical ML models and the simple recurrent models:

* **Standardization:** Raw epochs were first scaled using `mne.decoding.Scaler`.
* **Feature Extraction:** Power Spectral Density (PSD) features were extracted from the scaled epochs using the `psd_multitaper` method (`fmin=0.1`, `fmax=50`). This converts each epoch from a time-series signal into a feature vector representing power at different frequencies.
* **Data Balancing:** The dataset was filtered to include only the 6 valid task labels (0-5). Due to class imbalance, **BorderlineSMOTE** (a variant of SMOTE) was applied to oversample the minority classes, creating a balanced dataset for training.

### Pipeline 2: Raw Data Analysis (CNN-LSTM)

This pipeline was used for the hybrid deep learning model:

* **Data Filtering:** The raw epochs (`X_raw`) and labels (`y_raw`) were filtered to include only the 6 valid task labels (0-5).
* **No Feature Extraction:** The data was not converted to PSD. Instead, the raw (`n_epochs`, `n_channels`, `n_times`) tensors were used directly, allowing the CNN to learn spatial and temporal features automatically.

---

## Initial Data Visualization

Before feature extraction, `mne` was used to visualize the raw processed data:

* **Sensor Locations:** Plotted to confirm the 'standard\_1020' montage.
* **Average EEG Signal:** Visualized as an image plot and a standard line plot to show average activity across all trials.
* **Power Spectral Density (PSD):** A plot of the average PSD across all channels was generated.

---

## Model Evaluation and Results

The study was divided into three distinct comparative analyses.

### Part 1: Classical Machine learning Models (on PSD)

Nine classical ML models were trained and evaluated on the balanced PSD features. **XGBoost** achieved the highest accuracy, closely followed by **Random Forest**. The models and their resulting accuracies were:

* **Random Forest:** 92.35%
* **XGBoost:** 92.83%
* **CatBoost:** 90.14%
* **KNN:** 85.93%
* **Gradient Boosting:** 81.91%
* **Decision Tree:** 80.40%
* **LightGBM:** 78.59%
* **Logistic Regression:** 44.28%
* **Support Vector Machine (SVM):** 26.66%

Confusion matrices were generated for all models, revealing that despite SMOTE, models like SVM and Logistic Regression still struggled significantly with misclassifications, whereas XGBoost and Random Forest were far more robust.

### Part 2: Recurrent Models (RNN, GRU, LSTM) on PSD Features

To compare against the classical models, three recurrent architectures (**RNN**, **GRU**, **LSTM**) were trained on the exact same balanced PSD feature set. The models were trained for 3,000 epochs, and their test accuracy was plotted over time. This analysis aimed to see if sequence-modeling architectures (even on non-sequential features) offered any advantage. The resulting plots compare the learning and convergence speed of the three models.

### Part 3: Hybrid CNN-LSTM on Raw Data

This analysis tested a different approach: an end-to-end model that learns from raw signals, bypassing manual feature extraction. A hybrid model consisting of **1D CNN** layers (to extract spatial/temporal features) followed by a **bidirectional LSTM** (to model sequence dependencies) was used.

The focus of this part was to compare the performance of six different optimizers on this complex task. The model was trained for 50 epochs with each optimizer, and the training loss was plotted.

**Optimizers Compared:**

* SGD + Momentum
* NAG (Nesterov-accelerated Gradient)
* Adam
* RMSprop
* Adagrad
* Adadelta

The resulting loss curves and table show how quickly and effectively each optimizer minimized the model's loss, guiding the choice of the best optimizer for this specific architecture and dataset.

---

## Conclusion

This study performed a comprehensive three-part analysis for classifying motor imagery EEG data.

* In the classical ML comparison, **XGBoost (92.83%)** and **Random Forest (92.35%)** were the clear top performers on extracted PSD features.
* The analysis was extended to deep learning, first by comparing **RNN, GRU, and LSTM** models on the same PSD features to provide a performance baseline.
* Finally, a hybrid **CNN-LSTM** model on raw data was used to conduct a comparative study of optimizer performance, providing insight into the best training methods for end-to-end EEG classification.

This multi-faceted approach provides a broad and robust baseline for this dataset, highlighting that while tree-based models like XGBoost are highly effective on well-engineered features, end-to-end models also present a viable path, with optimizer choice being a critical factor.

---

## References

* Kaya, M., et al. (2018). A large electroencephalographic motor imagery dataset for EEG-BCIs. Figshare.
* Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.
