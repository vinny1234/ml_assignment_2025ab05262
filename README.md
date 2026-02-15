# \## ROLLNO:2025AB05262

# \## Machine Learning Assignment – 2

# \## Classification Model Comparison and Deployment using Streamlit

# 

# ---

# 

# \## a) Problem Statement

# 

# The objective of this project is to implement and compare multiple machine learning classification models on a medical diagnostic dataset.

# 

# The goal is to:

# \- Predict whether a tumor is Benign (B) or Malignant (M)

# \- Compare six different classification algorithms

# \- Evaluate them using multiple performance metrics

# \- Deploy an interactive Streamlit web application

# 

# This project demonstrates an end-to-end machine learning workflow including model training, evaluation, comparison, and deployment.

# 

# ---

# 

# \## b) Dataset Description

# 

# Dataset Name: Breast Cancer Wisconsin Dataset

# Source: UCI Machine Learning Repository

# 

# Shape: (569, 32)

# 

# After removing the `id` column, 30 numerical features were used for modeling.

# 

# Target Variable:

# \- B (Benign) → 0

# \- M (Malignant) → 1

# 

# Train-Test Split:

# \- Training Set: (455, 30)

# \- Test Set: (114, 30)

# 

# Class Distribution:

# \- Train: \[285 Benign, 170 Malignant]

# \- Test: \[72 Benign, 42 Malignant]

# 

# The dataset contains computed features from digitized images of breast mass cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

# 

# ---

# 

# \## c) Models Used and Evaluation Metrics

# 

# The following classification models were implemented:

# 

# 1\. Logistic Regression

# 2\. Decision Tree Classifier

# 3\. K-Nearest Neighbors (KNN)

# 4\. Gaussian Naive Bayes

# 5\. Random Forest (Ensemble)

# 6\. XGBoost (Ensemble)

# 

# Evaluation Metrics Used:

# \- Accuracy

# \- AUC Score

# \- Precision

# \- Recall

# \- F1 Score

# \- Matthews Correlation Coefficient (MCC)

# 

# ---

# 

# \## Model Comparison Table

# 

# ML Model Name	Acuuracy	AUC	Precision	Recall	F1	MCC

# Logistic Regression	0.9649	0.996	0.975	0.9286	0.9512	0.9245

# Decision Trees	0.9298	0.9246	0.9048	0.9048	0.9048	0.8492

# kNN	0.9561	0.9825	0.9744	0.9048	0.9383	0.9058

# Naïve Bayes	0.9386	0.9934	1	0.8333	0.9091	0.8715

# Random Forest (Ensemble)	0.9737	0.9967	1	0.9286	0.963	0.9442

# XGBoost (Ensemble)	0.9737	0.9927	1	0.9286	0.963	0.9442



# 

# 

# 

# 

# 

# 

# 

# 

# | ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |

# |----------|----------|-----|-----------|--------|----------|-----|

# | Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |

# | Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |

# | KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |

# | Gaussian NB | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |

# | Random Forest | 0.9737 | 0.9967 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

# | XGBoost | 0.9737 | 0.9927 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

# 

# ---

# 

# \## Observations on Model Performance

# 

# | ML Model | Observation |

# |----------|------------|

# | Logistic Regression | Performed extremely well with high AUC (0.9960), indicating strong class separability. Suitable due to near-linear separability of the dataset. |

# | Decision Tree | Showed lower performance compared to other models. Likely overfitting to training data and lacking ensemble stability. |

# | KNN | Achieved strong performance with high precision. Performance depends heavily on feature scaling and choice of K. |

# | Gaussian Naive Bayes | Achieved perfect precision (1.0) but lower recall (0.8333), indicating it missed some malignant cases. |

# | Random Forest | Achieved the best overall performance with highest AUC (0.9967) and MCC (0.9442). Demonstrates strength of ensemble averaging and reduced overfitting. |

# | XGBoost | Matched Random Forest in overall accuracy and F1 score. Strong boosting performance with effective regularization. |

# 

# ---

# 

# \## Final Conclusion

# 

# Ensemble models (Random Forest and XGBoost) performed best overall.

# Random Forest achieved the highest AUC and demonstrated strong generalization capability.

# 

# Given the medical nature of the dataset, high recall for malignant cases is critical. Both Random Forest and XGBoost maintained high recall (0.9286) while also achieving perfect precision.

# 

# ---

# 

# \## Streamlit Application Features

# 

# \- CSV dataset upload

# \- Model selection dropdown

# \- Evaluation metric display

# \- Confusion matrix visualization

# \- Classification report output

# 

# ---

# 

# \## How to Run Locally

# 

# pip install -r requirements.txt

# streamlit run app.py

# 

# ---

# 

# \## Deployment

# 

# Streamlit App Link: <Paste Your Link>

# GitHub Repository Link: <Paste Your Link>

#  Machine Learning Assignment – 2

# \## Classification Model Comparison and Deployment using Streamlit

# 

# ---

# 

# \## a) Problem Statement

# 

# The objective of this project is to implement and compare multiple machine learning classification models on a medical diagnostic dataset.

# 

# The goal is to:

# \- Predict whether a tumor is Benign (B) or Malignant (M)

# \- Compare six different classification algorithms

# \- Evaluate them using multiple performance metrics

# \- Deploy an interactive Streamlit web application

# 

# This project demonstrates an end-to-end machine learning workflow including model training, evaluation, comparison, and deployment.

# 

# ---

# 

# \## b) Dataset Description

# 

# Dataset Name: Breast Cancer Wisconsin Dataset

# Source: UCI Machine Learning Repository

# 

# Shape: (569, 32)

# 

# After removing the `id` column, 30 numerical features were used for modeling.

# 

# Target Variable:

# \- B (Benign) → 0

# \- M (Malignant) → 1

# 

# Train-Test Split:

# \- Training Set: (455, 30)

# \- Test Set: (114, 30)

# 

# Class Distribution:

# \- Train: \[285 Benign, 170 Malignant]

# \- Test: \[72 Benign, 42 Malignant]

# 

# The dataset contains computed features from digitized images of breast mass cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

# 

# ---

# 

# \## c) Models Used and Evaluation Metrics

# 

# The following classification models were implemented:

# 

# 1\. Logistic Regression

# 2\. Decision Tree Classifier

# 3\. K-Nearest Neighbors (KNN)

# 4\. Gaussian Naive Bayes

# 5\. Random Forest (Ensemble)

# 6\. XGBoost (Ensemble)

# 

# Evaluation Metrics Used:

# \- Accuracy

# \- AUC Score

# \- Precision

# \- Recall

# \- F1 Score

# \- Matthews Correlation Coefficient (MCC)

# 

# ---

# 

# \## Model Comparison Table

# 

# | ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |

# |----------|----------|-----|-----------|--------|----------|-----|

# | Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |

# | Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |

# | KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |

# | Gaussian NB | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |

# | Random Forest | 0.9737 | 0.9967 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

# | XGBoost | 0.9737 | 0.9927 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

# 

# ---

# 

# \## Observations on Model Performance

# 

# | ML Model | Observation |

# |----------|------------|

# | Logistic Regression | Performed extremely well with high AUC (0.9960), indicating strong class separability. Suitable due to near-linear separability of the dataset. |

# | Decision Tree | Showed lower performance compared to other models. Likely overfitting to training data and lacking ensemble stability. |

# | KNN | Achieved strong performance with high precision. Performance depends heavily on feature scaling and choice of K. |

# | Gaussian Naive Bayes | Achieved perfect precision (1.0) but lower recall (0.8333), indicating it missed some malignant cases. |

# | Random Forest | Achieved the best overall performance with highest AUC (0.9967) and MCC (0.9442). Demonstrates strength of ensemble averaging and reduced overfitting. |

# | XGBoost | Matched Random Forest in overall accuracy and F1 score. Strong boosting performance with effective regularization. |

# 

# ---

# 

# \## Final Conclusion

# 

# Ensemble models (Random Forest and XGBoost) performed best overall.

# Random Forest achieved the highest AUC and demonstrated strong generalization capability.

# 

# Given the medical nature of the dataset, high recall for malignant cases is critical. Both Random Forest and XGBoost maintained high recall (0.9286) while also achieving perfect precision.

# 

# ---

# 

# \## Streamlit Application Features

# 

# \- CSV dataset upload

# \- Model selection dropdown

# \- Evaluation metric display

# \- Confusion matrix visualization

# \- Classification report output

# 

# ---

# 

# \## How to Run Locally

# 

# pip install -r requirements.txt

# streamlit run app.py

# 

# ---

# 

# \## Deployment

# 

# Streamlit App Link: <Paste Your Link>

# GitHub Repository Link: <Paste Your Link>

)

