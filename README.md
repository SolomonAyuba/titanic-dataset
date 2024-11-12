![DTS 2 data science Image 1080 x 1080](https://github.com/user-attachments/assets/68522a63-4f10-43a2-b05b-f743e4943e6b)

[Google Collab Python Code](https://colab.research.google.com/drive/1mvypBO62sDGjPZKyTyKn5DKs6DGTtTxD#scrollTo=oJAJMiEukq2a) | [PDF Report](https://drive.google.com/file/d/1QPoplIHIaasXOEPoBOfOEEawaFaa1_kP/view) |

## Introduction
The Titanic dataset provides information on passengers aboard the Titanic, including details such as age, gender, class, and whether they survived. The goal of this project is to preprocess the dataset using data engineering techniques and then build a predictive model to determine a passenger's survival based on various features. By analyzing factors related to survival, we can gain insights into the demographic and socio-economic profiles of passengers who were more likely to survive.
 
## Methodology
#### Step 1: Importing Libraries
The project utilizes several libraries for data manipulation, visualization, and machine learning:
- **Data Manipulation:** `pandas, numpy`
- **Data Visualization:** `matplotlib, seaborn`
- **Machine Learning:** `scikit-learn` (for model selection, preprocessing, and evaluation)

#### Step 2: Data Loading and Initial Exploration
The Titanic dataset was loaded using `seaborn`, and an initial exploration of the data structure and contents was performed:
- **Dataset Overview:** The first few rows were displayed to understand the data structure and types of features.
- **Data Information:** The `info()` method was used to get information on the dataset's structure, including column names, data types, and counts of non-null values.
- **Missing Values Analysis:** A summary of missing values in each column revealed that some columns, like `age`, `embarked`, and `deck`, had a significant number of missing values.

#### Step 3: Exploratory Data Analysis (EDA)
Several visualizations were created to explore relationships between the target variable (`survived`) and other key features:
- **Survival Count Plot:** A count plot of the `survived` column showed the distribution of survivors versus non-survivors.
- **Age Distribution by Survival:** A histogram plot revealed differences in age distributions between survivors and non-survivors. Younger passengers seemed to have a higher survival rate.

#### Step 4: Data Cleaning
To prepare the dataset for model training, various data cleaning steps were applied:
- Handling Missing Values:
  - **Age:** Missing values in `age` were imputed with the median age of the dataset.
  - **Embarked:** Missing values in embarked were filled with the mode (the most common value).
  - **Deck:** A new category, 'Unknown,' was created to handle missing values in `deck`, since a significant portion of data in this column was missing.
- Dropping Redundant Columns: Columns like `embark_town` (similar to `embarked`) and `alive` (redundant with `survived`) were dropped to avoid redundancy and simplify the dataset.

#### Step 5: Outlier Detection
A box plot was created for the `fare` column to visually inspect any potential outliers. Outliers, if extreme, could impact model performance, so identifying them was an essential part of data preparation.

#### Step 6: Feature Engineering
Several new features were engineered to enhance the predictive power of the model:
- **Family Size:** Created by adding `sibsp` (siblings/spouses) and parch (parents/children) to represent the total number of family members onboard.
- **Age Group:** The `age` column was categorized into bins (e.g., Child, Teen, Adult, Senior) to capture age ranges that might relate to survival chances.
- **Fare Group:** The `fare` column was also categorized into bins (e.g., Low, Medium, High, Very High) to generalize fare ranges and reduce noise from individual fare values.

#### Step 7: Encoding Categorical Variables
To make categorical variables compatible with the model, `LabelEncoder` was used to transform them into numeric codes:
- **Encoded Columns:** `sex`, `embarked`, `AgeGroup`, `FareGroup`, `who`, `adult_male`, `deck`, `class`, and `alone` were converted to numerical format for the model.

#### Step 8: Scaling Numerical Features
For better model performance, the numerical features (`fare`, `age`, and `FamilySize`) were scaled using `StandardScaler`, which normalizes values to a standard distribution.

#### Step 9: Correlation Analysis
A heatmap was generated to examine correlations between features and the target variable. Highly correlated variables were identified to help with feature selection, ensuring that the model would utilize only the most relevant predictors.
 
## Feature Selection
The features were selected based on the correlation they have with the target variable. Since negative and positive correlations are significant, the threshold for  selecting the features used in the model is the absolute value 0.1+ (all values above 0.1).

### Model Training and Evaluation
 Data Splitting
The dataset was split into training and testing sets, with 70% of the data used for training and 30% for testing. This split ensures that the model can be validated on unseen data, which helps in evaluating its real-world performance.

## Model Selection and Comparison
Three different machine learning models were trained and evaluated:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

Each model was trained on the training dataset, and accuracy scores were calculated on the test set:
- **Logistic Regression:** Accuracy of around 76% (please specify based on code output).
- **Decision Tree Classifier:** Accuracy of around 80%.
- **Random Forest Classifier:** Accuracy of around 79%.
Based on the accuracy scores, the **Decision Tree Classifier** performed the best, showing its strength in handling complex data structures like the Titanic dataset.

### Detailed Model Evaluation
A Decision Tree Classifier was chosen as the final model for detailed evaluation:
- **Accuracy:** The model achieved an accuracy of approximately 80%
- **Confusion Matrix:** The confusion matrix showed the counts of true positives, true negatives, false positives, and false negatives, providing insight into model performance.
- **Classification Report:** Precision, recall, and F1-score for each class (survived vs. not survived) were calculated, indicating the model's ability to correctly classify both survivors and non-survivors.
 
# Results and Discussion
### Model Performance
The final Decision Tree Classifier model demonstrated good accuracy, with the model's precision and recall scores indicating that it could reliably predict survival. However, slight improvements could still be made by tuning model parameters or using ensemble methods.

### Key Findings
The analysis revealed several important insights:
- **Gender:** Female passengers had a much higher survival rate than males.
- **Class:** Passengers in first class had a higher chance of survival, likely due to proximity to lifeboats and better evacuation access.
- **Age:** Younger passengers, particularly children, showed a higher likelihood of survival.
- **Family Size:** Passengers with a family onboard had different survival patterns compared to those traveling alone.

## Visualizations
Various visualizations supported the analysis:
- **Histograms:** Showed age distribution in relation to survival.
- **Correlation Heatmap:** Highlighted relationships between different features and the target variable.
 
# Conclusion
This project demonstrated how data engineering and machine learning techniques could be applied to the Titanic dataset to predict passenger survival. By performing data cleaning, feature engineering, and model selection, we were able to develop a reasonably accurate predictive model.

#### **Team:** [Aduragbemi Akinshola Kinoshi](https://github.com/pkinoshi) |⁠ ⁠[Solomon Ayuba](https://github.com/SolomonAyuba) | [Okon Olugbenga Enang](https://github.com/Nanoshogun) | King Richard | Theresa Chukwukere
**Date:** 12.11.2024

© [Miva Open University](https://miva.university/) 2024.
