# junior-data-engineer-labs
Python-based lab projects covering data cleaning, analysis, visualization, and modeling.

# Lab 02 – Poverty Data Analysis

This lab focuses on analyzing global poverty indicators using Python. The goal was to understand the relationships between features such as life expectancy, education level, and income, and apply basic ETL and modeling techniques.

## What I did

- Loaded and combined two poverty-related datasets using `pandas`
- Checked and cleaned missing and duplicate values
- Performed data exploration and descriptive statistics
- Scaled features using `MinMaxScaler` and `StandardScaler`
- Visualized distributions and correlations using `seaborn`
- Built a simple linear regression model using `statsmodels`

## Tools used

- Python
- pandas
- numpy
- seaborn
- scikit-learn
- statsmodels

## Skills demonstrated

- Data cleaning and preprocessing (ETL)
- Exploratory data analysis (EDA)
- Feature scaling
- Simple regression modeling

# Lab 03 – Data Cleaning and Visualization

This lab focuses on exploratory data analysis (EDA) and basic data cleaning. The dataset includes various indicators which were analyzed through visualizations and summary statistics.

## What I did

- Loaded and cleaned an Excel dataset using `pandas`
- Checked for missing values and duplicate entries
- Created correlation heatmaps and distribution plots using `seaborn`
- Visualized data with boxplots, histograms, and pair plots
- Identified variable relationships and outliers

## Tools used

- Python
- pandas
- seaborn
- matplotlib

## Skills demonstrated

- Data cleaning
- Exploratory data analysis (EDA)
- Correlation analysis
- Data visualization


# Lab 04 – Outlier Detection and Feature Scaling

This lab focuses on detecting outliers and applying feature scaling techniques in preparation for data modeling.

## What I did

- Visualized outliers using boxplots
- Identified and removed outliers using the Z-score method
- Applied feature scaling:
  - Min-max normalization
  - Standardization (Z-score scaling)
- Used seaborn and matplotlib for visual comparison of distributions

## Tools used

- Python
- pandas
- numpy
- seaborn
- matplotlib
- scipy

## Skills demonstrated

- Data preprocessing
- Outlier detection and removal
- Feature scaling (MinMax, Z-score)
- Data visualization

# Lab 05 – Regression Modeling and Feature Selection

This lab focuses on building linear regression models using real-world poverty and development datasets. It involves data integration, cleaning, feature scaling, and multiple model comparisons to understand predictors of MPI Urban.

## What I did

- Merged two datasets: Poverty indicators and MPI index data
- Dropped irrelevant or duplicate columns
- Performed feature scaling (MinMax and StandardScaler)
- Conducted correlation analysis and visualizations
- Built several linear regression models:
  - Simple linear regression: `mpi_urban ~ child_mort`
  - Full feature multiple regression
  - Reduced models to minimize multicollinearity
- Evaluated models using R² and Adjusted R²
- Visualized regression and residual plots using `statsmodels`

## Tools used

- Python
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- statsmodels

## Skills demonstrated

- Data integration and preprocessing
- Feature scaling
- Linear regression modeling
- Feature selection
- Model evaluation (R², adjusted R²)
- Visual analytics

- # Lab 06 – Linear Regression with Train/Test Split

This lab focuses on building and evaluating a supervised linear regression model using poverty data. The model is trained and tested using scikit-learn, and performance is assessed using R² and MSE.

## What I did

- Selected relevant features and target variables
- Split the dataset into training and testing sets using `train_test_split`
- Built a linear regression model with `sklearn.linear_model`
- Trained the model on the training set
- Predicted and evaluated performance on the test set using:
  - R² score
  - Mean Squared Error (MSE)
- Visualized predictions and residuals

## Tools used

- Python
- pandas
- seaborn
- matplotlib
- scikit-learn

## Skills demonstrated

- Supervised learning workflow
- Linear regression with train/test split
- Model evaluation and error metrics
- Prediction visualization


# Lab 07 – K-Nearest Neighbors Classification

This lab applies the KNN algorithm to the Iris dataset to find the optimal k-value and evaluate classification performance.

## What I did

- Checked class balance of the dataset
- Iteratively trained KNN classifiers for K=1 to 30
- Plotted training vs testing accuracy to find the best k
- Created a decision region plot using the best k

## Tools used

- Python
- scikit-learn
- matplotlib

## Skills demonstrated

- KNN classification
- Model evaluation and selection
- Decision boundary visualization


# Lab 08 – Decision Tree Tuning

This lab involves tuning decision tree parameters (max_leaf_nodes) to improve classification performance using scikit-learn.

## What I did

- Tested max_leaf_nodes from 2 to 10
- Identified the best configuration with highest cross-validated accuracy
- Built and visualized the final decision tree
- Evaluated using accuracy, precision, and recall

## Tools used

- Python
- scikit-learn
- matplotlib

## Skills demonstrated

- Decision tree classification
- Hyperparameter tuning
- Model evaluation metrics


# Lab 09 – Decision Tree Classification on Car Data

This lab builds and evaluates a decision tree classifier for car evaluation data.

## What I did

- Trained a decision tree classifier on labeled car data
- Computed and interpreted confusion matrix
- Calculated accuracy, precision, recall, and f1-score for each class
- Analyzed model performance class by class

## Tools used

- Python
- scikit-learn
- pandas

## Skills demonstrated

- Decision tree modeling
- Model evaluation with confusion matrix
- Multi-class classification analysis

- # Lab 10 – Evaluation Metrics Calculation

This lab focuses on calculating classification evaluation metrics by hand, using provided counts of true positives, false positives, true negatives, and false negatives.

## What I did

- Manually computed:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Analyzed how these metrics apply in different classification scenarios

## Tools used

- Manual calculation (no coding)

## Skills demonstrated

- Understanding of core evaluation metrics
- Metric interpretation in classification


# Lab 11 – Feature Selection with Chi-square, ANOVA and Tree Importance

This lab compares different feature selection techniques on a classification problem.

## What I did

- Analyzed categorical feature influence using Chi-square
- Compared Chi-square results with decision tree feature importances
- Computed ANOVA F-scores to rank numerical features
- Selected top 6 features and visualized their distributions

## Tools used

- Python
- scikit-learn
- seaborn
- matplotlib

## Skills demonstrated

- Feature selection (Chi-square, ANOVA, Gini)
- Feature ranking and comparison
- Visual data exploration
