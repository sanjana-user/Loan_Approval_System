Problem Statement: 
We need efficient systems to assess loan applications and minimize default risk. The aim of this project is to develop a classification model that predicts whether a loan application should be approved or rejected based on applicants’ financial and demographic information. 
Dataset:
The dataset is taken from Kaggle, which includes loan application records with applicants’ financial and demographic details such as income, credit history, loan amount, and employment status. The target variable indicates whether a loan was approved or not.
Rows: 614
Features: Income, Credit History, Loan Amount, Education, Marital Status
Target Variable: Loan_Status. 
Tools:
Pandas
Numpy 
Matplotlib
Seaborn 
Scikit Learn
Approach and Methodology:
a) Missing/ Null values: 
Removed all the null values. 
For missing values, binary/ fixed output valued columns like gender, credit history, self employed and dependents are replaced with their respective modes. For continuous outputs like loan amount and loan amount term are replaced with their means. 
b) Feature Engineering: 
All the data is converted to numerical (for easy calculations)
c) Train Test split:
The data is split into training and testing data using the Scikitlearn function, train_test_split().
Training the data using the random forest classifier model, a test data accuracy of up to 76% is obtained.
d) Feature Importance: 
Creating a new table, with features and their importances (calculated using feature_importances_ attribute of random forest), a bar graph visualised their importances.
Models used:
Random Forest Classifier: Ensemble model, it combines several decision tree models to get stable predictions. Instead of relying on a single decision tree, a ‘forest’ of multiple trees are built, each trained on a random subset of the data and the features reducing overfitting.


Evaluation Metrics:
 Confusion Matrix: 
It compares the model predicted values and the actual known values in a tabular way. 
Confusion Matrix for our model:
[ [12,29],
  [0, 82]  ]
Accuracy: 
It calculates the proportion of correct predictions. For example, if a model correctly predicts 90 out of 100 instances, the accuracy is 0.9. Our model’s accuracy is 76.42%.
Insights:
The model successfully classified loan applications as approved or rejected with decent accuracy.
Credit history and applicant income were among the most important factors influencing approval.
Random Forest performed better than simpler models due to its ability to handle complex patterns.
How to run the project:
Download or clone the repository.
Open the project folder.
Open the Jupyter Notebook file (.ipynb).
Run all the cells from top to bottom.

Project Structure: 
├── loan_dataset.csv        # Dataset used in the project
├── loan.approval.ipynb      # Jupyter Notebook with full code
├── README.md          # Project explanation
