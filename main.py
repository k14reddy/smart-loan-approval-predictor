import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("C:/Users/kotla/OneDrive/Desktop/Prime_notes/Loan_approval_project/loan_approval_data.csv")

numerical_cols = df.select_dtypes(include="number").columns
categorical_cols = df.select_dtypes(include="object").columns

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy= "most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

df = df.drop("Applicant_ID", axis=1)

le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)

df =  pd.concat([df.drop(columns=cols), encoded_df], axis=1)

# Feature Engineering (Bases on correlational heatmap)
# df["DTI_Ratio"] =  df["DTI_Ratio"] ** 2 
# df["Credit_Score"] = df["Credit_Score"] ** 2

#df["Applicant_Income"] = np.log1p(df["Applicant_Income"]) # if it skewed => log(1+x)

X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

X_train, X_test, y_train, y_test  =  train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_log_pred = log_model.predict(X_test_scaled)

print("Logistic Regression \n")

print("Precision: ", precision_score(y_test, y_log_pred))
print("Recall: ", recall_score(y_test, y_log_pred))
print("f1_score: ", f1_score(y_test, y_log_pred))
print("Accuracy: ", accuracy_score(y_test, y_log_pred))
print("CM: ", confusion_matrix(y_test, y_log_pred))

# kNN

kNN_model = KNeighborsClassifier(n_neighbors=5)
kNN_model.fit(X_train_scaled, y_train)

y_kNN_pred = kNN_model.predict(X_test_scaled)

print("kNN \n")

print("Precision: ", precision_score(y_test, y_kNN_pred))
print("Recall: ", recall_score(y_test, y_kNN_pred))
print("f1_score: ", f1_score(y_test, y_kNN_pred))
print("Accuracy: ", accuracy_score(y_test, y_kNN_pred))
print("CM: ", confusion_matrix(y_test, y_kNN_pred))

# Naive Bayes
NB_model = GaussianNB()
NB_model.fit(X_train_scaled, y_train)

y_NB_pred = NB_model.predict(X_test_scaled)

print("Naive Bayes \n")

print("Precision: ", precision_score(y_test, y_NB_pred))
print("Recall: ", recall_score(y_test, y_NB_pred))
print("f1_score: ", f1_score(y_test, y_NB_pred))
print("Accuracy: ", accuracy_score(y_test, y_NB_pred))
print("CM: ", confusion_matrix(y_test, y_NB_pred))

# Best Model is Naive Bayes

