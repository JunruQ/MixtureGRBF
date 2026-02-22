import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import re
import warnings; warnings.filterwarnings('ignore')

# Load data (unchanged)
data_1_path = 'data/ClinicalLabData.csv'
df_1 = pd.read_csv(data_1_path)

data_2_path = 'data/prot_Modifiable_bl_data.csv'
df_2 = pd.read_csv(data_2_path)

data_cov_path = 'data/ukb_cov_info.csv'
df_cov = pd.read_csv(data_cov_path)
df_cov = df_cov[['eid', 'sex', 'age', 'education', 'Ethnic2']]

df= pd.merge(df_1, df_cov, how='left', on='eid')
df= pd.merge(df, df_2, how='left', on='eid')

exp_name = 'ukb_MixtureGRBF_test_biom17'
nsubtype = 4
subtype_path = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
df_subtype = pd.read_csv(subtype_path)

df = df.merge(df_subtype, how='left', left_on='eid', right_on='PTID')
df.drop(columns=['PTID', 'stage'], inplace=True)

def clean_feature_name(name):
    return re.sub(r'[\\,\s{}\[\]"]', '_', name)

df.columns = [clean_feature_name(col) for col in df.columns]

# Split into train/validation and test sets
train_val_df = df[df['subtype'].notna()]
test_df = df[df['subtype'].isna()]

# Prepare data
X_train_val = train_val_df.drop(columns=['eid', 'subtype'])
y_train_val = train_val_df['subtype'] - 1  # 0-based adjustment
X_test = test_df.drop(columns=['eid', 'subtype'])

# Handle missing values
for col in X_train_val.columns:
    if X_train_val[col].isnull().any():
        median_val = X_train_val[col].median()
        X_train_val[col].fillna(median_val, inplace=True)
        if col in X_test.columns:
            X_test[col].fillna(median_val, inplace=True)

# Feature scaling
scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val)
X_test_scaled = scaler.transform(X_test)

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val_scaled, 
    y_train_val, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train_val
)

# Define classifiers
classifiers = {
    'LightGBM': lgb.LGBMClassifier(
        objective='multiclass',
        num_class=nsubtype,
        random_state=42,
        verbose=-1
    ),
    'Logistic Lasso': LogisticRegression(
        penalty='l1',
        solver='liblinear',
        multi_class='auto',
        random_state=42,
        max_iter=1000
    ),
    'Gaussian Process': GaussianProcessClassifier(
        kernel=RBF(),
        random_state=42,
        n_jobs=-1
    ),
    'RBF SVM': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        random_state=42
    )
}

# Train and evaluate each classifier
results = {}
print("Dataset Shapes:")
print("Training Set Shape:", X_train.shape)
print("Validation Set Shape:", X_val.shape)
print("\nClassifier Results:")
print("-" * 50)

for name, clf in classifiers.items():
    # Train the model
    clf.fit(X_train, y_train)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    
    # Store results
    results[name] = {'train_accuracy': train_acc, 'val_accuracy': val_acc}
    
    # Print results
    print(f"\n{name}:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(classification_report(y_val, clf.predict(X_val)))
    

# Print summary table
print("\n" + "-" * 50)
print("Summary of Results:")
print("-" * 50)
print("Classifier".ljust(20) + "Train Acc".ljust(15) + "Val Acc")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name.ljust(20)}{metrics['train_accuracy']:.4f}".ljust(35) + f"{metrics['val_accuracy']:.4f}")

# Note: Test set prediction is commented out but can be added for any selected classifier
"""
# Example for predicting with best classifier (e.g., LightGBM)
best_clf = classifiers['LightGBM']
y_pred_test = best_clf.predict(X_test_scaled)
y_pred_test_class = [x + 1 for x in y_pred_test]
test_df['predicted_subtype'] = y_pred_test_class
print("\nTest set predictions (first 5 rows):")
print(test_df.head())
"""