import wandb
import numpy as np
import os
import random
import glob
import pandas as pd
import time
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

def evaluate(y_test, predictions, heading='-----Evaluation-----'):
    print(heading)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    plt.figure(figsize=(15,10))
    categories = np.unique(y_test)
    df_cm = pd.DataFrame(cm, index = [i for i in categories], columns = [i for i in categories])
    sns.heatmap(df_cm,annot=True,cmap='Reds')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    cr = classification_report(y_test, predictions, output_dict=True)
    print("\nClassification Report:")
    print(cr)
    acc = accuracy_score(y_test, predictions)
    print("Accuracy:", acc)
    return (cm, cr, acc)

def remove_files_from_directory(directory):
    # Get all files in the directory
    files = glob.glob(os.path.join(directory, "*"))
    
    # Loop through the files and remove each one
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
    
    print(f"All files in {directory} have been removed.")

def get_anomaly_X_y_from_csv(csv_file, main_labels, target_column, normal_target, output_folder):
    df = pd.read_csv(os.path.join(output_folder, csv_file),usecols=main_labels)
    df = df.fillna(0)
    anomaly_or_not=[]
    for i in df[target_column]: #it changes the normal label to "1" and the anomaly tag to "0" for use in the machine learning algorithm
        if i == normal_target:
            anomaly_or_not.append(1)
        else:
            anomaly_or_not.append(0)           
    df[target_column]=anomaly_or_not

    # y = df[target_column].values
    # del df[target_column]
    # X = df.values
    y_df = df[target_column]
    X_df = df.drop(columns=[target_column])
    
    # X = np.float32(X)
    # X[np.isnan(X)] = 0
    # X[np.isinf(X)] = 0
    # print('X', type(X), X)
    # print('y', type(y), y)
    return (X_df, y_df, df)

def process_csv(csv_file, main_labels, target_column, normal_target, numerical_columns, output_folder, scaler):
    print('Processing CSV file:', csv_file)

    X_df, y_df, df = get_anomaly_X_y_from_csv(csv_file, main_labels, target_column, normal_target, output_folder)

    try:
        # Compute feature importances
        forest = RandomForestRegressor(n_estimators=250, random_state=0)
        forest.fit(X_df, y_df)
        importances = forest.feature_importances_
        label = csv_file.split(".")[0]

        # Prepare important features DataFrame
        refclasscol = list(df.columns.values)
        impor_bars = pd.DataFrame({'Features': refclasscol[0:20], 'importance': importances[0:20]})
        impor_bars = impor_bars.sort_values('importance', ascending=False)
        important_features = impor_bars['Features'].to_list()[:5]
        impor_bars = impor_bars.set_index('Features')

        # Scale numerical columns
        X_scaled_df = X_df.copy()
        X_scaled_df[numerical_columns] = scaler.transform(X_scaled_df[numerical_columns])

        # Fit SVC if there are samples for the class
        svm = SVC()
        knn = KNeighborsClassifier(n_neighbors=5)
        decision_tree = DecisionTreeClassifier()
        random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        logistic_regression = LogisticRegression(max_iter=1000)
        gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)
        voting_clf = VotingClassifier(estimators=[
            ('svm', SVC()),
            ('knn', knn),
            ('dt', decision_tree),
            ('rf', random_forest),
            ('lr', logistic_regression),
            ('gb', gradient_boosting)
            ], voting='hard')
        
        column_indices = df.columns.get_indexer(important_features)
        # print('column_indices', column_indices, df.columns)
        X_train_class = df.iloc[:, column_indices]
        X_train_class_scaled = X_scaled_df.iloc[:, column_indices]
        y_train_class = y_df

        if len(y_train_class) > 0:
            # svm.fit(X_train_class, y_train_class)
            svm.fit(X_train_class_scaled, y_train_class)
            voting_clf.fit(X_train_class_scaled, y_train_class)
        else:
            print(f'No data for {label}')

        return label, important_features, svm, impor_bars, voting_clf
    except ValueError as e:
        print(f'csv_file: {csv_file}, error: {e}')

# Preprocessing

def one_hot_encode(df, categorical_columns):
    # Initialize the OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Create a copy of the DataFrame for encoding
    X_encoded = df.copy()
    
    # List to store the one-hot encoded DataFrames
    encoded_dfs = []
    
    # Loop over categorical columns to encode them
    for col in categorical_columns:
        # Fit and transform the column with one-hot encoding
        encoded_array = ohe.fit_transform(X_encoded[[col]])
        
        # Create a DataFrame for the one-hot encoded columns
        encoded_columns = ohe.get_feature_names_out([col])
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=X_encoded.index)
        
        # Append the encoded DataFrame to the list
        encoded_dfs.append(encoded_df)
    
    # Drop the original categorical columns from the DataFrame
    X_encoded = X_encoded.drop(columns=categorical_columns)
    
    # Concatenate the original DataFrame (without categorical columns) with the encoded DataFrames
    X_encoded = pd.concat([X_encoded] + encoded_dfs, axis=1)
    
    # Ensure the DataFrame is de-fragmented by making a copy
    X_encoded = X_encoded.copy()
    
    # print(list(X_encoded.columns))
    return (ohe, X_encoded)

def label_encode(df, columns):
    le = LabelEncoder()
    X_encoded = df.copy()
    for col in columns:
        X_encoded[col] = le.fit_transform(X_encoded[col])
    return (le, X_encoded)

def standardise(df, columns, scaler=None):
    X_standardised = df.copy()
    
    if not scaler:
        scaler = StandardScaler()
        # Fit and transform the numeric columns
        X_standardised[columns] = scaler.fit_transform(X_standardised[columns])
    else:
        X_standardised[columns] = scaler.transform(X_standardised[columns])
    return (scaler, X_standardised)

# Display

def show_missing_values(all_df):
    plt.figure(figsize=(12,4))
    sns.heatmap(all_df.isnull(),cbar=False,cmap='Wistia',yticklabels=False)
    plt.title('Missing value in the dataset');

def show_target_values(all_df, target_column):
    target_counts = all_df[target_column].value_counts()

    fig, ax = plt.subplots(1, 2, figsize=(15,7))
    target_counts_barplot = sns.barplot(x = target_counts.index,y = target_counts.values, ax = ax[0], hue=target_counts.index, palette='Set2', legend=False)
    target_counts_barplot.set_ylabel('Number of classes in the dataset')
    
    target_counts.plot.pie(autopct="%1.1f%%", ax=ax[1])

def show_feature_correlation(all_df):
    plt.figure(figsize=(20,15))
    sns.heatmap(all_df.corr(), cmap='hsv')

def show_feature_importance(impor_bars, label, feature_folder):
    plt.rcParams['figure.figsize'] = (10, 5)
    impor_bars.plot.bar()
    count = 0
    feature = label+"=["
    for i in impor_bars.index:
        feature = feature+"\""+str(i)+"\","
        count += 1
        if count == 5:
            feature = feature[0:-1]+"]"
            break     
    print(label,"importance list:")
    print(label,"\n",impor_bars.head(20),"\n\n\n")
    print(feature)
    plt.title(label+" Cover type - Feature Importance")
    plt.ylabel('Importance')
    plt.savefig(os.path.join(feature_folder, label+".pdf"),bbox_inches='tight', format = 'pdf')
    plt.tight_layout()
    plt.show()

def wandb_log(conf_matrix, class_report, acc_score):
    wandb.log({
        "Accuracy Score": acc_score
    })
        
    # Create a table for classification metrics
    class_report_table = wandb.Table(columns=["class", "precision", "recall", "f1-score", "support"])
    
    # Populate the table
    for class_name, metrics in class_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:  # Skip overall avg metrics
            class_report_table.add_data(
                class_name, 
                metrics["precision"], 
                metrics["recall"], 
                metrics["f1-score"], 
                metrics["support"]
            )
    
    # Log the table to WandB
    wandb.log({"Classification Report": class_report_table})
    
    # You can also log the metrics separately if needed (for overall comparison/graphing)
    wandb.log({
        "precision_avg": class_report["weighted avg"]["precision"],
        "recall_avg": class_report["weighted avg"]["recall"],
        "f1-score_avg": class_report["weighted avg"]["f1-score"]
    })

    # Convert confusion matrix into a DataFrame for better clarity
    conf_df = pd.DataFrame(conf_matrix)
    wandb.log({"Confusion Matrix": wandb.Table(dataframe=conf_df)})