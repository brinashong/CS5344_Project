import wandb
import numpy as np
import os
import random
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

base_path = '/Users/suyeetan/Downloads/CS5344_Project/work'

def wandb_log(conf_matrix, class_report, acc_score):
    wandb.log({
        "Accuracy Score": acc_score
    })
        
    class_report_table = wandb.Table(columns=["class", "precision", "recall", "f1-score", "support"])
    
    for class_name, metrics in class_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            class_report_table.add_data(
                class_name, 
                metrics["precision"], 
                metrics["recall"], 
                metrics["f1-score"], 
                metrics["support"]
            )
    
    wandb.log({"Classification Report": class_report_table})
    
    wandb.log({
        "precision_avg": class_report["weighted avg"]["precision"],
        "recall_avg": class_report["weighted avg"]["recall"],
        "f1-score_avg": class_report["weighted avg"]["f1-score"]
    })

    conf_df = pd.DataFrame(conf_matrix)
    wandb.log({"Confusion Matrix": wandb.Table(dataframe=conf_df)})

def evaluate(y_test, predictions, heading='-----Evaluation-----'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    print(heading)

    # Confusion matrix
    axes[0].set_title('Confusion Matrix')
    cm = confusion_matrix(y_test, predictions)
    y_test = np.ravel(y_test)
    predictions = np.ravel(predictions)
    categories = np.unique(np.concatenate((y_test, predictions)))

    df_cm = pd.DataFrame(cm, index = [i for i in categories], columns = [i for i in categories])
    sns.heatmap(df_cm, annot=True, cmap='Reds', ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Classification report
    axes[1].set_title('Classification Report')
    cr = classification_report(y_test, predictions, output_dict=True)
    sns.heatmap(pd.DataFrame(cr).iloc[:-1, :].T, annot=True, ax=axes[1], fmt='.3f')

    # Display the subplots
    plt.tight_layout()
    plt.show()

    acc = accuracy_score(y_test, predictions)
    print("Accuracy:", acc)
    wandb_log(cm, cr, acc)

def remove_files_from_directory(directory):
    # Get all files in the directory
    files = glob.glob(os.path.join(directory, "*"))
    
    # Loop through the files and remove each one
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
    
    print(f"All files in {directory} have been removed.")

def get_minor_X_y_from_csv(csv_file, main_labels, target_column, normal_target, output_folder):
    df = pd.read_csv(os.path.join(output_folder, csv_file),usecols=main_labels)
    df = df.fillna(0)
    minor_or_not=[]
    for i in df[target_column]:
        if i == normal_target:
            minor_or_not.append(1)
        else:
            minor_or_not.append(0)           
    df[target_column]=minor_or_not

    y_df = df[target_column]
    X_df = df.drop(columns=[target_column])
    
    return (X_df, y_df)

def process_csv_with_args(csv_file, main_labels, target_column, normal_target, numerical_columns, output_folder, top_features_num, scaler, model_name):
    print('Processing CSV file:', csv_file)

    X_df, y_df = get_minor_X_y_from_csv(csv_file, main_labels, target_column, normal_target, output_folder)

    try:
        # Compute feature importances
        forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        forest.fit(X_df, y_df)
        importances = forest.feature_importances_
        label = csv_file.split(".")[0]

        # Prepare important features DataFrame
        refclasscol = list(X_df.columns.values)
        impor_bars = pd.DataFrame({'Features': refclasscol[0:20], 'importance': importances[0:20]})
        impor_bars = impor_bars.sort_values('importance', ascending=False)
        important_features = impor_bars['Features'].to_list()[:top_features_num]
        impor_bars = impor_bars.set_index('Features')

        # Scale numerical columns
        X_scaled_df = X_df.copy()
        if len(numerical_columns) > 0:
            X_scaled_df[numerical_columns] = scaler.transform(X_scaled_df[numerical_columns])

        if model_name == "svm":
            model = SVC()
        elif model_name == "knn":
            model = KNeighborsClassifier(weights='distance', n_jobs=-1)
        else:
            raise Exception(f"{model_name} is not supported")
        
        column_indices = X_df.columns.get_indexer(important_features)
        X_train_class_scaled = X_scaled_df.iloc[:, column_indices]
        y_train_class = y_df

        if len(y_train_class) > 0:
            model.fit(X_train_class_scaled, y_train_class)
        else:
            print(f'No data for {label}')

        return label, important_features, model, impor_bars
    except ValueError as e:
        print(f'csv_file: {csv_file}, error: {e}')
        raise Exception()

def create_dataset_for_label(label, name, major, minor_type_dict, major_ratio, min_major_samples, all_df, target_index, TARGET_COLUMN, NORMAL_TARGET, OUTPUT_FOLDER, main_labels):
    minor_count, major_count = 0, 0
    
    output_path = os.path.join(OUTPUT_FOLDER, f"{name}.csv")
    with open(output_path, "w") as ths:
        ths.write(','.join(main_labels) + "\n")
        
        # Calculate the number of major samples based on the fixed ratio
        minor_dict_count = minor_type_dict[label]
        major_num = max(min(int(minor_dict_count * major_ratio), major), min_major_samples)
        
        # Collect normal (major) rows and minor rows
        major_rows = []
        minor_rows = []

        # Read all_data.csv line by line and collect rows
        with open("all_data.csv", "r") as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue  # Skip the header row
                k = line.strip().split(",")  # Strip newline and split the line
                
                # Collect normal rows
                if int(k[target_index]) == NORMAL_TARGET:
                    major_rows.append(line)
                
                # Collect minor rows that match the current label
                elif int(k[target_index]) == label:
                    minor_rows.append(line)

        # Randomly sample major rows
        if len(major_rows) > major_num:
            major_rows = random.sample(major_rows, major_num)
        else:
            major_rows = random.sample(major_rows, len(major_rows))  # Shuffle if fewer than required

        # Concatenate major and minor rows
        combined_rows = major_rows + minor_rows
        
        # Shuffle the combined rows
        random.shuffle(combined_rows)

        # Write the shuffled rows to the output file
        for row in combined_rows:
            ths.write(row)

        # Print number of rows written
        major_count = len(major_rows)
        minor_count = len(minor_rows)
        print(f"{name}.csv created with {major_count + minor_count} rows. ({major_count} major and {minor_count} minor rows)")
    return name

def get_dataset_for_label(label, name, target_index, NORMAL_TARGET, OUTPUT_FOLDER, main_labels):
    minor_count, major_count = 0, 0  # Track minor and major sample counts
    
    output_path = os.path.join(OUTPUT_FOLDER, f"{name}.csv")
    with open(output_path, "w") as ths:
        ths.write(','.join(main_labels) + "\n")
        
        # Collect normal (major) rows and minor rows
        major_rows = []
        minor_rows = []

        # Read all_data.csv line by line and collect rows
        with open("all_data.csv", "r") as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue  # Skip the header row
                k = line.strip().split(",")  # Strip newline and split the line
                
                # Collect normal rows
                if int(k[target_index]) == NORMAL_TARGET:
                    major_rows.append(line)
                
                # Collect minor rows that match the current label
                elif int(k[target_index]) == label:
                    minor_rows.append(line)

        # Concatenate major and minor rows
        combined_rows = major_rows + minor_rows
        
        # Shuffle the combined rows
        random.shuffle(combined_rows)

        # Write the shuffled rows to the output file
        for row in combined_rows:
            ths.write(row)

        # Print number of rows written
        major_count = len(major_rows)
        minor_count = len(minor_rows)
        print(f"{name}.csv created with {major_count + minor_count} rows. ({major_count} major and {minor_count} minor rows)")
    return name

# Preprocessing

def one_hot_encode(df, categorical_columns):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    X_encoded = df.copy()
    encoded_dfs = []
    
    for col in categorical_columns:
        encoded_array = ohe.fit_transform(X_encoded[[col]])
        encoded_columns = ohe.get_feature_names_out([col])
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=X_encoded.index)
        encoded_dfs.append(encoded_df)
    
    X_encoded = X_encoded.drop(columns=categorical_columns)
    X_encoded = pd.concat([X_encoded] + encoded_dfs, axis=1)
    X_encoded = X_encoded.copy()
    return (ohe, X_encoded)

def label_encode(df, columns):
    le = LabelEncoder()
    X_encoded = df.copy()
    for col in columns:
        X_encoded[col] = le.fit_transform(X_encoded[col])
    return (le, X_encoded)

def standardise(df, columns, scaler=None):
    X_standardised = df.copy()
    if not columns:
        return None, X_standardised
    if not scaler:
        scaler = StandardScaler()
        X_standardised[columns] = scaler.fit_transform(X_standardised[columns])
    else:
        X_standardised[columns] = scaler.transform(X_standardised[columns])
    return (scaler, X_standardised)

# Display

def show_missing_values(all_df):
    plt.figure(figsize=(12,4))
    sns.heatmap(all_df.isnull(),cbar=False,cmap='Wistia',yticklabels=False)
    plt.title('Missing value in the dataset')

def show_boxplots(all_df):
    l = all_df.columns.values
    number_of_columns = 5
    number_of_rows = len(l)//number_of_columns
    plt.figure(figsize=(20, 5 * number_of_rows))
    for i in range(0,len(l)):
        plt.subplot(number_of_rows + 1, number_of_columns, i+1)
        sns.set_style('whitegrid')
        sns.boxplot(all_df[l[i]], color='green', orient='v')
        plt.tight_layout()

def show_distribution_graph(dist_df, dist_col):
    max_columns = 5
    number_of_rows = (len(dist_col) + max_columns - 1) // max_columns 
    plt.figure(figsize=(3 * max_columns, 5 * number_of_rows))

    for i in range(len(dist_col)):
        plt.subplot(number_of_rows, max_columns, i + 1) 
        
        sns.histplot(dist_df[dist_col[i]], kde=True)
        
        plt.title(dist_col[i])
        plt.xlabel(dist_col[i])
        plt.ylabel('Density')

    plt.tight_layout()
    plt.show()

def show_target_values(all_df, target_column):
    target_counts = all_df[target_column].value_counts()

    fig, ax = plt.subplots(1, 2, figsize=(15,7))
    target_counts_barplot = sns.barplot(x = target_counts.index,y = target_counts.values, ax = ax[0], hue=target_counts.index, palette='Set2', legend=False)
    target_counts_barplot.set_ylabel('Number of classes in the dataset')
    
    target_counts.plot.pie(autopct="%1.1f%%", ax=ax[1])

def show_feature_correlation(all_df):
    plt.figure(figsize=(18, 15))
    sns.heatmap(all_df.corr(), cmap='coolwarm')

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
