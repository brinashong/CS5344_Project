import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_covtype
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.io import arff
import sklearn as sk
import time
from sklearn.datasets import fetch_openml

data = fetch_covtype(as_frame=True)  # Set as_frame=True to get the data as a DataFrame
X = data['data']
y = data['target']

# Combine features and target into one DataFrame
all_df = pd.concat([X, y], axis=1)
all_df.head()

def remove_files_from_directory(directory):
    # Get all files in the directory
    files = glob.glob(os.path.join(directory, "*"))
    
    # Loop through the files and remove each one
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
    
    print(f"All files in {directory} have been removed.")

normal_target = 2
output_folder = 'csv_files'
target_column = 'Cover_Type'
feature_folder = 'features'

# Create a folder to save the CSVs
remove_files_from_directory(output_folder)
remove_files_from_directory(feature_folder)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(feature_folder, exist_ok=True)

# List of target class names (cover types)
cover_types = {
    1: "Spruce-Fir",
    2: "LodgepolePine",
    3: "PonderosaPine",
    4: "CottonwoodWillow",
    5: "Aspen",
    6: "DouglasFir",
    7: "Krummholz"
}

inv_cover_types = {
    "Spruce-Fir": 1,
    "LodgepolePine": 2,
    "PonderosaPine": 3,
    "CottonwoodWillow": 4,
    "Aspen": 5,
    "DouglasFir": 6,
    "Krummholz": 7,
}

# Loop through each cover type and create a dataset
for label, name in cover_types.items():
    # 30% of the current cover type
    class_data = all_df[all_df[target_column] == label]
    class_sample = class_data.sample(frac=0.30, random_state=42)

    # 70% of normal data (from other cover types)
    normal_data = all_df[all_df[target_column] != label]
    normal_sample = normal_data.sample(n=len(class_sample) * (7 // 3), random_state=42)

    # Combine the class and normal data
    combined_data = pd.concat([class_sample, normal_sample])

    # Save the dataset to CSV
    path = os.path.join(output_folder, f"{name}.csv")
    combined_data.to_csv(path, index=False)
    print(f"{name}.csv created with {len(combined_data)} rows.")

print("All datasets created successfully!")

print(all_df.head())
print(all_df[target_column].mode())
print(data.feature_names)

# Assuming `X` and `y` are defined from all_df
X = all_df.drop(columns=[target_column])
y = all_df[target_column]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

def get_anomaly_X_y_from_csv(csv_file, main_labels):
    df=pd.read_csv(os.path.join(output_folder, csv_file),usecols=main_labels)
    df=df.fillna(0)
    anomaly_or_not=[]
    for i in df[target_column]: #it changes the normal label to "1" and the anomaly tag to "0" for use in the machine learning algorithm
        if i == normal_target:
            anomaly_or_not.append(1)
        else:
            anomaly_or_not.append(0)           
    df[target_column]=anomaly_or_not

    y = df[target_column].values
    del df[target_column]
    X = df.values
    
    X = np.float32(X)
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    # print('X', type(X), X)
    # print('y', type(y), y)
    return (X, y, df)

seconds = time.time()

# CSV files names:
csv_files=os.listdir(output_folder)# It creates a list of file names in the "attacks" folder.
print('csv_files',csv_files)

# Headers of column
main_labels=data.feature_names[:]
main_labels.append(target_column)

ths = open("importance_list.csv", "w")
feature_importances = {}
SVMs = {}
for csv_file in csv_files:
    print('csv file', csv_file)
    
    X, y, df = get_anomaly_X_y_from_csv(csv_file, main_labels)

    #computing the feature importances
    forest = sk.ensemble.RandomForestRegressor(n_estimators=250,random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    label = csv_file.split(".")[0]
    print('importances', importances, label)
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    refclasscol=list(df.columns.values)
    impor_bars = pd.DataFrame({'Features':refclasscol[0:20],'importance':importances[0:20]})
    impor_bars = impor_bars.sort_values('importance',ascending=False)
    important_features = impor_bars['Features'].to_list()[:5]
    impor_bars = impor_bars.set_index('Features')
    print('important_features', important_features)
    feature_importances[label] = important_features

    svm = SVC()
    X_train_class = df.iloc[:, df.columns.get_indexer(important_features)]
    # print('X_train_class', X_train_class)
    y_train_class = y
    # print('y_train_class', y_train_class)
    if len(y_train_class) > 0:  # Ensure there are samples for this class
        svm.fit(X_train_class, y_train_class)
    else:
        print(f'no data for {label}')
    SVMs[label] = svm

    
    plt.rcParams['figure.figsize'] = (10, 5)
    impor_bars.plot.bar();
    #printing the feature importances  
    count=0
    fea_ture=label+"=["
    for i in impor_bars.index:
        fea_ture=fea_ture+"\""+str(i)+"\","
        count+=1
        if count==5:
            fea_ture=fea_ture[0:-1]+"]"
            break     
    print(label,"importance list:")
    print(label,"\n",impor_bars.head(20),"\n\n\n")
    print(fea_ture)
    plt.title(label+" Cover type - Feature Importance")
    plt.ylabel('Importance')
    plt.savefig(os.path.join(feature_folder, label+".pdf"),bbox_inches='tight', format = 'pdf')
    ths.write((  fea_ture ) )
    plt.tight_layout()
    plt.show()
    print("-----------------------------------------------------------------------------------------------\n\n\n\n")

print('feature_importances', feature_importances)
print("mission accomplished!")
print("Total operation time: = ",time.time()- seconds ,"seconds")
ths.close()

TEST_COUNT = 100

# Step 1: Train KNN to classify
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Step 2: Predict classes for the test set
knn_predictions = knn.predict(X_test[:TEST_COUNT])
print('knn_predictions', knn_predictions)

y_test = y_test[:TEST_COUNT]
print("KNN Evaluation:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))
print(confusion_matrix(y_test, knn_predictions))
print("\nClassification Report:")
print(classification_report(y_test, knn_predictions))
print("Accuracy:", accuracy_score(y_test, knn_predictions))

svm_predictions = []

for i in range(len(X_test)):
    test_instance = X_test[i].reshape(1, -1)
    predicted_label = cover_types[knn_predictions[i]]
    # print('predicted_label', predicted_label)
    selected_features = feature_importances[predicted_label]
    # print('selected_features', selected_features)
    svm_model = SVMs[predicted_label]
    
    # Select features for the SVM
    dd = all_df.drop(columns=[target_column])
    # test_instance_selected = test_instance[:, dd.columns.get_indexer(selected_features)]
    # print('dd.columns.get_indexer(selected_features)', dd.columns.get_indexer(selected_features))
    test_instance_selected = pd.DataFrame(
        test_instance[:, all_df.columns.get_indexer(selected_features)], 
        columns=selected_features
    )
    # print('test_instance_selected', test_instance_selected)
    
    svm_prediction = svm_model.predict(test_instance_selected)
    print((svm_prediction, inv_cover_types[predicted_label], y_test[i]))
    svm_predictions.append((svm_prediction, inv_cover_types[predicted_label], y_test[i]))
# print(svm_predictions)

# Now evaluate SVM predictions only for the anomalies detected by KNN
# Create a mask for test instances that KNN classified as anomalies
anomaly_mask = knn_predictions != normal_target  # Assuming normal_target is your normal class
anomaly_mask = anomaly_mask[:TEST_COUNT]
print('anomaly_mask', len(anomaly_mask), type(anomaly_mask))

# Get true labels and predictions for anomalies
svm_predictions_actual = [t[1] for t in svm_predictions]
print('svm_predictions_actual', len(svm_predictions_actual), type(svm_predictions_actual))
svm_predictions_actual = np.array(svm_predictions_actual)
print('svm_predictions_actual', svm_predictions_actual)
y_test_anomalies = y_test[anomaly_mask]
svm_predictions_anomalies = svm_predictions_actual[anomaly_mask]

# Evaluate SVM only on the anomalies
print("\nSVM Evaluation (for anomalies):")
print("Confusion Matrix:")
print(confusion_matrix(y_test_anomalies, svm_predictions_anomalies))
print("\nClassification Report:")
print(classification_report(y_test_anomalies, svm_predictions_anomalies))
print("Accuracy:", accuracy_score(y_test_anomalies, svm_predictions_anomalies))