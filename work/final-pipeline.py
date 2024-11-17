import wandb
import os
import pandas as pd
import time
import importlib
from common import common
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
# import eli5
# from eli5.sklearn import explain_prediction_sklearn

wandb.login()

# Shared variables
OUTPUT_FOLDER = 'csv_files'
FEATURE_FOLDER = 'features'
TOP_FEATURES_NUM = 5
CONF_THRESHOLD = 0.95

def predict_knn(predicted_label, test_instance_df, KNNs, feature_importances):
    selected_features = feature_importances[predicted_label]
    knn_model = KNNs[predicted_label]
    # Select features for the KNN
    test_instance_selected_df = test_instance_df[selected_features]
    knn_prediction = knn_model.predict_proba(test_instance_selected_df)
    # print(knn_prediction)
    major_class_prob = knn_prediction[:, 1]
    minor_class_prob = knn_prediction[:, 0]
    
    return (minor_class_prob, major_class_prob)
    
def process_instance(i, test_instance_df, test_predictions, config, KNNs, feature_importances, y_test_df):
    NORMAL_TARGET = config['NORMAL_TARGET']
    INV_TARGET_DICT = config['INV_TARGET_DICT']
    TARGET_DICT = config['TARGET_DICT']
    if test_predictions[i] == NORMAL_TARGET:
        max_prob = CONF_THRESHOLD
        final_class_1 = NORMAL_TARGET
        final_class_2 = final_class_1

        for key, knn in KNNs.items():
            sub_minor_class_prob, sub_major_class_prob = predict_knn(key, test_instance_df, KNNs, feature_importances)
            if sub_minor_class_prob > sub_major_class_prob and sub_minor_class_prob > max_prob:
                max_prob = sub_minor_class_prob
                final_class_2 = INV_TARGET_DICT[key]

        status = ''
        if final_class_1 != final_class_2 and final_class_2 == y_test_df.iloc[i]:
            status = 'NORMAL improve'
            print('final_class1', final_class_1, 'final_class2', final_class_2, 'actual', y_test_df.iloc[i], status, max_prob)
        if final_class_1 != final_class_2 and final_class_1 == y_test_df.iloc[i]:
            status = 'NORMAL deprove'
            print('final_class1', final_class_1, 'final_class2', final_class_2, 'actual', y_test_df.iloc[i], status, max_prob)

    else:
        predicted_label = TARGET_DICT[test_predictions[i]]
        minor_class_prob, major_class_prob = predict_knn(predicted_label, test_instance_df, KNNs, feature_importances)
        max_prob = minor_class_prob if minor_class_prob > major_class_prob else major_class_prob
        final_class_1 = INV_TARGET_DICT[predicted_label] if minor_class_prob > major_class_prob else NORMAL_TARGET
        final_class_2 = final_class_1

        if major_class_prob > 1.0 - CONF_THRESHOLD and major_class_prob < CONF_THRESHOLD:
            for key, knn in KNNs.items():
                if key != predicted_label:
                    sub_minor_class_prob, sub_major_class_prob = predict_knn(key, test_instance_df, KNNs, feature_importances)
                    if sub_minor_class_prob > sub_major_class_prob and sub_minor_class_prob > max_prob:
                        max_prob = sub_minor_class_prob
                        final_class_2 = INV_TARGET_DICT[key]

        status = ''
        if final_class_1 != final_class_2 and final_class_2 == y_test_df.iloc[i]:
            status = 'improve'
            print('final_class1', final_class_1, 'final_class2', final_class_2, 'actual', y_test_df.iloc[i], status, major_class_prob, max_prob)
        if final_class_1 != final_class_2 and final_class_1 == y_test_df.iloc[i]:
            status = 'deprove'
            print('final_class1', final_class_1, 'final_class2', final_class_2, 'actual', y_test_df.iloc[i], status, major_class_prob, max_prob)

    return i, final_class_2


if __name__ == "__main__":
    # project name should correspond to dataset below
    # project_name = "thyroid"
    # project_name = "cirrhosis"
    project_name = "heart"
    # project_name = "hepatitis"

    if project_name == "thyroid":
        from datasets.thyroid import get_processed_thyroid_df
        all_df, main_labels, config = get_processed_thyroid_df()
    elif project_name == "cirrhosis":
        from datasets.cirrhosis import get_processed_cirrhosis_df
        all_df, main_labels, config = get_processed_cirrhosis_df()
    elif project_name == "heart":
        from datasets.heart import get_processed_heart_df
        all_df, main_labels, config = get_processed_heart_df()
    else:
        from datasets.hepatitis import get_processed_hepatitis_df
        all_df, main_labels, config = get_processed_hepatitis_df()



    # Should already be one hot encoded and label encoded
    TARGET_COLUMN = config['TARGET_COLUMN']
    NORMAL_TARGET = config['NORMAL_TARGET']
    TARGET_DICT = config['TARGET_DICT']
    INV_TARGET_DICT = config['INV_TARGET_DICT']
    NUMERICAL_COLUMNS = config['NUMERICAL_COLUMNS'].copy()
    CATEGORICAL_COLUMNS = config['CATEGORICAL_COLUMNS']
    ORDINAL_COLUMNS = config['ORDINAL_COLUMNS']


    # Get X and y from all_df
    X_df = all_df.drop(columns=[TARGET_COLUMN])
    y_df = all_df[TARGET_COLUMN]

    # Split the data into training and test sets
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
    print(X_train_df.shape, X_test_df.shape, y_train_df.shape, y_test_df.shape)



    seconds = time.time()

    # Define the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_df, y_train_df)

    # Predict classes for the test set
    test_predictions = rf.predict(X_test_df)
    print('test_predictions', test_predictions)

    print("Total operation time: = ", time.time() - seconds, "seconds")



    pn = "final-pipeline-" + project_name
    wandb.init(project=pn, name="classification")
    common.evaluate(y_test_df, test_predictions, heading='Multiclass Classification Evaluation')
    wandb.finish()



    # # Extract decision rules and feature importance
    # rules = eli5.explain_weights(rf, feature_names=list(X_train_df.columns))
    # # display(rules)
    # print(eli5.format_as_text(rules))

    # from sklearn.tree import export_text, plot_tree
    # # Here we use out-of-bag score or use Gini Importance as criteria
    # best_tree_index = max(range(len(rf.estimators_)),
    #                     key=lambda i: rf.estimators_[i].score(X_train_df, y_train_df))
    # best_tree = rf.estimators_[best_tree_index]

    # feature_names = X_train_df.columns
    # tree_rules = export_text(best_tree, feature_names=feature_names)
    # print("Best Tree's Decision Rules:")
    # print(tree_rules)

    # Create a folder to save the CSVs
    common.remove_files_from_directory(OUTPUT_FOLDER)
    common.remove_files_from_directory(FEATURE_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FEATURE_FOLDER, exist_ok=True)

    if project_name == "thyroid":
        # Applying log transformation to reduce skewness
        all_df['log_TSH'] = np.log1p(all_df['TSH'])
        all_df['log_FTI'] = np.log1p(all_df['FTI'])
        all_df['log_TT4'] = np.log1p(all_df['TT4'])
        all_df['log_T3'] = np.log1p(all_df['T3'])
        all_df = all_df.drop(['TSH', 'FTI', 'TT4', 'T3'], axis=1)

    elif project_name == "cirrhosis":
        all_df['Age_Bilirubin_interaction'] = all_df['Age'] * all_df['Bilirubin']
        all_df['Age_Copper_interaction'] = all_df['Age'] * all_df['Copper']
        all_df['Age_Prothrombin_interaction'] = all_df['Age'] * all_df['Prothrombin']
        all_df['Age_SGOT_interaction'] = all_df['Age'] * all_df['SGOT']
        all_df['Age_Platelets_interaction'] = all_df['Age'] * all_df['Platelets']
        all_df = all_df.drop(['Bilirubin', 'Copper', 'Prothrombin', 'SGOT', 'Platelets'], axis=1)
        
    elif project_name == "heart":
        all_df['age_chol_interaction'] = all_df['age'] * all_df['chol']
        all_df['age_thalch_interaction'] = all_df['age'] * all_df['thalch']
        all_df['age_trestbps_interaction'] = all_df['age'] * all_df['trestbps']
        all_df['age_oldpeak_interaction'] = all_df['age'] * all_df['oldpeak']
        all_df = all_df.drop(['chol', 'thalch', 'trestbps', 'oldpeak'], axis=1)

    else:
        all_df['Age_AST_interaction'] = all_df['Age'] * all_df['AST']
        all_df['Age_CHE_interaction'] = all_df['Age'] * all_df['CHE']
        all_df['Age_ALT_interaction'] = all_df['Age'] * all_df['ALT']
        all_df['Age_ALP_interaction'] = all_df['Age'] * all_df['ALP']
        all_df['Age_GGT_interaction'] = all_df['Age'] * all_df['GGT']
        all_df['Age_BIL_interaction'] = all_df['Age'] * all_df['BIL']
        all_df['Age_PROT_interaction'] = all_df['Age'] * all_df['PROT']
        all_df['Age_ALB_interaction'] = all_df['Age'] * all_df['ALB']
        all_df = all_df.drop(['Age', 'AST', 'CHE', 'ALT', 'ALP', 'GGT', 'BIL', 'PROT', 'ALB'], axis=1)

    # Get X and y from all_df
    X_df = all_df.drop(columns=[TARGET_COLUMN])
    y_df = all_df[TARGET_COLUMN]

    # Split the data into training and test sets
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
    print(X_train_df.shape, X_test_df.shape, y_train_df.shape, y_test_df.shape)

    # Save all data as csv
    all_df.to_csv('all_data.csv' ,index = False)

    main_labels = all_df.columns
    print(main_labels)



    NUMERICAL_COLUMNS = config['NUMERICAL_COLUMNS'].copy()

    if project_name == "thyroid":
        for col in ['log_TSH', 'log_FTI', 'log_TT4', 'log_T3']:
            NUMERICAL_COLUMNS.append(col)
        for col in ['TSH', 'FTI', 'TT4', 'T3']:
            NUMERICAL_COLUMNS.remove(col)

    elif project_name == "cirrhosis":
        for col in ['Age_Bilirubin_interaction', 'Age_Copper_interaction', 'Age_Prothrombin_interaction', 'Age_SGOT_interaction', 'Age_Platelets_interaction']:
            NUMERICAL_COLUMNS.append(col)
        for col in ['Bilirubin', 'Copper', 'Prothrombin', 'SGOT', 'Platelets']:
            NUMERICAL_COLUMNS.remove(col)
            
    elif project_name == "heart":    
        for col in ['age_chol_interaction', 'age_thalch_interaction', 'age_trestbps_interaction',
                    'age_oldpeak_interaction']:
            NUMERICAL_COLUMNS.append(col)
        for col in ['chol', 'thalch', 'trestbps', 'oldpeak']:
            NUMERICAL_COLUMNS.remove(col)

    else:
        for col in ['Age_AST_interaction', 'Age_CHE_interaction', 'Age_ALT_interaction', 'Age_ALP_interaction',
                    'Age_GGT_interaction', 'Age_BIL_interaction', 'Age_PROT_interaction', 'Age_ALB_interaction']:
            NUMERICAL_COLUMNS.append(col)
        for col in ['Age', 'AST', 'CHE', 'ALT', 'ALP', 'GGT', 'BIL', 'PROT', 'ALB']:
            NUMERICAL_COLUMNS.remove(col)

    print(NUMERICAL_COLUMNS)


    # Fit and transform the numeric columns
    scaler, X_train_scaled_df = common.standardise(X_train_df, NUMERICAL_COLUMNS)

    # Use the same scaler to transform X_test
    scaler, X_test_scaled_df = common.standardise(X_test_df, NUMERICAL_COLUMNS, scaler=scaler)



    seconds = time.time()

    minor_type_counts = all_df[TARGET_COLUMN].value_counts()
    minor_type_dict = minor_type_counts.to_dict()
    print('minor_type_dict', minor_type_dict)
    target_index = all_df.columns.get_loc(TARGET_COLUMN)

    # Linear method
    for label, name in TARGET_DICT.items():
        if label == NORMAL_TARGET:
            continue  # Skip the normal target

        common.get_dataset_for_label(label, name, target_index, NORMAL_TARGET, OUTPUT_FOLDER, main_labels)

    print("All datasets created successfully!")
    print("Total operation time: =", time.time() - seconds, "seconds")



    seconds = time.time()

    # CSV files names:
    csv_files=os.listdir(OUTPUT_FOLDER)
    print('csv_files',csv_files)

    feature_importances = {}
    KNNs = {}
    model_name = "knn"

    print(main_labels)
    # Linear way
    for csv in csv_files:
        label, important_features, knn, impor_bars = common.process_csv_with_args(csv, 
            main_labels=main_labels, 
            target_column=TARGET_COLUMN, 
            normal_target=NORMAL_TARGET, 
            numerical_columns=NUMERICAL_COLUMNS, 
            output_folder=OUTPUT_FOLDER,
            top_features_num=TOP_FEATURES_NUM,
            scaler=scaler,
            model_name=model_name)
        feature_importances[label] = important_features
        KNNs[label] = knn
        common.show_feature_importance(impor_bars, label, FEATURE_FOLDER)
        print("-----------------------------------------------------------------------------------------------\n\n\n\n")

    print('feature_importances:', feature_importances)
    print("Total operation time: =", time.time() - seconds, "seconds")



    seconds = time.time()

    
    # Run all test data instances in parallel and retain order
    knn_predictions = [None] * len(X_test_scaled_df) 

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_instance, idx, X_test_scaled_df.iloc[idx].to_frame().T, test_predictions, config, KNNs, feature_importances, y_test_df): idx for idx in range(len(X_test_scaled_df))}

        for future in as_completed(futures):
            idx, prediction = future.result()
            knn_predictions[idx] = prediction 

    print("Total operation time: = ", time.time() - seconds, "seconds")



    # Evaluate KNN on all test data
    pn = "final-pipeline-" + project_name
    wandb.init(project=pn, name="final")

    final_knn_predictions_df = pd.DataFrame(knn_predictions, columns=[TARGET_COLUMN])
    common.evaluate(y_test_df, final_knn_predictions_df, heading='KNN Evaluation (overall)')
    wandb.finish()