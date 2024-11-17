import wandb
import time
import importlib
from common import common
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

wandb.login()

# Function to evaluate and print model performance
def evaluate_model(model, X_train, y_train, X_test, y_test):
    seconds = time.time()
    # Train the model
    model.fit(X_train, y_train)
    print("Train operation time: = ",time.time()- seconds ,"seconds")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Print model evaluation metrics
    common.evaluate(y_test, y_pred, f"\nModel: {model.__class__.__name__}")


if __name__ == "__main__":
    # Dictionary to store models and their names
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine (SVM)": SVC()
    }



    # project name should correspond to dataset below
    # project_name = "kdd"
    # project_name = "cicids"
    # project_name = "thyroid"
    # project_name = "unsw"
    # project_name = "cirrhosis"
    project_name = "heart"
    # project_name = "hepatitis"

    if project_name == "kdd":
        from datasets.kdd import get_processed_kdd_df
        all_df, main_labels, config = get_processed_kdd_df()
    elif project_name == "cicids":
        from datasets.cicids import get_processed_cicids_df
        all_df, main_labels, config = get_processed_cicids_df()
    elif project_name == "thyroid":
        from datasets.thyroid import get_processed_thyroid_df
        all_df, main_labels, config = get_processed_thyroid_df()
    elif project_name == "unsw":
        from datasets.unsw import get_processed_unsw_df
        all_df, main_labels, config = get_processed_unsw_df()
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
    NUMERICAL_COLUMNS = config['NUMERICAL_COLUMNS']
    CATEGORICAL_COLUMNS = config['CATEGORICAL_COLUMNS']
    ORDINAL_COLUMNS = config['ORDINAL_COLUMNS']


    # Get X and y from all_df
    X_df = all_df.drop(columns=[TARGET_COLUMN])
    y_df = all_df[TARGET_COLUMN]

    # Split the data into training and testing sets (80% train, 20% test)
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
    print(X_train_df.shape, X_test_df.shape, y_train_df.shape, y_test_df.shape)



    # Fit and transform the numeric columns
    scaler, X_train_scaled_df = common.standardise(X_train_df, NUMERICAL_COLUMNS)

    # Use the same scaler to transform X_test
    scaler, X_test_scaled_df = common.standardise(X_test_df, NUMERICAL_COLUMNS, scaler=scaler)



    # Loop through models and evaluate each one
    project_name = "baseline-" + project_name
    seconds = time.time()
    for model_name, model in models.items():
        wandb.init(project=project_name, name=model_name)
        
        # For SVM and Logistic Regression, use scaled data
        if model_name in ["Logistic Regression", "Support Vector Machine (SVM)", "K-Nearest Neighbors"]:
            evaluate_model(model, X_train_scaled_df, y_train_df, X_test_scaled_df, y_test_df)
        else:
            evaluate_model(model, X_train_df, y_train_df, X_test_df, y_test_df)
            
        wandb.finish()
    print("Total operation time: = ", time.time() - seconds, "seconds")



    wandb.finish()

