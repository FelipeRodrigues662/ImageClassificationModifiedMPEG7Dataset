import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Segmentação
dataset_path = ".\\mpeg7_mod"

output_path = ".\\mpeg7_mod_segmented"

if not os.path.exists(output_path):
    os.makedirs(output_path)

def segment_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return None

    _, segmented = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return segmented

def process_dataset(dataset_path, output_path):
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  
                image_path = os.path.join(root, file)
                segmented = segment_image(image_path)
                
                if segmented is not None:
                    relative_path = os.path.relpath(root, dataset_path)
                    save_dir = os.path.join(output_path, relative_path)
                    
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    save_path = os.path.join(save_dir, file)
                    cv2.imwrite(save_path, segmented)
                    

process_dataset(dataset_path, output_path)
print(f"Imagems segmentadas com sucesso")

# Extração de Características Morfológicas
segmented_path = ".\\mpeg7_mod_segmented"


def extract_morphological_features(segmented_image):

    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"area": 0, "perimeter": 0, "circularity": 0, "aspect_ratio": 0, "eccentricity": 0}

    contour = max(contours, key=cv2.contourArea)
    
    area = cv2.contourArea(contour)
    
    perimeter = cv2.arcLength(contour, True)
    
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0

    x, y, width, height = cv2.boundingRect(contour)
    
    aspect_ratio = width / height if height != 0 else 0

    try:
        (x_ellipse, y_ellipse), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
        major_axis = max(major_axis, minor_axis)
        minor_axis = min(major_axis, minor_axis)
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
    except:
        eccentricity = 0  
    

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "eccentricity": eccentricity
    }


def process_segmented_images(segmented_path):
    features = []
    for root, dirs, files in os.walk(segmented_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  
                image_path = os.path.join(root, file)
                segmented = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if segmented is not None:
                    morphological_features = extract_morphological_features(segmented)
                    morphological_features["class"] = os.path.basename(root)  
                    morphological_features["file"] = file  
                    features.append(morphological_features)
    
    return pd.DataFrame(features)

features_df = process_segmented_images(segmented_path)

print(features_df.head())

features_df.to_csv("morphological_features_with_eccentricity_aspect_ratio.csv", index=False)
print("Características morfológicas salvas em 'morphological_features_with_eccentricity_aspect_ratio.csv'.")

#Divisão do Conjunto de Dados
features_file = "morphological_features_with_eccentricity_aspect_ratio.csv"
features_df = pd.read_csv(features_file)

X = features_df.drop(columns=["class", "file"])  
y = features_df["class"] 

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

train_df = pd.concat([X_train, y_train], axis=1)
train_df["set"] = "train"

val_df = pd.concat([X_val, y_val], axis=1)
val_df["set"] = "validation"

test_df = pd.concat([X_test, y_test], axis=1)
test_df["set"] = "test"

full_dataset_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
full_dataset_df.to_csv("full_dataset_with_splits.csv", index=False)
print("Conjunto completo com divisões salvo em 'full_dataset_with_splits.csv'.")

#Normalização dos Dados
dataset_file = "full_dataset_with_splits.csv"
dataset_df = pd.read_csv(dataset_file)

X = dataset_df.drop(columns=["class", "set"])  
y = dataset_df[["class", "set"]]  

scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

normalized_dataset_df = pd.concat([X_normalized, y.reset_index(drop=True)], axis=1)

normalized_dataset_df.to_csv("normalized_dataset_with_splits.csv", index=False)

print("Conjunto de dados normalizado salvo em 'normalized_dataset_with_splits.csv'.")

# Treinamento e Teste do Classificador
dataset_file = "normalized_dataset_with_splits.csv"
dataset_df = pd.read_csv(dataset_file)

X_train = dataset_df[dataset_df["set"] == "train"].drop(columns=["class", "set"])
y_train = dataset_df[dataset_df["set"] == "train"]["class"]

X_val = dataset_df[dataset_df["set"] == "validation"].drop(columns=["class", "set"])
y_val = dataset_df[dataset_df["set"] == "validation"]["class"]

X_test = dataset_df[dataset_df["set"] == "test"].drop(columns=["class", "set"])
y_test = dataset_df[dataset_df["set"] == "test"]["class"]

X_train_combined = pd.concat([X_train, X_val])
y_train_combined = pd.concat([y_train, y_val])

def save_results_to_dataframe(y_true, y_pred, model_name, best_params=None):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    confusion = confusion_matrix(y_true, y_pred)
    confusion_df = pd.DataFrame(confusion, columns=["Predicted_" + str(i) for i in range(confusion.shape[1])],
                                index=["Actual_" + str(i) for i in range(confusion.shape[0])])
    
    params_df = pd.DataFrame([best_params]) if best_params else pd.DataFrame()
    
    return report_df, confusion_df, params_df

knn = KNeighborsClassifier()
param_grid_knn = {"n_neighbors": [3, 5, 7, 9]}
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring="accuracy")
grid_knn.fit(X_train_combined, y_train_combined)

best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test)

knn_report_df, knn_confusion_df, knn_params_df = save_results_to_dataframe(
    y_test, y_pred_knn, "k-NN", best_params=grid_knn.best_params_
)

rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="accuracy")
grid_rf.fit(X_train_combined, y_train_combined)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

rf_report_df, rf_confusion_df, rf_params_df = save_results_to_dataframe(
    y_test, y_pred_rf, "Random Forest", best_params=grid_rf.best_params_
)

with pd.ExcelWriter("classification_results.xlsx") as writer:
    knn_report_df.to_excel(writer, sheet_name="k-NN_Report")
    knn_confusion_df.to_excel(writer, sheet_name="k-NN_Confusion_Matrix")
    knn_params_df.to_excel(writer, sheet_name="k-NN_Best_Params")
    
    rf_report_df.to_excel(writer, sheet_name="RF_Report")
    rf_confusion_df.to_excel(writer, sheet_name="RF_Confusion_Matrix")
    rf_params_df.to_excel(writer, sheet_name="RF_Best_Params")

print("Resultados salvos em 'classification_results.xlsx'.")
