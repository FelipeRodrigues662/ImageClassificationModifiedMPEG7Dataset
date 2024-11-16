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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# Caminho dos dados
dataset_path = ".\\mpeg7_mod"  # Ajuste o caminho conforme o sistema
output_path = ".\\mpeg7_mod_segmented"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Funções para segmentação
def segment_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return None
    _, segmented = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return segmented

def process_dataset(dataset_path, output_path):
    for root, _, files in os.walk(dataset_path):
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
    print(f"Imagens segmentadas e salvas em {output_path}")

# Funções para extração de características
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
        "eccentricity": eccentricity,
    }

def process_segmented_images(segmented_path):
    features = []
    for root, _, files in os.walk(segmented_path):
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

# Processamento inicial
process_dataset(dataset_path, output_path)
features_df = process_segmented_images(output_path)
features_df.to_csv("features.csv", index=False)

# Divisão do conjunto de dados
X = features_df.drop(columns=["class", "file"])
y = features_df["class"]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

# Normalização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Treinamento e Avaliação dos Classificadores
knn = KNeighborsClassifier()
grid_knn = GridSearchCV(knn, {"n_neighbors": [3, 5, 7, 9]}, cv=5, scoring="accuracy")
grid_knn.fit(X_train, y_train)
best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test)

rf = RandomForestClassifier(random_state=42)
param_grid_rf = {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="accuracy")
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Resultados
print("k-NN Report:\n", classification_report(y_test, y_pred_knn))
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))


# Redução de Dimensionalidade com PCA para Visualização
pca = PCA(n_components=2)  # Reduz para 2 dimensões
X_test_pca = pca.fit_transform(X_test)

# Mapeamento de Cores para Classes
unique_classes = y_test.unique()
class_color_map = {cls: idx for idx, cls in enumerate(unique_classes)}
colors = [class_color_map[label] for label in y_test]

# Gráfico de Dispersão no Espaço PCA
plt.figure(figsize=(10, 6))

# Plot para cada classe
for cls in unique_classes:
    cls_points = X_test_pca[np.array(y_test) == cls]
    plt.scatter(
        cls_points[:, 0], 
        cls_points[:, 1], 
        label=f"Classe {cls}", 
        alpha=0.7
    )

plt.title("Visualização das Classes no Espaço PCA (Dados de Teste)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Classes")
plt.grid(True)
plt.show()

# Visualização para k-NN com k=1 e k=3
knn_k1 = KNeighborsClassifier(n_neighbors=1)
knn_k1.fit(X_train, y_train)
y_pred_knn_k1 = knn_k1.predict(X_test)

knn_k3 = KNeighborsClassifier(n_neighbors=3)
knn_k3.fit(X_train, y_train)
y_pred_knn_k3 = knn_k3.predict(X_test)

# Visualização para k-NN com k=1
plt.figure(figsize=(10, 6))
X_test_pca_knn_k1 = pca.transform(X_test)
for cls in unique_classes:
    cls_points = X_test_pca_knn_k1[np.array(y_test) == cls]
    plt.scatter(
        cls_points[:, 0],
        cls_points[:, 1],
        label=f"Classe Real {cls}",
        alpha=0.4,
        edgecolor="black"
    )

for cls in unique_classes:
    pred_points = X_test_pca_knn_k1[np.array(y_pred_knn_k1) == cls]
    plt.scatter(
        pred_points[:, 0],
        pred_points[:, 1],
        label=f"Predição k=1 {cls}",
        alpha=0.6,
        marker="x"
    )

plt.title("Visualização de Predições k-NN (k=1) no Espaço PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Legenda")
plt.grid(True)
plt.show()

# Visualização para k-NN com k=3
plt.figure(figsize=(10, 6))
X_test_pca_knn_k3 = pca.transform(X_test)
for cls in unique_classes:
    cls_points = X_test_pca_knn_k3[np.array(y_test) == cls]
    plt.scatter(
        cls_points[:, 0],
        cls_points[:, 1],
        label=f"Classe Real {cls}",
        alpha=0.4,
        edgecolor="black"
    )

for cls in unique_classes:
    pred_points = X_test_pca_knn_k3[np.array(y_pred_knn_k3) == cls]
    plt.scatter(
        pred_points[:, 0],
        pred_points[:, 1],
        label=f"Predição k=3 {cls}",
        alpha=0.6,
        marker="x"
    )

plt.title("Visualização de Predições k-NN (k=3) no Espaço PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Legenda")
plt.grid(True)
plt.show()

# Treinamento de um Perceptron Multicamada
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

# Resultados do Perceptron
print("MLP Report:\n", classification_report(y_test, y_pred_mlp))

# Visualização das Predições do Perceptron no Espaço PCA
plt.figure(figsize=(10, 6))

# Predições no espaço PCA
X_test_pca_mlp = pca.transform(X_test)
for cls in unique_classes:
    cls_points = X_test_pca_mlp[np.array(y_test) == cls]
    plt.scatter(
        cls_points[:, 0], 
        cls_points[:, 1], 
        label=f"Classe Real {cls}", 
        alpha=0.4,
        edgecolor='black'
    )

# Predições do modelo MLP no espaço PCA
for cls in unique_classes:
    pred_points = X_test_pca_mlp[np.array(y_pred_mlp) == cls]
    plt.scatter(
        pred_points[:, 0], 
        pred_points[:, 1], 
        label=f"Predição {cls}", 
        alpha=0.6,
        marker='x'
    )

plt.title("Visualização de Predições do Perceptron no Espaço PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Legenda")
plt.grid(True)
plt.show()