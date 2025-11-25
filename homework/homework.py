# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import json
import os
import gzip
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent  # Carpeta LAB-12...

class DataPreprocessor:
    @staticmethod
    def read_dataset(relative_path: str) -> pd.DataFrame:
        return pd.read_csv(BASE_DIR / relative_path, compression="zip")

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        processed = df.copy()
        processed = processed[
            (processed["MARRIAGE"] != 0) & (processed["EDUCATION"] != 0)
        ]
        processed.loc[processed["EDUCATION"] >= 4, "EDUCATION"] = 4

        processed = (
            processed.rename(columns={"default payment next month": "default"})
            .drop("ID", axis=1)
            .dropna()
        )

        return processed


class ModelBuilder:
    def __init__(self):
        self.categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
        self.target_col = "default"

    def create_feature_pipeline(self, X):
        numeric_cols = [col for col in X.columns if col not in self.categorical_cols]

        preprocessor = ColumnTransformer(
            [
                ("categorical", OneHotEncoder(), self.categorical_cols),
                ("numeric", StandardScaler(), numeric_cols),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("feature_selection", SelectKBest(score_func=f_classif)),
                ("pca", PCA()),
                ("classifier", MLPClassifier(max_iter=15000, random_state=21)),
            ]
        )

        param_grid = {
            "pca__n_components": [None],
            "feature_selection__k": [20],
            "classifier__hidden_layer_sizes": [(50, 30, 40, 60)],
            "classifier__alpha": [0.26],
            "classifier__learning_rate_init": [0.001],
        }

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=10,
            scoring="balanced_accuracy",
            n_jobs=-1,
            refit=True,
        )

    def train_model(self, X, y):
        grid_search = self.create_feature_pipeline(X)
        return grid_search.fit(X, y)


class MetricsCalculator:
    @staticmethod
    def compute_performance_metrics(dataset_name: str, y_true, y_pred) -> dict:
        return {
            "type": "metrics",
            "dataset": dataset_name,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }

    @staticmethod
    def compute_confusion_matrix(dataset_name: str, y_true, y_pred) -> dict:
        cm = confusion_matrix(y_true, y_pred)
        return {
            "type": "cm_matrix",
            "dataset": dataset_name,
            "true_0": {
                "predicted_0": int(cm[0, 0]),
                "predicted_1": int(cm[0, 1]),
            },
            "true_1": {
                "predicted_0": int(cm[1, 0]),
                "predicted_1": int(cm[1, 1]),
            },
        }


class CreditDefaultPredictor:
    def __init__(self):
        self.data_preprocessor = DataPreprocessor()
        self.model_builder = ModelBuilder()
        self.metrics_calculator = MetricsCalculator()

    def prepare_data(self, df: pd.DataFrame):
        X = df.drop("default", axis=1)
        y = df["default"]
        return X, y

    def save_model(self, model, relative_path: str):
        path = BASE_DIR / relative_path
        os.makedirs(path.parent, exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(model, f)

    def save_metrics(self, metrics_list: list, relative_path: str):
        path = BASE_DIR / relative_path
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for metric in metrics_list:
                f.write(json.dumps(metric) + "\n")

    def run_pipeline(self):
        train_df = self.data_preprocessor.read_dataset("files/input/train_data.csv.zip")
        test_df = self.data_preprocessor.read_dataset("files/input/test_data.csv.zip")

        train_clean = self.data_preprocessor.preprocess_data(train_df)
        test_clean = self.data_preprocessor.preprocess_data(test_df)

        X_train, y_train = self.prepare_data(train_clean)
        X_test, y_test = self.prepare_data(test_clean)

        model = self.model_builder.train_model(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        metrics = [
            self.metrics_calculator.compute_performance_metrics("train", y_train, train_preds),
            self.metrics_calculator.compute_performance_metrics("test", y_test, test_preds),
            self.metrics_calculator.compute_confusion_matrix("train", y_train, train_preds),
            self.metrics_calculator.compute_confusion_matrix("test", y_test, test_preds),
        ]

        self.save_model(model, "files/models/model.pkl.gz")
        self.save_metrics(metrics, "files/output/metrics.json")


if __name__ == "__main__":
    predictor = CreditDefaultPredictor()
    predictor.run_pipeline()