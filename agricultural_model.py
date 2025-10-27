import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class AgriculturalProductionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.cultivo_mapping = {}
        self.region_mapping = {}
        self.is_trained = False

    def initialize_model(self):
        """Inicializa el modelo Random Forest"""
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    def _create_consistent_features(self, training_data):
        """Crea características consistentes para entrenamiento y predicción"""
        print("Creando características consistentes...")

        # Características numéricas base - SOLO LAS QUE EXISTEN
        possible_numeric_features = [
            'hectareas', 'precip_total', 'precip_promedio', 'precip_std', 'precip_max',
            'temp_max_promedio', 'temp_max_max', 'temp_max_std', 'temp_min_promedio',
            'temp_min_min', 'temp_min_std', 'grados_dia_total', 'dias_helada',
            'dias_lluvia_suficiente', 'temp_promedio_media', 'temp_promedio_std'
        ]

        # Filtrar solo las características que realmente existen en los datos
        available_numeric = [col for col in possible_numeric_features if col in training_data.columns]
        print(f"Características numéricas disponibles: {available_numeric}")

        # Crear mapeo consistente de cultivos
        cultivos_unicos = sorted(training_data['nombre_cultivo'].unique())
        self.cultivo_mapping = {cultivo: f'cultivo_{cultivo}' for cultivo in cultivos_unicos}

        # Crear mapeo consistente de regiones
        regiones_unicas = sorted(training_data['id_region'].unique())
        self.region_mapping = {region: f'region_{region}' for region in regiones_unicas}

        # Crear one-hot encoding consistente para cultivos
        cultivo_dummies = pd.get_dummies(
            training_data['nombre_cultivo'],
            prefix='cultivo'
        )

        # Asegurar que todas las columnas de cultivo estén presentes
        expected_cultivo_cols = [f'cultivo_{cultivo}' for cultivo in cultivos_unicos]
        for col_name in expected_cultivo_cols:
            if col_name not in cultivo_dummies.columns:
                cultivo_dummies[col_name] = 0

        # Crear one-hot encoding consistente para regiones
        region_dummies = pd.get_dummies(
            training_data['id_region'].astype(str),
            prefix='region'
        )

        # Asegurar que todas las columnas de región estén presentes
        expected_region_cols = [f'region_{region}' for region in regiones_unicas]
        for col_name in expected_region_cols:
            if col_name not in region_dummies.columns:
                region_dummies[col_name] = 0

        # Combinar todas las características
        X_components = [training_data[available_numeric]] if available_numeric else []
        X_components.extend([cultivo_dummies, region_dummies])

        X = pd.concat(X_components, axis=1)

        # Ordenar columnas alfabéticamente para consistencia
        X = X.reindex(sorted(X.columns), axis=1)

        # Variable objetivo
        y = training_data['toneladas']

        # Guardar nombres de columnas para uso futuro
        self.feature_columns = X.columns.tolist()

        print(f"Características finales creadas: {len(self.feature_columns)} columnas")
        return X, y

    def train(self, training_data, test_size=0.2):
        """Entrena el modelo con características consistentes"""
        print("Iniciando entrenamiento del modelo con características consistentes...")

        if self.model is None:
            self.initialize_model()

        # Crear características consistentes
        X, y = self._create_consistent_features(training_data)

        print(f"Características finales: {len(self.feature_columns)} columnas")
        print(f"Cultivos en el modelo: {len(self.cultivo_mapping)}")
        print(f"Regiones en el modelo: {len(self.region_mapping)}")

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
        print(f"Tamaño del conjunto de prueba: {X_test.shape}")

        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Entrenar modelo
        print("Entrenando modelo...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluar modelo
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        # Predicciones
        y_pred = self.model.predict(X_test_scaled)

        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("\n=== RESULTADOS DEL MODELO ===")
        print(f"R² Train: {train_score:.4f}")
        print(f"R² Test: {test_score:.4f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")

        # Validación cruzada
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"Validación Cruzada R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        self.is_trained = True

        return {
            'train_score': train_score,
            'test_score': test_score,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_scores': cv_scores
        }

    def feature_importance(self, top_n=15):
        """Muestra la importancia de las características"""
        if not self.is_trained:
            raise Exception("El modelo no ha sido entrenado")

        importances = self.model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Graficar
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_imp_df.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Características Más Importantes')
        plt.tight_layout()
        plt.show()

        return feature_imp_df

    def prepare_prediction_input(self, input_data, cultivo_nombre, region_id):
        """Prepara datos de entrada para predicción con características consistentes"""
        if not self.is_trained:
            raise Exception("El modelo no ha sido entrenado")

        # Crear DataFrame base con ceros
        prediction_df = pd.DataFrame(0, index=[0], columns=self.feature_columns)

        # Llenar características numéricas
        for col in self.feature_columns:
            if col in input_data.columns and not col.startswith(('cultivo_', 'region_')):
                prediction_df[col] = input_data[col].values[0]

        # Configurar one-hot encoding para cultivo
        cultivo_col = f'cultivo_{cultivo_nombre}'
        if cultivo_col in self.feature_columns:
            prediction_df[cultivo_col] = 1
        else:
            print(f"Advertencia: Cultivo '{cultivo_nombre}' no visto en entrenamiento.")
            # Buscar cultivo similar o usar el más común
            if self.cultivo_mapping:
                first_cultivo = list(self.cultivo_mapping.keys())[0]
                prediction_df[f'cultivo_{first_cultivo}'] = 1
                print(f"Usando '{first_cultivo}' como reemplazo.")

        # Configurar one-hot encoding para región
        region_col = f'region_{region_id}'
        if region_col in self.feature_columns:
            prediction_df[region_col] = 1
        else:
            print(f"Advertencia: Región '{region_id}' no vista en entrenamiento.")
            # Usar la primera región disponible
            if self.region_mapping:
                first_region = list(self.region_mapping.keys())[0]
                prediction_df[f'region_{first_region}'] = 1
                print(f"Usando región '{first_region}' como reemplazo.")

        return prediction_df

    def predict(self, input_data, cultivo_nombre, region_id):
        """Realiza predicciones con manejo robusto de características"""
        if not self.is_trained:
            raise Exception("El modelo no ha sido entrenado")

        # Preparar características de entrada
        X_pred = self.prepare_prediction_input(input_data, cultivo_nombre, region_id)

        # Escalar y predecir
        X_pred_scaled = self.scaler.transform(X_pred)
        prediction = self.model.predict(X_pred_scaled)[0]

        return prediction

    def save_model(self, filepath):
        """Guarda el modelo entrenado con toda la información de características"""
        if not self.is_trained:
            raise Exception("El modelo no ha sido entrenado")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'cultivo_mapping': self.cultivo_mapping,
            'region_mapping': self.region_mapping
        }

        joblib.dump(model_data, filepath)
        print(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath):
        """Carga un modelo guardado"""
        loaded_data = joblib.load(filepath)
        self.model = loaded_data['model']
        self.scaler = loaded_data['scaler']
        self.feature_columns = loaded_data['feature_columns']
        self.cultivo_mapping = loaded_data.get('cultivo_mapping', {})
        self.region_mapping = loaded_data.get('region_mapping', {})
        self.is_trained = True
        print(f"Modelo cargado desde: {filepath}")