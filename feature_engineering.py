# feature_engineering.py
import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self):
        self.feature_columns = None

    def create_climate_features(self, clima_df):
        """Crea características climáticas agregadas por año y región"""
        print("Creando características climáticas...")

        # Agregar características básicas
        climate_annual = clima_df.groupby(['id_region', 'anio']).agg({
            'precipitacion': ['sum', 'mean', 'std', 'max'],
            'temperatura_max': ['mean', 'max', 'std'],
            'temperatura_min': ['mean', 'min', 'std']
        }).reset_index()

        # Aplanar columnas multi-nivel
        climate_annual.columns = [
            'id_region', 'anio',
            'precip_total', 'precip_promedio', 'precip_std', 'precip_max',
            'temp_max_promedio', 'temp_max_max', 'temp_max_std',
            'temp_min_promedio', 'temp_min_min', 'temp_min_std'
        ]

        # Características adicionales - grados día de crecimiento
        clima_df['temp_promedio'] = (clima_df['temperatura_max'] + clima_df['temperatura_min']) / 2
        clima_df['grados_dia'] = np.where(
            (clima_df['temp_promedio'] > 5) & (clima_df['temp_promedio'] < 30),
            clima_df['temp_promedio'] - 5,
            0
        )

        # Días con heladas
        clima_df['es_helada'] = (clima_df['temperatura_min'] < 0).astype(int)

        # Días con lluvia suficiente
        clima_df['lluvia_suficiente'] = (clima_df['precipitacion'] > 5).astype(int)

        # Agregar características avanzadas
        climate_advanced = clima_df.groupby(['id_region', 'anio']).agg({
            'grados_dia': 'sum',
            'es_helada': 'sum',
            'lluvia_suficiente': 'sum',
            'temp_promedio': ['mean', 'std']
        }).reset_index()

        climate_advanced.columns = [
            'id_region', 'anio',
            'grados_dia_total', 'dias_helada', 'dias_lluvia_suficiente',
            'temp_promedio_media', 'temp_promedio_std'
        ]

        # Combinar todas las características climáticas
        climate_features = climate_annual.merge(
            climate_advanced,
            on=['id_region', 'anio'],
            how='left'
        )

        return climate_features

    def prepare_training_data(self, clima_df, siembra_df, cosecha_df, cultivo_df):
        """Prepara el dataset completo para entrenamiento"""
        print("Preparando datos de entrenamiento...")

        # Crear características climáticas
        climate_features = self.create_climate_features(clima_df)

        # Combinar datos de siembra con características climáticas
        siembra_completa = siembra_df.merge(
            climate_features,
            on=['id_region', 'anio'],
            how='left'
        ).merge(
            cultivo_df,
            on='id_cultivo',
            how='left'
        )

        # Combinar con datos de cosecha (variable objetivo)
        training_data = siembra_completa.merge(
            cosecha_df[['id_region', 'id_cultivo', 'anio', 'toneladas']],
            on=['id_region', 'id_cultivo', 'anio'],
            how='inner',  # Solo registros que tienen datos de cosecha
            suffixes=('', '_cosecha')
        )

        return training_data

    def create_feature_matrix(self, training_data):
        """Crea la matriz de características para el modelo"""
        print("Creando matriz de características...")

        # Características numéricas base
        numeric_features = [
            'hectareas',
            'precip_total', 'precip_promedio', 'precip_std', 'precip_max',
            'temp_max_promedio', 'temp_max_max', 'temp_max_std',
            'temp_min_promedio', 'temp_min_min', 'temp_min_std',
            'grados_dia_total', 'dias_helada', 'dias_lluvia_suficiente',
            'temp_promedio_media', 'temp_promedio_std'
        ]

        # One-hot encoding para cultivos y regiones
        cultivo_dummies = pd.get_dummies(
            training_data['nombre_cultivo'],
            prefix='cultivo'
        )

        region_dummies = pd.get_dummies(
            training_data['id_region'].astype(str),
            prefix='region'
        )

        # Combinar todas las características
        X = pd.concat([
            training_data[numeric_features],
            cultivo_dummies,
            region_dummies
        ], axis=1)

        # Variable objetivo
        y = training_data['toneladas']

        # Guardar nombres de columnas para uso futuro
        self.feature_columns = X.columns.tolist()

        return X, y