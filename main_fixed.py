# main_fixed.py
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from agricultural_model_fixed import AgriculturalProductionModelFixed
import pandas as pd
import os


def main():
    print("=== SISTEMA DE PREDICCIÓN DE PRODUCCIÓN AGRÍCOLA (VERSIÓN CORREGIDA) ===\n")

    # Paso 1: Cargar datos
    print("Paso 1: Cargando datos desde la base de datos...")
    data_loader = DataLoader()
    clima_df, siembra_df, cosecha_df, cultivo_df = data_loader.load_all_data()

    # Verificar que tenemos datos
    if clima_df is None or siembra_df is None or cosecha_df is None:
        print("Error: No se pudieron cargar todos los datos")
        return

    print(f"Datos climáticos: {clima_df.shape}")
    print(f"Datos de siembra: {siembra_df.shape}")
    print(f"Datos de cosecha: {cosecha_df.shape}")
    print(f"Datos de cultivos: {cultivo_df.shape}")

    # Paso 2: Ingeniería de características
    print("\nPaso 2: Realizando ingeniería de características...")
    feature_engineer = FeatureEngineer()
    training_data = feature_engineer.prepare_training_data(
        clima_df, siembra_df, cosecha_df, cultivo_df
    )

    print(f"Datos de entrenamiento combinados: {training_data.shape}")
    print(f"Cultivos únicos en datos: {training_data['nombre_cultivo'].nunique()}")
    print(f"Regiones únicas en datos: {training_data['id_region'].nunique()}")

    # Paso 3: Entrenar modelo corregido
    print("\nPaso 3: Entrenando modelo con características consistentes...")
    model = AgriculturalProductionModelFixed()
    results = model.train(training_data)

    # Paso 4: Guardar modelo
    print("\nPaso 4: Guardando modelo...")
    if not os.path.exists('models'):
        os.makedirs('models')

    model.save_model('models/agricultural_production_model_fixed.pkl')

    # Paso 5: Probar predicción de ejemplo
    print("\nPaso 5: Probando predicción de ejemplo...")
    try:
        # Crear datos de ejemplo para prueba
        example_data = pd.DataFrame({
            'hectareas': [100],
            'precip_total': [800],
            'temp_max_promedio': [25],
            'temp_min_promedio': [12],
            'dias_helada': [5]
        })

        # Obtener un cultivo y región que existan en los datos
        example_cultivo = training_data['nombre_cultivo'].iloc[0]
        example_region = training_data['id_region'].iloc[0]

        prediction = model.predict(example_data, example_cultivo, example_region)
        print(f"Predicción de ejemplo: {prediction:.2f} toneladas para {example_cultivo} en región {example_region}")

    except Exception as e:
        print(f"Error en prueba de predicción: {e}")

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print("El modelo corregido está listo para usar!")


if __name__ == "__main__":
    main()