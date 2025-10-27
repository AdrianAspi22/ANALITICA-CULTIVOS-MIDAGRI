from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from agricultural_model import AgriculturalProductionModel
import pandas as pd
import os


def main():
    print("=== SISTEMA DE PREDICCIÓN DE PRODUCCIÓN AGRÍCOLA ===\n")

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

    # Paso 3: Entrenar modelo
    print("\nPaso 3: Entrenando modelo con características consistentes...")
    model = AgriculturalProductionModel()
    results = model.train(training_data)

    # Paso 4: Mostrar importancia de características
    print("\nPaso 4: Analizando importancia de características...")
    feature_importance_df = model.feature_importance()
    print("\nTop 10 características más importantes:")
    print(feature_importance_df.head(10))

    # Paso 5: Guardar modelo
    print("\nPaso 5: Guardando modelo...")
    if not os.path.exists('models'):
        os.makedirs('models')

    model.save_model('models/agricultural_production_model.pkl')

    # Paso 6: Probar predicción de ejemplo
    print("\nPaso 6: Probando predicción de ejemplo...")
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

        # Probar con varios ejemplos
        print("\nProbando con múltiples ejemplos:")
        for i in range(min(3, len(training_data))):
            sample = training_data.iloc[i]
            example_data = pd.DataFrame({
                'hectareas': [sample['hectareas']],
                'precip_total': [sample.get('precip_total', 800)],
                'temp_max_promedio': [sample.get('temp_max_promedio', 25)],
                'temp_min_promedio': [sample.get('temp_min_promedio', 12)],
                'dias_helada': [sample.get('dias_helada', 5)]
            })

            pred = model.predict(example_data, sample['nombre_cultivo'], sample['id_region'])
            actual = sample['toneladas']
            print(f"Ejemplo {i + 1}: Predicho={pred:.2f}, Actual={actual:.2f}, Diferencia={abs(pred - actual):.2f}")

    except Exception as e:
        print(f"Error en prueba de predicción: {e}")

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print("El modelo está listo para usar!")


if __name__ == "__main__":
    main()