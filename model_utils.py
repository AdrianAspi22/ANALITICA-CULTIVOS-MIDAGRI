# model_utils.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


class ModelFeatureManager:
    def __init__(self):
        self.expected_features = None
        self.cultivo_columns = None
        self.region_columns = None
        self.numeric_columns = None

    def load_model_with_features(self, model_path):
        """Carga el modelo y extrae la información de características"""
        try:
            model_data = joblib.load(model_path)
            self.expected_features = model_data['feature_columns']
            self._analyze_features()
            return model_data
        except Exception as e:
            raise Exception(f"Error cargando el modelo: {e}")

    def _analyze_features(self):
        """Analiza las características esperadas por el modelo"""
        if self.expected_features is None:
            return

        # Separar tipos de características
        self.cultivo_columns = [col for col in self.expected_features if col.startswith('cultivo_')]
        self.region_columns = [col for col in self.expected_features if col.startswith('region_')]
        self.numeric_columns = [col for col in self.expected_features
                                if not col.startswith('cultivo_') and not col.startswith('region_')]

    def prepare_prediction_features(self, input_data, cultivo_nombre, region_id):
        """
        Prepara las características para predicción asegurando que coincidan
        con las del entrenamiento
        """
        if self.expected_features is None:
            raise Exception("Modelo no cargado correctamente")

        # Crear DataFrame base con ceros para TODAS las características esperadas
        prediction_df = pd.DataFrame(0, index=[0], columns=self.expected_features)

        # Llenar características numéricas
        for col in self.numeric_columns:
            if col in input_data.columns:
                prediction_df[col] = input_data[col].values[0]
            else:
                # Si la característica numérica no está en los datos de entrada, usar 0
                print(f"Advertencia: Característica numérica '{col}' no encontrada en datos de entrada. Usando 0.")

        # Configurar one-hot encoding para cultivo
        cultivo_col = f'cultivo_{cultivo_nombre}'
        if cultivo_col in self.expected_features:
            prediction_df[cultivo_col] = 1
            print(f"✓ Cultivo '{cultivo_nombre}' configurado correctamente")
        else:
            print(f"❌ ERROR: Cultivo '{cultivo_nombre}' no está en el modelo entrenado.")
            print(f"   Cultivos disponibles: {[col.replace('cultivo_', '') for col in self.cultivo_columns[:5]]}...")
            # NO intentar reemplazar - mejor lanzar error claro
            raise ValueError(
                f"Cultivo '{cultivo_nombre}' no está en el modelo entrenado. Use uno de los cultivos disponibles.")

        # Configurar one-hot encoding para región
        region_col = f'region_{region_id}'
        if region_col in self.expected_features:
            prediction_df[region_col] = 1
            print(f"✓ Región '{region_id}' configurada correctamente")
        else:
            print(f"❌ ERROR: Región '{region_id}' no está en el modelo entrenado.")
            print(f"   Regiones disponibles: {[col.replace('region_', '') for col in self.region_columns]}")
            raise ValueError(
                f"Región '{region_id}' no está en el modelo entrenado. Use una de las regiones disponibles.")

        # Verificar que todas las columnas estén presentes
        missing_cols = set(self.expected_features) - set(prediction_df.columns)
        if missing_cols:
            print(f"Advertencia: Columnas faltantes: {missing_cols}")

        return prediction_df


def get_all_cultivos_from_db(data_loader):
    """Obtiene todos los cultivos de la base de datos"""
    try:
        cultivo_df = data_loader.load_crop_data()
        return sorted(cultivo_df['nombre_cultivo'].unique().tolist())
    except:
        # Lista de respaldo basada en los datos que proporcionaste
        return [
            "Aceituna", "Acelga", "Achiote", "Aguaje", "Aguaymanto", "Ají", "Ajo",
            "Albahaca", "Alcachofa", "Alfalfa", "Algodón", "Algodón rama", "Anona",
            "Apio", "Arandanos", "Arracacha", "Arroz", "Arroz cáscara",
            "Arveja grano seco", "Arveja grano verde", "Avena forrajera", "Avena grano",
            "Berenjena", "Betarraga", "Braquearia", "Brócoli", "Cacao", "Café pergamino",
            "Caigua", "Caimito", "Calabaza", "Camote", "Camu camu", "Caña para alcohol",
            "Caña para azúcar", "Caña para etanol", "Cañihua", "Capuli", "Carambola",
            "Cebada forrajera", "Cebada grano", "Cebolla china", "Cebolla cabeza",
            "Cerezo", "Chirimoya", "Cirolero", "Ciruela", "Coco", "Cocona", "Col",
            "Coliflor", "Copoazú", "Culantro", "Damasco", "Dátil", "Espárrago",
            "Espinaca", "Fresa", "Frijol castilla", "Frijol de palo", "Frijol loctao",
            "Frijol seco", "Frijol verde", "Garbanzo", "Grama azul", "Grama chilena",
            "Gramalote", "Granada", "Granadilla", "Guanabana", "Guayaba", "Guinda",
            "Haba seca", "Haba verde", "Higo", "Huasai", "Kiwicha", "Lechuga",
            "Lenteja", "Lima", "Limón dulce", "Limón sutíl", "Lucuma", "Maca",
            "Maíz a. duro", "Maíz amiláceo", "Maíz chala", "Maíz choclo", "Maíz morado",
            "Mamey", "Mandarina", "Mango", "Maní", "Manzana", "Maracuyá", "Marañón",
            "Mashua", "Melocotón", "Melón", "Membrillo", "Nabo", "Naranja", "Nispero",
            "Nuez", "Oca", "Olluco", "Orégano", "Pacae", "Pallar seco", "Pallar verde",
            "Palma aceitera", "Palta", "Pan de árbol", "Papa", "Papaya", "Páprika",
            "Pasto elefante", "Pecana", "Pepinillo", "Pepino", "Pera", "Perejil",
            "Pijuayo", "Pimiento", "Piña", "Piquillo", "Pituca", "Plátano", "Pomarrosa",
            "Poro", "Quinua", "Rabanito", "Rocoto", "Rye grass", "Sacha inchi", "Sandía",
            "Sauco", "Sorgo grano", "Soya", "Tamarindo", "Tangelo", "Taperibá", "Tarhui",
            "Té", "Tomate", "Toronja", "Trebol", "Trigo", "Tumbo", "Tuna", "Umari",
            "Ungurahui", "Uva", "Vainita", "Yuca", "Zanahoria", "Zapallo", "Zapote", "Zarandaja"
        ]