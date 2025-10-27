# data_loader.py
import pyodbc
import pandas as pd
from database_config import CONNECTION_STRING


class DataLoader:
    def __init__(self):
        self.connection_string = CONNECTION_STRING

    def get_connection(self):
        """Establece conexión con la base de datos"""
        try:
            conn = pyodbc.connect(self.connection_string)
            return conn
        except Exception as e:
            print(f"Error conectando a la base de datos: {e}")
            return None

    def load_climate_data(self):
        """Carga datos climáticos"""
        query = """
        SELECT 
            id_region,
            anio,
            mes,
            dia,
            precipitacion,
            temperatura_max,
            temperatura_min
        FROM Clima
        """
        conn = self.get_connection()
        if conn:
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        return None

    def load_planting_data(self):
        """Carga datos de siembra"""
        query = """
        SELECT 
            id_siembra,
            id_region,
            id_cultivo,
            hectareas,
            anio
        FROM Siembra
        """
        conn = self.get_connection()
        if conn:
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        return None

    def load_harvest_data(self):
        """Carga datos de cosecha"""
        query = """
        SELECT 
            id_cosecha,
            id_region,
            id_cultivo,
            toneladas,
            anio
        FROM Cosecha
        """
        conn = self.get_connection()
        if conn:
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        return None

    def load_crop_data(self):
        """Carga datos de cultivos"""
        query = """
        SELECT 
            id_cultivo,
            nombre_cultivo
        FROM Cultivo
        """
        conn = self.get_connection()
        if conn:
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        return None

    def load_all_data(self):
        """Carga todos los datos necesarios"""
        print("Cargando datos climáticos...")
        clima_df = self.load_climate_data()

        print("Cargando datos de siembra...")
        siembra_df = self.load_planting_data()

        print("Cargando datos de cosecha...")
        cosecha_df = self.load_harvest_data()

        print("Cargando datos de cultivos...")
        cultivo_df = self.load_crop_data()

        return clima_df, siembra_df, cosecha_df, cultivo_df