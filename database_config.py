# database_config.py
DB_CONFIG = {
    'server': '161.132.67.82',
    'database': 'GESTION-AGRICOLA',
    'username': 'sa',
    'password': '@4dmiN123456',
    'driver': 'ODBC Driver 17 for SQL Server'
}

CONNECTION_STRING = (
    f"DRIVER={DB_CONFIG['driver']};"
    f"SERVER={DB_CONFIG['server']};"
    f"DATABASE={DB_CONFIG['database']};"
    f"UID={DB_CONFIG['username']};"
    f"PWD={DB_CONFIG['password']}"
)