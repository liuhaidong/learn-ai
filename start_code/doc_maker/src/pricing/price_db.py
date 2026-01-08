import sqlite3
from typing import Optional, Dict
from pathlib import Path
import os


class PriceDatabase:
    def __init__(self, db_path: str = "data/prices/price_db.sqlite"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_db (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material TEXT NOT NULL,
                spec TEXT,
                unit TEXT,
                price REAL NOT NULL,
                source TEXT,
                updated DATE DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_price(self, material: str, spec: str, unit: str, price: float, source: str = "manual"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO price_db (material, spec, unit, price, source)
            VALUES (?, ?, ?, ?, ?)
        ''', (material, spec, unit, price, source))
        conn.commit()
        conn.close()
    
    def query_price(self, material: str, spec: Optional[str] = None) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if spec:
            cursor.execute('''
                SELECT material, spec, unit, price, source, updated
                FROM price_db
                WHERE material = ? AND spec = ?
                ORDER BY updated DESC
                LIMIT 1
            ''', (material, spec))
        else:
            cursor.execute('''
                SELECT material, spec, unit, price, source, updated
                FROM price_db
                WHERE material = ?
                ORDER BY updated DESC
                LIMIT 1
            ''', (material,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "material": result[0],
                "spec": result[1],
                "unit": result[2],
                "price": result[3],
                "source": result[4],
                "updated": result[5]
            }
        return None
    
    def get_all_prices(self) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM price_db')
        results = cursor.fetchall()
        conn.close()
        return results
