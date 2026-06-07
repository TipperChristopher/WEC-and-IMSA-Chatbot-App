# db/db_manager.py
import sqlite3

def setup_database():
    conn = sqlite3.connect("trackside_timing.db")
    cursor = conn.cursor()
    
    # Table for normalized cross-series class mapping
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS class_mapping (
            series_code TEXT,      -- 'wec' or 'imsa'
            raw_class TEXT,        -- 'GTP', 'LMH', 'GTD Pro', 'LMGT3'
            normalized_class TEXT  -- 'PROTOTYPE', 'GT'
        )
    """)
    
    # Populate mapping configurations
    mappings = [
        ('wec', 'GTP', 'PROTOTYPE'),
        ('wec', 'LMH', 'PROTOTYPE'),
        ('wec', 'GTD Pro', 'GT'),
        ('wec', 'LMGT3', 'GT'),
        ('imsa', 'GTP', 'PROTOTYPE'),
        ('imsa', 'LMH', 'PROTOTYPE'),
        ('imsa', 'GTD Pro', 'GT'),
        ('imsa', 'LMGT3', 'GT')
    ]
    cursor.executemany("INSERT INTO class_mapping VALUES (?,?,?)", mappings)
    
    # Main laps and timing table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS laps (
            series_code TEXT,
            event TEXT,
            car_number TEXT,
            class TEXT,
            driver_name TEXT,
            lap_number INTEGER,
            lap_time_s REAL,
            s1_s REAL,
            s2_s REAL,
            s3_s REAL,
            pit_time_s REAL
        )
    """)
    conn.commit()
    conn.close()