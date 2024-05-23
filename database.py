import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (timestamp TEXT, class_name TEXT, confidence REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL)''')
    conn.commit()
    conn.close()

def log_detection(class_name, confidence, x1, y1, x2, y2):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''INSERT INTO detections (timestamp, class_name, confidence, x1, y1, x2, y2)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''', 
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), class_name, confidence, x1, y1, x2, y2))
    conn.commit()
    conn.close()


def get_analytics():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, class_name, COUNT(*) as count FROM detections GROUP BY class_name''')
    data = c.fetchall()
    conn.close()
    return data

def clear_table():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''DROP TABLE detections''')
    conn.close()
