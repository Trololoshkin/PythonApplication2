#1 объявление переменных, и всех import
#2 функции ордеров (покупка, продажа, ждать, закывать)
#3 интерфейс, два поля для ввода информации
# 3.1текущий баланс
# 3.2 процент - будет использован в фукциях ордеров
#4 три кнопки
# 4.1 первая - загрузка файла , на которм будет проводится обчуение Нейросети
# 4.2 вторая -  подключение базы нейросети (создание, если база отсутствует). Используем SQLite.
# 4.3 третья — запуск обучения нейросети
#5 функция обучения нейросети на основе фала из пукта 4.1, и сохранение базы нейронной сети в файле указанном в пункте 4.2.
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sqlite3
import time
import csv
import os

# Путь к базе данных
database_path = ""

# Создание таблицы в базе данных, если она не существует
def create_table():
    global database_path
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stocks
                 (date text, trans text, symbol text, qty real, price real)''')
    conn.commit()
    conn.close()

# Загрузка данных из CSV-файла в базу данных
def load_csv_data(csv_path):
    global database_path
    try:
        conn = sqlite3.connect(database_path)
        c = conn.cursor()
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                c.execute("INSERT INTO stocks VALUES (?, ?, ?, ?, ?)", (row[0], row[1], row[2], row[3], row[4]))
        conn.commit()
        conn.close()
        return True
    except:
        return False

# Создание или выбор базы данных
def choose_database():
    global database_path
    database_path = filedialog.askopenfilename()
    if not os.path.exists(database_path):
        conn = sqlite3.connect(database_path)
        conn.close()
        create_table()
    else:
        create_table()

# Функция обучения нейросети
def train_network(database_path, status_label):
    try:
        conn = sqlite3.connect(database_path)
        df = pd.read_sql_query("SELECT * from stocks", conn)
        conn.close()

        # Создание датасета для обучения
        data = []
        for i in range(len(df) - 1):
            d = np.array([df['qty'][i], df['price'][i], df['qty'][i + 1]])
            data.append(d)

        # Нормализация датасета
        data_norm = []
        for i in data:
            norm = i / np.linalg.norm(i)
            data_norm.append(norm)

        # Подготовка обучающих данных
        x_train = []
        y_train = []
        for i in range(len(data_norm) - 1):
            x_train.append(data_norm[i])
            y_train.append(data_norm[i + 1])

        # Преобразование данных в тензоры PyTorch
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)

        # Создание нейросети
        input_size = 3
        hidden_size = 5
        output_size = 3
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

        # Обучение нейросети
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        num_epochs = 1000
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
        status_label.config(text="Neural network successfully trained")
    except
