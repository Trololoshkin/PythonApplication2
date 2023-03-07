import pandas as pd
import numpy as np
import tensorflow as tf

def train_network(data_path='data.csv', train_test_split_ratio=0.8, window_size=10, batch_size=32, epochs=50, learning_rate=0.001, prediction_length=1, order_size=1, profit_margin=0.05):
    # Загрузка и предобработка данных
    data = pd.read_csv(data_path)
    data = data[['Close']]
    data = np.array(data).reshape(-1)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Разделение данных на обучающую и тестовую выборки
    train_size = int(len(data) * train_test_split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Создание окон для обучения нейронной сети
    def create_window(data, window_size):
        X = []
        y = []
        for i in range(len(data) - window_size - prediction_length + 1):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size:i+window_size+prediction_length])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_window(train_data, window_size)
    X_test, y_test = create_window(test_data, window_size)
    
    # Создание и обучение нейронной сети
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(window_size,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(prediction_length)
    ])
    
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test
