import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Параметры модели
data_path = 'data.csv'
train_test_split_ratio = 0.8
window_size = 30
batch_size = 32
epochs = 50
learning_rate = 0.001
prediction_length = 1
order_size = 1000
profit_margin = 0.05

# Загрузка данных
data = pd.read_csv(data_path)
data = data.dropna()
data = data.drop('date', axis=1)
scaler = StandardScaler()
data = scaler.fit_transform(data)
X, y = [], []
for i in range(window_size, len(data)):
    X.append(data[i-window_size:i])
    y.append(data[i, 0])
X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_test_split_ratio, shuffle=False)

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, activation='relu', return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(prediction_length)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

# Обучение модели
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

# Прогнозирование цены
last_window = data[-window_size:]
last_window = np.array(last_window).reshape(1, window_size, -1)
prediction = model.predict(last_window)
predicted_price = scaler.inverse_transform(prediction)[0, 0]

# Открытие позиции
position = None
if predicted_price > data[-1, 0]:
    position = 'LONG'
    order_price = data[-1, 0] + profit_margin * data[-1, 0]
    order_quantity = order_size // order_price
elif predicted_price < data[-1, 0]:
    position = 'SHORT'
    order_price = data[-1, 0] - profit_margin * data[-1, 0]
    order_quantity = order_size // order_price
if position is not None:
    print(f'Opening {position} position at price {order_price} with quantity {order_quantity}')

# Закрытие позиции
if position == 'LONG' and data[-1, 0] >= order_price:
    profit = order_quantity * (data[-1, 0] - order_price)
    print(f'Closing LONG position with profit {profit}')
    position = None
elif position == 'SHORT' and data[-1, 0] <= order_price:
    profit = order_quantity * (order_price - data[-1, 0])
    print(f'Closing SHORT position with profit {profit}')
    position = None
