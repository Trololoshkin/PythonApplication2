import tkinter as tk
import requests

# Создаем окно приложения
app = tk.Tk()
app.title("Программа-помощник")

# Создаем функцию, которая будет обрабатывать запрос пользователя
def submit_question():
    # Получаем текст из поля ввода
    question = input_field.get()
    
    # Отправляем запрос на сервер
    response = requests.get(f"https://api.openai.com/v1/engines/davinci-codex/completions?prompt={question}&max_tokens=2048", headers={
        "Content-Type": "application/json",
        "Authorization": "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"
    })
    
    # Обновляем текст в окне вывода
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, response.json()["choices"][0]["text"])
    output_text.config(state=tk.DISABLED)

# Создаем поле для ввода вопроса и кнопку для отправки запроса
input_field = tk.Entry(app, width=50)
input_field.pack(side=tk.LEFT, padx=10, pady=10)
submit_button = tk.Button(app, text="Отправить", command=submit_question)
submit_button.pack(side=tk.LEFT, padx=10, pady=10)

# Создаем окно для вывода информации
output_text = tk.Text(app, height=20, width=80)
output_text.config(state=tk.DISABLED)
output_text.pack(side=tk.BOTTOM, padx=10, pady=10)

# Запускаем главный цикл приложения
app.mainloop()
