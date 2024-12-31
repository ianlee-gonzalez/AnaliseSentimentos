import tkinter as tk
from tkinter import messagebox, filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import pickle

# Dados iniciais
data = {
    "feedback": [
        "Excelente produto", "Horrível experiência", "Muito bom atendimento",
        "Não gostei do serviço", "Produto de ótima qualidade", "Entrega atrasada",
        "Adorei o design", "Péssimo suporte", "Muito útil", "Não funcionou como esperado",
        "Ótimo preço", "Terrível atendimento", "Boa compra", "Nunca mais volto",
        "Recomendo demais", "Decepção total", "Bom custo-benefício", "Horrível entrega"
    ],
    "sentimento": [
        "positivo", "negativo", "positivo", "negativo", "positivo", "negativo",
        "positivo", "negativo", "positivo", "negativo", "positivo", "negativo",
        "positivo", "negativo", "positivo", "negativo", "positivo", "negativo"
    ]
}

# Treinamento inicial do modelo
df = pd.DataFrame(data)
df["sentimento"] = df["sentimento"].map({"positivo": 1, "negativo": 0})
vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["feedback"])
y = df["sentimento"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = SVC(C=1.0, kernel='linear', gamma='scale')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Funções auxiliares
def predict_feedback():
    feedback = feedback_entry.get("1.0", tk.END).strip()
    if feedback:
        vectorized_input = vectorizer.transform([feedback])
        prediction = model.predict(vectorized_input)[0]
        sentiment = "Positivo" if prediction == 1 else "Negativo"
        messagebox.showinfo("Resultado", f"O sentimento previsto é: {sentiment}")
    else:
        messagebox.showwarning("Atenção", "Por favor, insira um feedback.")

def show_metrics():
    report = classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"])
    messagebox.showinfo("Métricas do Modelo", f"Acurácia: {accuracy:.2f}\n\n{report}")

def retrain_model():
    new_feedback = new_feedback_entry.get("1.0", tk.END).strip()
    new_label = sentiment_var.get()
    if new_feedback and new_label in ["Positivo", "Negativo"]:
        label = 1 if new_label == "Positivo" else 0
        df.loc[len(df)] = [new_feedback, label]
        global X, y, X_train, X_test, y_train, y_test, model, y_pred, accuracy
        X = vectorizer.fit_transform(df["feedback"])
        y = df["sentimento"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Re-treinado", "O modelo foi re-treinado com os novos dados!")
    else:
        messagebox.showwarning("Atenção", "Insira um feedback válido e selecione um sentimento.")

def save_model():
    file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])
    if file_path:
        with open(file_path, 'wb') as file:
            pickle.dump((vectorizer, model), file)
        messagebox.showinfo("Salvo", "Modelo salvo com sucesso!")

# Interface Gráfica
root = tk.Tk()
root.title("Análise de Sentimento")
root.geometry("600x500")

# Entrada de feedback
tk.Label(root, text="Digite seu feedback:", font=("Arial", 12)).pack(pady=10)
feedback_entry = tk.Text(root, height=4, width=50)
feedback_entry.pack(pady=5)

# Botões principais
tk.Button(root, text="Prever Sentimento", command=predict_feedback, bg="blue", fg="white", font=("Arial", 12)).pack(pady=10)
tk.Button(root, text="Ver Métricas do Modelo", command=show_metrics, bg="green", fg="white", font=("Arial", 12)).pack(pady=10)

# Re-treinar modelo
tk.Label(root, text="Re-treinar o modelo com novos dados", font=("Arial", 12)).pack(pady=10)
new_feedback_entry = tk.Text(root, height=2, width=50)
new_feedback_entry.pack(pady=5)
sentiment_var = tk.StringVar(value="Positivo")
tk.Radiobutton(root, text="Positivo", variable=sentiment_var, value="Positivo").pack(side=tk.LEFT, padx=20)
tk.Radiobutton(root, text="Negativo", variable=sentiment_var, value="Negativo").pack(side=tk.LEFT)
tk.Button(root, text="Re-treinar Modelo", command=retrain_model, bg="orange", fg="black", font=("Arial", 12)).pack(pady=10)

# Salvar modelo
tk.Button(root, text="Salvar Modelo", command=save_model, bg="purple", fg="white", font=("Arial", 12)).pack(pady=10)

# Iniciar o aplicativo
root.mainloop()
