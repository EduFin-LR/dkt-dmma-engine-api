import torch
import torch.nn as nn

class DKTModel(nn.Module):
    def __init__(self, num_skills, embed_dim, hidden_dim):
        super(DKTModel, self).__init__()

        # 1. Capa de Incrustación (Embedding)
        self.embedding = nn.Embedding(num_embeddings=num_skills * 2, embedding_dim=embed_dim)

        # 2. Capa LSTM (Long Short-Term Memory)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)

        # 3. Capa Lineal de Salida (Fully Connected)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=num_skills)

        # 4. Función de Activación
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x representa el tensor de entrada con las interacciones del estudiante

        # Paso 1: Convertir los IDs enteros a vectores densos (Embeddings)
        embeds = self.embedding(x)

        # Paso 2: Pasar los embeddings por la red recurrente
        lstm_out, _ = self.lstm(embeds)

        # Paso 3: Pasar el resultado de la LSTM por la capa lineal
        logits = self.fc(lstm_out)

        # Paso 4: Aplastar el resultado para obtener probabilidades
        preds = self.sigmoid(logits)

        return preds

if __name__ == "__main__":
    modelo_prueba = DKTModel(num_skills=10, embed_dim=32, hidden_dim=64)
    print("modelo compilado")
    print(modelo_prueba)