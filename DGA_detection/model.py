import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        """
        Параметры:
        - vocab_size (int): Размер словаря (количество уникальных символов).
        - embedding_dim (int): Размерность эмбеддинга.
        - hidden_dim (int): Размерность скрытого слоя.
        - output_dim (int): Размерность выходного слоя (количество классов для классификации).
        - n_layers (int): Количество слоев GRU.
        - bidirectional (bool): Если True, модель будет двунаправленной.
        - dropout (float): dropout для регуляризации.
        """

        super(GRU, self).__init__()
        # Преобразование в эмбеддинг
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Слой GRU
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        # hidden_dim * 2 используется, так как GRU двунаправленная
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Параметры:
        - x (Tensor): Входные данные (индексы символов).

        Возвращает:
        - output (Tensor): Предсказания модели.
        """        
        # Преобразование входных данных в эмбеддинг и применение Dropout
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)
        # Объединение скрытых состояний из двух направлений
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)
