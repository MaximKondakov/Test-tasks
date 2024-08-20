import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model import GRU 
from dataset import Char2Index, Dataset, collate_fn  


# Функция для создания обучающего и тестового наборов данных
def create_dataset(df):
    """
    Разделяет данные на обучающую и тестовую выборки, создает DataLoader для каждой выборки.

    Параметры:
    - df (DataFrame): Датасет, содержащий доменные имена и метки.

    Возвращает:
    - train_iter (DataLoader): DataLoader для обучающей выборки.
    - test_iter (DataLoader): DataLoader для тестовой выборки.
    """   
    # Разделяем данные на обучающую и тестовую выборки
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    # Создаем DataLoader для итерирования обучающей и тестовой выборок
    train_dataset = Dataset(train_data)
    test_dataset = Dataset(test_data)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=lambda x: collate_fn(x, is_train=True))
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True, collate_fn=lambda x: collate_fn(x, is_train=True))
    
    return train_iter, test_iter 


def train_and_save_model():
    """
    Обучает модель GRU на тренировочных данных, валидирует на тестовых данных,
    и сохраняет лучшую модель по точности (accuracy).

    Возвращает:
    - bool: Возвращает True при завершении обучения.
    """

    df = pd.read_csv('train.csv')
    # Словарь для преобразования символов в индексы
    char2index = Char2Index(df)    
    df['domain_index'] = df['domain'].apply(lambda x: char2index.string2index(x))
    
    # Создаем обучающую и тестовую выборки
    train_iter, test_iter = create_dataset(df)

    # Инициализируем модель GRU
    gru_model = GRU(vocab_size=len(char2index), embedding_dim=128, hidden_dim=256, output_dim=2, n_layers=2, bidirectional=True, dropout=0.2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(gru_model.parameters(), lr=0.002)

    early_stop = 0
    epochs = 5
    min_val_loss = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gru_model = gru_model.to(device)

    # Основной цикл обучения
    for epoch in range(epochs):
        gru_model.train()
        loss_sum = 0
        for input, label in tqdm(train_iter):
            input = input.to(device)
            label = label.to(device)

            output = gru_model(input)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f'epoch: {epoch}, train-loss: {loss_sum/len(train_iter)}')

        # Оценка модели на тестовых данных
        gru_model.eval()
        loss_sum = 0
        correct = 0
        total = 0
        for input, label in tqdm(test_iter):
            input = input.to(device)
            label = label.to(device)
            output = gru_model(input)
            loss = loss_fn(output, label)
            loss_sum += loss.item()
            pred = torch.argmax(output, dim=1)
            correct += (pred == label).sum().item()
            total += len(label)

        val_loss = loss_sum/len(test_iter)
        print(f'epoch: {epoch}, val-loss: {val_loss}, acc: {correct/total}')
        
        # early stop
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            early_stop = 0
            torch.save(gru_model, 'gru_model.pth')
            print('save model')
        else:
            early_stop += 1
            if early_stop == 2:
                print('early stop')
                break

    return True


if __name__ == '__main__':
    result = train_and_save_model()
    print(f"train complete - {result}")