import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model import GRU  
from dataset import Char2Index, Dataset, collate_fn  


def read_model_and_create_validation():
    """
    Функция для загрузки модели, выполнения валидации и записи метрик в файл.

    Возвращает:
    - bool: Возвращает True при завершении работы.
    """    
    df = pd.read_csv('train.csv')
    char2index = Char2Index(df)

    df_val = pd.read_csv('val.csv')
    df_val['domain_index'] = df_val['domain'].apply(lambda x: char2index.string2index(x))

    # for validation.txt
    val_dataset = Dataset(df_val)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True, collate_fn=lambda x: collate_fn(x, is_train=True))
    
    # Загрузка модели 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gru_model = torch.load('./gru_model.pth')
    gru_model.eval()
    targets = []
    preds = []

    for input, label in tqdm(val_iter):
        input = input.to(device)
        label = label.to(device)
        output = gru_model(input)
        pred = torch.argmax(output, dim=1)
        targets.extend(label.tolist())
        preds.extend(pred.tolist())

    # Подсчет метрик
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    # Запись метрик в файл
    with open('validation.txt', 'w') as f:
        f.write(f'True positive: {tp}\n')
        f.write(f'False positive: {fp}\n')
        f.write(f'False negative: {fn}\n')
        f.write(f'True negative: {tn}\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1: {f1:.4f}\n')

    return True


if __name__ == '__main__':
    result = read_model_and_create_validation()
    print(f"validate complete - {result}")