import pandas as pd
import torch
from tqdm import tqdm
from model import GRU  
from dataset import Char2Index, Dataset, collate_fn 


def read_model_create_predict():
    """
    Функция для загрузки модели, выполнения предсказаний и сохранения их в файл

    Возвращает:
    - bool: Возвращает True при завершении работы.
    """        
    # Данные для создания словаря символов
    df = pd.read_csv('train.csv')
    char2index = Char2Index(df)

    df_test = pd.read_csv('test.csv')
    df_test['domain_index'] = df_test['domain'].apply(lambda x: char2index.string2index(x))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Загрузка  модели
    gru_model = torch.load('./gru_model.pth')
    gru_model.eval()    
    
    # Подготовка тестового набора данных для предсказаний
    pred_dataset = Dataset(df_test, train_or_test = 'test')
    pred_iter = torch.utils.data.DataLoader(pred_dataset, batch_size=256, shuffle=False, collate_fn=lambda x: collate_fn(x, is_train=False))
    predictions = []

    with torch.no_grad():
        for input in tqdm(pred_iter):        
            input = input.to(device)
            output = gru_model(input)       
            pred = torch.argmax(output, dim=1)
            predictions.extend(pred.cpu().numpy())


    df_test['is_dga'] = predictions
    # Сохранение предсказаний в файл CSV
    output_path = 'prediction.csv'
    df_test[['domain', 'is_dga']].to_csv(output_path, index=False)
    return True


if __name__ == "__main__":
    result = read_model_create_predict()
    print(f"predict complete - {result}")
