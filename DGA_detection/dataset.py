import torch
import torch.nn.utils.rnn


# Класс для преобразования символов в индексы и обратно
class Char2Index():
    def __init__(self, dataset):
        self.char2index = {} # Словарь символ -> индекс
        self.index2char = {} # Словарь индекс -> символ
        for item in dataset['domain']:
            for char in item:
                if char not in self.char2index:
                    self.char2index[char] = len(self.char2index)
                    self.index2char[len(self.index2char)] = char
    
    def __len__(self):
        return len(self.char2index)
    
    # Преобразование строки в последовательность индексов
    def string2index(self, string):
        return [self.char2index[char] for char in string]
    
    # Преобразование последовательности индексов обратно в строку
    def index2string(self, index):
        return ''.join([self.index2char[i] for i in index])


class Dataset(torch.utils.data.Dataset):
    """
    Параметры:
    - data (DataFrame): Датасет с данными.
    - train_or_test (str): Указывает, используется ли датасет для обучения ('train') или тестирования ('test').
    """
    def __init__(self, data, train_or_test='train'):
        self.data = data
        self.train_or_test = train_or_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Возвращаем данные и метки для обучения, только данные для тестирования
        if self.train_or_test == 'train':
            input = self.data['domain_index'][idx]
            label = self.data['is_dga'][idx]
            return torch.tensor(input), torch.tensor(label)
        elif self.train_or_test == 'test':
            input = self.data['domain_index'][idx]
            return torch.tensor(input)
        

# Функция для обработки батчей 
def collate_fn(batch, is_train=True):
    """
    Обрабатывает батчи данных перед передачей их в модель.

    Параметры:
    - batch (list): Список образцов, полученных из датасета.
    - is_train (bool): Указывает, используется ли функция для обучения (True) или тестирования (False).

    Возвращает:
    - Для обучения: padding данных и метки классов.
    - Для теста: только padding данных.
    """    
    if is_train:
        domains, labels = zip(*batch)
        domains = torch.nn.utils.rnn.pad_sequence(domains, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        return domains, labels
    else:
        domains = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        return domains
