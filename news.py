# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # Carregar dados
# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     # Criar uma coluna 'content' concatenando 'titulo' e 'introducao'
#     df['content'] = df['titulo'].fillna('') + " " + df['introducao'].fillna('')
#     return df
#
# # Função para determinar se a notícia é negativa
# def is_negative(news_content, keywords):
#     return any(keyword in news_content.lower() for keyword in keywords)
#
# # Preparar os dados
# def prepare_data(df, keywords):
#     df['label'] = df['content'].apply(lambda x: is_negative(x, keywords))
#     vectorizer = TfidfVectorizer(max_features=1000)
#     features = vectorizer.fit_transform(df['content'])
#     labels = df['label'].values.astype(int)
#     return features, labels
#
# # Definição da rede neural
# class NewsClassifier(nn.Module):
#     def __init__(self):
#         super(NewsClassifier, self).__init__()
#         self.fc1 = nn.Linear(1000, 50)
#         self.fc2 = nn.Linear(50, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.sigmoid(self.fc2(x))
#         return x
#
# # Principal função para executar o processo
# def main():
#     file_path = 'data/news.csv'
#     keywords = ["queimadas", "incêndios", "vazamento de gás", "poluição"]
#
#     news_data = load_data(file_path)
#     X, y = prepare_data(news_data, keywords)
#     X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=42)
#
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
#
#     model = NewsClassifier()
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#
#     model.train()
#     for epoch in range(100):
#         optimizer.zero_grad()
#         outputs = model(X_train_tensor)
#         loss = criterion(outputs, y_train_tensor)
#         loss.backward()
#         optimizer.step()
#         print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
#
# if __name__ == '__main__':
#     main()


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Carregar dados
def load_data(file_path):
    return pd.read_csv(file_path, encoding='utf-8')

# Função para determinar se a notícia é negativa
def is_negative(news_content, keywords):
    news_content = str(news_content)
    return any(keyword in news_content.lower() for keyword in keywords)

# Preparar os dados
def prepare_data(df, keywords):
    df['label'] = df['introducao'].apply(lambda x: is_negative(x, keywords))
    print("Distribution of labels:", df['label'].value_counts(normalize=True))
    vectorizer = TfidfVectorizer(max_features=1000)
    features = vectorizer.fit_transform(df['introducao'].fillna('').astype(str))
    labels = df['label'].values.astype(int)
    return features, labels, vectorizer

# Definição da rede neural
class NewsClassifier(nn.Module):
    def __init__(self, input_size):
        super(NewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Principal função para executar o processo
def main():
    file_path = 'data/generated_news_dataset.csv'
    keywords = [
        "queimadas", "incêndios", "vazamento de gás", "poluição",
        "desmatamento", "derramamento de óleo", "contaminação",
        "extinção de espécies", "erosão", "assoreamento"
    ]

    news_data = load_data(file_path)
    X, y, vectorizer = prepare_data(news_data, keywords)
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=42)

    model = NewsClassifier(X_train.shape[1])
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predicted_labels = (predictions > 0.5).float().numpy()
        good_news = (1 - predicted_labels).sum()
        bad_news = predicted_labels.sum()
        total = len(predicted_labels)
        print(f"Percentage of Good News: {good_news / total * 100:.2f}%")
        print(f"Percentage of Bad News: {bad_news / total * 100:.2f}%")

if __name__ == '__main__':
    main()