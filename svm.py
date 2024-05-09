import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(file_path):
    data = pd.read_csv(file_path)
    data['text'] = data['title'] + " " + data['introducao']  # Combining title and introduction
    return data


def prepare_data(df):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(df['text']).toarray()
    y = pd.factorize(df['label'])[0]  # Encode the labels as integers
    return X, y, tfidf


def plot_metrics(metrics, title='Model Evaluation Metrics'):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    plt.bar(labels, metrics, color=['blue', 'green', 'red', 'purple'])
    plt.title(title)
    plt.ylim([0, 1])
    plt.ylabel('Score')
    plt.savefig('evaluation_metrics.png')
    plt.close()


def main():
    file_path = 'data/generated_news_dataset.csv'
    data = load_data(file_path)
    X, y, vectorizer = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', random_state=42)  # Using linear kernel for simplicity
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Extracting metrics
    # accuracy = report['accuracy']
    # precision = report['weighted avg']['precision']
    # recall = report['weighted avg']['recall']
    # f1 = report['weighted avg']['f1-score']
    #
    # metrics = [accuracy, precision, recall, f1]
    # plot_metrics(metrics)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Suponha que este código esteja no final do script svm.py após a avaliação do modelo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


if __name__ == '__main__':
    main()
