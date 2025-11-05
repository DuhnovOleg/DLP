import pandas as pd # библиотека pandas нужна для работы с данными
import matplotlib.pyplot as plt # matplotlib для построения графиков
import numpy as np # numpy для работы с векторами и матрицами

def neuron(w,x):
    if((w[1]*x[0]+w[2]*x[1]+w[3]*x[2]+w[0])>=0):
        predict = 1
    else: 
        predict = -1
    return predict

def train_perceptron(X, y):
    w = np.random.random(4)
    eta = 0.01  # скорость обучения
    w_iter = [] # пустой список, в него будем добавлять веса, чтобы потом построить график
    for xi, target, j in zip(X, y, range(X.shape[0])):
        predict = neuron(w,xi)
        w[1:] += (eta * (target - predict)) * xi # target - predict - это и есть ошибка
        w[0] += eta * (target - predict)
        # каждую 10ю итерацию будем сохранять набор весов в специальном списке
        if(j%10==0):
            w_iter.append(w.tolist())
    return w, w_iter

def print_error(X, y, w):
    sum_err = 0
    for xi, target in zip(X, y):
        predict = neuron(w,xi)
        sum_err += (target - predict)/2

    print("Всего ошибок: ", sum_err)

    correct = 0
    total = len(X)
    for xi, target in zip(X, y):
        predict = neuron(w, xi)
        if predict == target:
            correct += 1

    accuracy = correct / total * 100
    print(f"\nТочность классификации: {accuracy:.2f}%")
    pass


def main() -> None:
    df = pd.read_csv('data.csv')
    y = df.iloc[:, 4].values
    y = np.where(y == "Iris-setosa", 1, -1)
    X = df.iloc[:, [0, 1, 2]].values

    w, w_iter = train_perceptron(X, y)
    print_error(X, y, w)

    xl = np.linspace(min(X[:, 0]), max(X[:, 0]))

    for i, w in zip(range(len(w_iter)), w_iter):
        yl = -(xl * w[1] + w[0]) / w[2]  # уравнение линии
        plt.plot(xl, yl)  # строим разделяющую границу
        plt.text(xl[-1], yl[-1], i, dict(size=10, color='gray'))
        plt.pause(1)

    plt.text(xl[-1] - 0.3, yl[-1], 'END', dict(size=14, color='red'))
    plt.show()

    pass

if __name__ == '__main__':
    main()