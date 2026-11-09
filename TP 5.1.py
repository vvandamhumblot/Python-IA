import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


data = load_breast_cancer()
X = data.data        # 569 patientes x 30 mesures
y = data.target      # 0 = maligne, 1 = bénigne

reseau = MLPClassifier(hidden_layer_sizes=(15,), max_iter=3250)
reseau.fit(X, y)

print("Précision :", reseau.score(X, y))