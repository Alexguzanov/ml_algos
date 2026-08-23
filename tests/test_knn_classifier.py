import pandas as pd

from my_ml_lib.metrics import accuracy_score
from my_ml_lib.model_selection import train_test_split
from my_ml_lib.neighbors import KNNClassifier

RANDOM_SEED = 42


iris_path = "datasets/iris.csv"

df = pd.read_csv(iris_path)
X = df.drop("virginica", axis=1)
y = df["virginica"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

for k in range(1, 10):
    clf = KNNClassifier(k=k, p=k % 2 + 1)

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    # print(preds)
    # print(np.asarray(y_test))
    print(f"Accuracy for {k} nearest neighbours: {accuracy_score(preds, y_test)}")
