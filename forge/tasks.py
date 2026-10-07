"""FORGE benchmark task suite: 12 diverse small tasks, zero downloads.

Every task is fully specified: NL description (what the Planner sees),
data code (injected into generated scripts; fixed seeds -> deterministic),
input shape, classes, and a success threshold. Tasks are CPU-tiny by design:
the benchmark measures the *agent loop*, not GPU throughput.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    name: str
    description: str
    n_classes: int          # 0 = regression
    input_shape: tuple
    data_code: str          # defines X_train,y_train,X_val,y_val,X_test,y_test
    target: float           # success threshold on test metric
    higher_is_better: bool = True


_SPLIT = """
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=7)
X_test, y_test = X_te, y_te
"""

TASKS: list[Task] = [
    Task(
        name="digits_8x8",
        description="Classify 8x8 grayscale digit images into 10 digit classes.",
        n_classes=10, input_shape=(8, 8, 1),
        data_code="""
from sklearn.datasets import load_digits
d = load_digits()
X = (d.images / 16.0)[..., None].astype("float32")
y = d.target.astype("int64")
""" + _SPLIT,
        target=0.93,
    ),
    Task(
        name="shapes_cnn",
        description="Classify 28x28 synthetic images as circle vs square.",
        n_classes=2, input_shape=(28, 28, 1),
        data_code="""
rng = np.random.RandomState(11)
n = 800
X = np.zeros((n, 28, 28, 1), dtype="float32"); y = np.zeros(n, dtype="int64")
yy, xx = np.mgrid[0:28, 0:28]
for i in range(n):
    cx, cy = rng.randint(8, 20, 2)
    if i % 2 == 0:
        r = rng.randint(5, 8); X[i, :, :, 0] = ((xx-cx)**2 + (yy-cy)**2) < r*r; y[i] = 0
    else:
        s = rng.randint(9, 13); X[i, cy-s//2:cy+s//2, cx-s//2:cx+s//2, 0] = 1.0; y[i] = 1
X += rng.normal(0, 0.08, X.shape).astype("float32"); X = np.clip(X, 0, 1)
""" + _SPLIT,
        target=0.95,
    ),
    Task(
        name="blobs_noisy",
        description="Classify noisy 28x28 blob images (medical-like) as benign vs suspicious.",
        n_classes=2, input_shape=(28, 28, 1),
        data_code="""
rng = np.random.RandomState(21)
n = 700
X = np.zeros((n, 28, 28, 1), dtype="float32"); y = np.zeros(n, dtype="int64")
yy, xx = np.mgrid[0:28, 0:28]
for i in range(n):
    cx, cy = rng.randint(9, 19, 2); r = rng.randint(3, 6)
    blob = ((xx-cx)**2 + (yy-cy)**2) < r*r
    if i % 2 == 1:  # suspicious: irregular edge
        blob = blob & (rng.random((28, 28)) > 0.25)
        y[i] = 1
    X[i, :, :, 0] = blob.astype("float32")
X += rng.normal(0, 0.25, X.shape).astype("float32"); X = np.clip(X, 0, 1)
""" + _SPLIT,
        target=0.9,
    ),
    Task(
        name="iris_tabular",
        description="Classify iris flowers into 3 species from 4 tabular measurements.",
        n_classes=3, input_shape=(4,),
        data_code="""
from sklearn.datasets import load_iris
d = load_iris()
X = d.data.astype("float32"); y = d.target.astype("int64")
""" + _SPLIT,
        target=0.9,
    ),
    Task(
        name="wine_tabular",
        description="Classify wines into 3 cultivars from 13 tabular chemical features.",
        n_classes=3, input_shape=(13,),
        data_code="""
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
d = load_wine()
X = StandardScaler().fit_transform(d.data).astype("float32"); y = d.target.astype("int64")
""" + _SPLIT,
        target=0.93,
    ),
    Task(
        name="breast_cancer",
        description="Classify breast tumors as malignant vs benign from 30 tabular features.",
        n_classes=2, input_shape=(30,),
        data_code="""
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
d = load_breast_cancer()
X = StandardScaler().fit_transform(d.data).astype("float32"); y = d.target.astype("int64")
""" + _SPLIT,
        target=0.93,
    ),
    Task(
        name="diabetes_regression",
        description="Predict disease progression (continuous value) from 10 tabular features. Regression task.",
        n_classes=0, input_shape=(10,),
        data_code="""
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
d = load_diabetes()
X = StandardScaler().fit_transform(d.data).astype("float32")
y = StandardScaler().fit_transform(d.target.reshape(-1, 1)).ravel().astype("float32")
""" + _SPLIT,
        target=0.55, higher_is_better=False,  # MSE on standardized target; lower better
    ),
    Task(
        name="two_moons",
        description="Classify 2D two-moons points into 2 classes.",
        n_classes=2, input_shape=(2,),
        data_code="""
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=800, noise=0.15, random_state=3)
X = X.astype("float32"); y = y.astype("int64")
""" + _SPLIT,
        target=0.78,
    ),
    Task(
        name="circles",
        description="Classify 2D concentric-circles points into 2 classes.",
        n_classes=2, input_shape=(2,),
        data_code="""
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=800, noise=0.12, factor=0.4, random_state=5)
X = X.astype("float32"); y = y.astype("int64")
""" + _SPLIT,
        target=0.9,
    ),
    Task(
        name="text_sentiment",
        description="Classify short synthetic product reviews as positive vs negative sentiment from word sequences.",
        n_classes=2, input_shape=(40,),
        data_code="""
rng = np.random.RandomState(31)
pos = ["great","love","excellent","amazing","best","wonderful","perfect","happy"]
neg = ["terrible","hate","awful","worst","bad","poor","disappointing","broken"]
neu = ["the","a","it","was","is","very","quite","really","product","item","this"]
vocab = {w: i+1 for i, w in enumerate(pos + neg + neu)}
n = 1200; L = 40
X = np.zeros((n, L), dtype="int64"); y = np.zeros(n, dtype="int64")
for i in range(n):
    lab = i % 2; y[i] = lab
    words = rng.choice(neu, size=L - 6).tolist()
    words += rng.choice(pos if lab else neg, size=6).tolist()
    rng.shuffle(words)
    X[i] = [vocab[w] for w in words[:L]]
""" + _SPLIT.replace("train_test_split(X, y", "train_test_split(X, y"),
        target=0.93,
    ),
    Task(
        name="parity_sequence",
        description="Given a binary sequence of length 12, predict its parity (even vs odd number of ones).",
        n_classes=2, input_shape=(12,),
        data_code="""
rng = np.random.RandomState(41)
n = 1500; L = 12
X = rng.randint(0, 2, size=(n, L)).astype("int64") + 1  # token ids 1..2
y = (X.sum(axis=1) % 2).astype("int64")
""" + _SPLIT,
        target=0.7,
    ),
    Task(
        name="digits_imbalanced",
        description="Classify 8x8 digit images with severe class imbalance: digit 0 is 10x more common than others.",
        n_classes=10, input_shape=(8, 8, 1),
        data_code="""
from sklearn.datasets import load_digits
d = load_digits()
X_all = (d.images / 16.0)[..., None].astype("float32"); y_all = d.target.astype("int64")
rng = np.random.RandomState(51)
keep = np.ones(len(y_all), dtype=bool)
for c in range(1, 10):
    idx = np.where(y_all == c)[0]
    drop = rng.choice(idx, size=int(0.9 * len(idx)), replace=False)
    keep[drop] = False
X, y = X_all[keep], y_all[keep]
""" + _SPLIT,
        target=0.72,
    ),
]


def get_task(name: str) -> Task:
    for t in TASKS:
        if t.name == name:
            return t
    raise KeyError(name)
