import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd

y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)
df_conf_norm = df_confusion / df_confusion.sum(axis=1)
print(df_conf_norm)

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

def test(df_conf_norm, df_confusion):
    fig, ax = plt.subplots(figsize=(10,8))
    cb = ax.imshow(df_conf_norm, cmap=plt.cm.gray_r)

    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

    for i in range(len(df_confusion)):
        for j in range(len(df_confusion)):
            color='green' if df_conf_norm[i,j] < 0.5 else 'white'
            ax.annotate(f'{df_confusion[i,j]}', (i,j),
                        color=color, va='center', ha='center')

    plt.colorbar(cb, ax=ax)

test(df_conf_norm,df_confusion)

#plot_confusion_matrix(df_confusion)
#plot_confusion_matrix(df_conf_norm)
plt.show()


exit()


# Normalize color coding

accuracies = conf_mat/conf_mat.sum(1)
fig, ax = plt.subplots(figsize=(10,8))
cb = ax.imshow(accuracies, cmap='Greens')
plt.xticks(range(len(classes)), classes,rotation=90)
plt.yticks(range(len(classes)), classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        color='green' if accuracies[i,j] < 0.5 else 'white'
        ax.annotate(f'{conf_mat[i,j]}', (i,j),
                    color=color, va='center', ha='center')

plt.colorbar(cb, ax=ax)
plt.show()

exit()


# Generate some example data
X, y = make_moons(noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)

# Train the classifier
clf = LogisticRegression()

clf.fit(X, y)

plot_confusion_matrix(clf, X_test, y_test); plt.title("Not normalized");
#plot_confusion_matrix(clf, X_test, y_test, values_format= '.0%', normalize='true'); plt.title("normalize='true'");
#plot_confusion_matrix(clf, X_test, y_test, values_format= '.0%', normalize='pred'); plt.title("normalize='pred'");
#plot_confusion_matrix(clf, X_test, y_test, values_format= '.0%', normalize='all'); plt.title("normalize='all'");

plt.show()
exit()





# Normalise entire table
