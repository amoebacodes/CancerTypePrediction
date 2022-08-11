# %%
import pandas as pd
import numpy as np
from pathlib import Path
# %%
"""
Reading in data...
data and label are the two variables containing necessary information
data: a pandas dataframe object with rows/index = genes, columns = expression levels of all instances
label: a list, indices correspond to the columns in data
"""
adrenalDataDir = Path("unzipped_data/adrenal_gland")
kidneyDataDir = Path("unzipped_data/kidney")
adrenalData = list(adrenalDataDir.glob("*.txt"))
kidneyData = list(kidneyDataDir.glob("*.txt"))
data = pd.DataFrame(columns = ["gene", "exp"])
label = []
idx = 0
for item in adrenalData:
    instance = pd.read_csv(str(item), header = None, sep = "\t")
    instance.columns = ["gene",f"exp{idx}"]
    if idx == 0:
        data = instance.copy()
    else:
        data = data.join(instance.set_index('gene'), on = "gene")
    idx += 1
    label.append("adrenal")

for item in kidneyData:
    instance = pd.read_csv(str(item), header = None, sep = "\t")
    instance.columns = ["gene",f"exp{idx}"]
    if idx == 0:
        data = instance.copy()
    else:
        data = data.join(instance.set_index('gene'), on = "gene")
    idx += 1
    label.append("kidney")

data = data.set_index("gene")
assert not data.isnull().values.any(), "has some nan values"
# %%
"""
Preprocessing for Neural Net
get rid of rows that are all 0
encode labels into 0 and 1
have train-test split
Commoent out this block when building neural net without preprocessing
"""
dataNp = data.to_numpy()
data = data.drop(labels = data.index[np.where(np.mean(dataNp, axis = 1) == 0)])

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encLabels = le.fit_transform(label)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.to_numpy().T, encLabels.reshape(-1,1), random_state = 22)
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)
# NOTE: x_train has dim = 212, 60483 because data is transposed to allow for train test split
# %%
"""
buildling a neural net
"""
import tensorflow as tf

batchSize = 10
model = tf.keras.Sequential([
            tf.keras.Input(x_train.shape[1]),
            tf.keras.layers.Dense(1024, activation='tanh'),
            tf.keras.layers.Dense(512, activation='tanh'),
            tf.keras.layers.Dense(256, activation='tanh'),
            tf.keras.layers.Dense(128, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ]) 
model.compile(
            optimizer = 'adam', 
            loss =tf.keras.losses.BinaryCrossentropy(), 
            metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
model.fit(x = x_train, y = y_train, batch_size = batchSize, epochs = 5)
# %%
"""
Evaluate results
"""
loss,accuracy,precision,recall = model.evaluate(x_test,y_test)
F1 = 2 * precision * recall / (precision + recall)
F1
# %%
model.predict(x_test)
# %%
from sklearn.metrics import roc_curve, precision_recall_curve
from matplotlib import pyplot as plt

precROC = roc_curve(y_test, model.predict(x_test))
plt.plot(precROC[0],precROC[1])
plt.title("roc curve")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.show()
precPRC = precision_recall_curve(y_test, model.predict(x_test))
plt.plot(precPRC[1],precPRC[0])
plt.title("precision recall curve")
plt.xlabel("recall")
plt.ylabel("precision")
plt.ylim([0,1.1])
plt.show()

#accuracy: 0.9437 - precision: 1.0000 - recall: 0.9298
#F1 == 0.9636363793601672
# 71 test data points
# 212 training data points
# 60483 genes
# %%
#dropping zeros
dataNp = data.to_numpy()
colSum = dataNp.sum(axis = 1)
dataNp = np.delete(dataNp,np.where(colSum == 0), axis = 0)
#len(dataNp) == 58233
# %%
from scipy.stats import ttest_ind
labelNp = np.array(label)
pVals = []
for row in dataNp:
    adr = row[np.where(labelNp == 'adrenal')]
    kid = row[np.where(labelNp == 'kidney')]
    t, p = ttest_ind(adr,kid)
    pVals.append(p)
pVals = np.array(pVals)
bonferroniPval = pVals * len(dataNp)
# %%
interestingTargets = np.argsort(bonferroniPval)[0:5]
geneNames = data.index[interestingTargets]
geneNamesGeneric = ["IPO9", "CTHRC1", "PDZD4", "KIAA1949","AAR2"]
# %%
plt.figure(figsize = (10,6))
idx = 1
for i in interestingTargets:
    plt.subplot(2,3,idx).title.set_text(geneNamesGeneric[idx -1])
    plt.boxplot([np.squeeze(dataNp[i,np.where(labelNp == 'adrenal')]), np.squeeze(dataNp[i,np.where(labelNp == 'kidney')])])
    plt.xticks([1,2], ["adrenal", "kidney"])
    idx += 1
plt.show()
# %%
