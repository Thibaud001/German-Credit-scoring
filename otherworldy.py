from Toolkit import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import seaborn as sns
from sklearn.model_selection import cross_val_score
import optuna

importer = DataImporter()
df_test, df_train, df_submission = importer.load_data()

X = df_train.drop(columns=["Risk"]).copy()
y = df_train["Risk"].copy()

encoder = Encode(X)
labeler = Label(y)
X_encoded = encoder.dummying()
y_labeled = labeler.labeling()

splitter = Split(X_encoded, y_labeled)
X_train, X_test, y_train, y_test = splitter.splitting(test_size=0.1, random_state=42)


