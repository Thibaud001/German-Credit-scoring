from Toolkit import *
from sklearn.ensemble import RandomForestClassifier

importer = DataImporter()
df_test, df_train, df_submission = importer.load_data()

X = df_train.drop(columns=["Risk"]).copy()
y = df_train["Risk"].copy()

y_submit = df_test[["Id"]].copy()
X_submit = df_test.drop(columns=["Id"])

encoder = Encode(X)
labeler = Label(y)
X_encoded = encoder.dummying()
y_labeled = labeler.labeling()

splitter = Split(X_encoded, y_labeled)
X_train, X_test, y_train, y_test = splitter.splitting(test_size=0.1, random_state=42)

encoder = Encode(X_submit)
X_submit = encoder.dummying()

preds = np.zeros((1001, 1))

n_estimators = [292, 342, 338, 286, 276, 313, 313, 281, 325, 383]
max_depth = [10, 11, 11, 20, 12, 17, 17, 13, 10, 16]
min_samples_split = [6, 11, 9, 10, 8, 8, 8, 9, 9, 4]
min_samples_leaf = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
thresholds = [0.3994826628152474, 0.3891266965839575, 0.3860848792441428, 0.3638924813515165,
              0.38919365651080895, 0.3859867604047514, 0.38877980561301817, 0.38855480487187366,
              0.37131280440102843, 0.3819227730724812]

# Parcourir les 10 meilleurs modèles
for i in range(10):
    rf = RandomForestClassifier(n_estimators=n_estimators[i], max_depth=max_depth[i],
                               min_samples_split=min_samples_split[i], min_samples_leaf=min_samples_leaf[i],
                               max_features=None, bootstrap=True, class_weight='balanced', random_state=42)

    rf.fit(X_train, y_train)

    y_pred_proba = rf.predict_proba(X_submit)[:, 1]
    print(y_pred_proba.shape)
    threshold = thresholds[i]
    y_pred = np.where(y_pred_proba > threshold, 1, 0)

    preds += y_pred.reshape(-1, 1)

y_submit['Risk'] = np.where(preds > 5, 'Risk', 'No Risk')

# Sauvegarder les résultats dans un fichier CSV
y_submit.to_csv('predictions_result_15.csv', index=False)

print("Fichier CSV enregistré avec succès.")


