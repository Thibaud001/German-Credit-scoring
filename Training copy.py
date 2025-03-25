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


def objective(trial):
    # Définir les paramètres à optimiser
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Initialiser le modèle avec les paramètres suggérés
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=None,
        bootstrap=True,
        class_weight='balanced',
        random_state=42
    )

    # Entraîner le modèle
    rf.fit(X_train, y_train)

    # Prédire les probabilités
    y_pred_proba = rf.predict_proba(X_test)[:, 1]

    # Ajuster le seuil
    threshold = trial.suggest_float('threshold', 0.01, 0.5)
    y_pred = [1 if proba > threshold else 0 for proba in y_pred_proba]

    # Calculer le F1 score
    f1 = f1_score(y_test, y_pred)

    return f1

# Créer une étude Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Afficher les meilleurs paramètres
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params = study.best_params

# Afficher les meilleurs hyperparamètres
print("Best hyperparameters:", best_params)

rf = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

recall = recall_score(y_test, y_pred)
print(recall)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted no risk', 'Predicted risk'],
            yticklabels=['Actual no risk', 'Actual risk'])

plt.title('Confusion matrix on first XGB trial')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.show()
