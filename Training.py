from Toolkit import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score
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
X_train, X_test, y_train, y_test = splitter.splitting(test_size=0.05, random_state=42)


def objective(trial):
    # Définir les hyperparamètres à optimiser
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 38)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])


    # Créer le modèle avec les hyperparamètres suggérés
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=-1,
        random_state=42
    )

    # Évaluer le modèle avec validation croisée
    score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)
    return score.mean()

# Créer une étude Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=120)

best_params = study.best_params

# Afficher les meilleurs hyperparamètres
print("Best hyperparameters:", best_params)

rf = RandomForestClassifier(**best_params, random_state=42)

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
