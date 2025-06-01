from xgboost import XGBClassifier  # Importiert den XGBoost-Klassifikator für maschinelles Lernen
import pandas as pd  # Importiert pandas für die Datenverarbeitung
from sklearn.model_selection import train_test_split  # Für das Aufteilen in Trainings- und Testdaten
from sklearn.preprocessing import OneHotEncoder  # Zum Kodieren kategorialer Zielvariablen (z. B. income)
from sklearn import metrics  # Enthält Funktionen zur Bewertung des Modells

path = "/content/adult-2.csv"  # Pfad zur CSV-Datei (hier z. B. bei Google Colab)
data = pd.read_csv(path, delimiter=';')  # Lädt die CSV-Datei mit ; als Trenner

print(data.head())  # Gibt die ersten 5 Zeilen des DataFrames aus, um sich einen Überblick zu verschaffen
print("Empty columns: ", data.columns[data.isnull().any()])  # Gibt Spalten aus, die fehlende Werte enthalten

# Zielvariable 'income' soll vorhergesagt werden (z. B. <=50K oder >50K)
oe = OneHotEncoder()  # Erstellt einen OneHotEncoder
col = oe.fit_transform(data[['income']])  # Wendet One-Hot-Encoding auf die income-Spalte an
col = col.toarray()  # Wandelt das Ergebnis in ein NumPy-Array um

data = data.drop(['income'], axis = 1)  # Entfernt die Zielspalte aus dem ursprünglichen Datensatz

data = data.astype('category')  # Wandelt alle Spalten in kategoriale Datentypen um
data = data.apply(lambda x: x.cat.codes)  # Wandelt Kategorien in Ganzzahlen (Codes) um

print(data.info())  # Zeigt Infos über den bearbeiteten DataFrame (z. B. Spaltennamen, Datentypen)

# Teilt Daten in Trainings- und Testsets auf (80% Training, 20% Test)
train_data, test_data, train_col, test_col = train_test_split(
    data, col, test_size=0.2, random_state=42
)

xgb = XGBClassifier()  # Erstellt ein XGBoost-Klassifikator-Modell

xgb.fit(train_data, train_col)  # Trainiert das Modell mit Trainingsdaten und zugehörigen Zielwerten

predicted_col = xgb.predict(test_data)  # Wendet das Modell auf Testdaten an und sagt Zielvariable voraus

score = metrics.accuracy_score(test_col, predicted_col)  # Berechnet die Vorhersagegenauigkeit
print(score)  # Gibt die Genauigkeit (zwischen 0 und 1) aus
