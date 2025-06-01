import pandas as pd  # Importiert Pandas zum Arbeiten mit Daten
from sklearn.model_selection import train_test_split  # Importiert Funktion, um Daten in Trainings- und Testsets aufzuteilen
from sklearn import metrics  # Importiert Modul für Bewertung von Modellen (z.B. Accuracy)
from sklearn.preprocessing import OneHotEncoder  # Importiert OneHotEncoder für kategoriale Merkmale
from xgboost import XGBClassifier  # Importiert XGBoost-Klassifikator für Klassifizierungsaufgaben

# Für Kategorien wie 'sex' (klassifikationsproblem)

path = "adult-2.csv"  # Pfad zur CSV-Datei
data = pd.read_csv(path, delimiter=';')  # CSV-Datei mit Semikolon als Trenner laden
print(data.head())  # Ausgabe der ersten 5 Zeilen zur Datenübersicht

# Zielvariable 'sex' soll vorhergesagt werden

oe = OneHotEncoder()  # Initialisiert OneHotEncoder für die Umwandlung der Zielvariable in One-Hot-Encoding
col = oe.fit_transform(data[['sex']])  # One-Hot-Encode der Spalte 'sex' (z.B. 'male' und 'female' in Binärspalten umwandeln)
col = col.toarray()  # Konvertiert das Ergebnis in ein NumPy-Array

data = data.drop(['sex'], axis=1)  # Entfernt die Spalte 'sex' aus den Eingabedaten, da sie Zielvariable ist

# Daten in Trainings- und Testdatensätze aufteilen
train_data, test_data, train_col, test_col = train_test_split(
    data, col, test_size=0.2
)
# 80% Trainingsdaten, 20% Testdaten, Eingabedaten (data) und Zielwerte (col)

xgb = XGBClassifier()  # Initialisiert den XGBoost Klassifikator mit Standardparametern

xgb.fit(train_data, train_col)  # Trainiert das Modell mit den Trainingsdaten und Zielwerten

predicted_col = xgb.predict(test_data)  # Modell sagt die Klassen für Testdaten voraus

score = metrics.accuracy_score(test_col, predicted_col)  # Berechnet die Genauigkeit (Accuracy) der Vorhersagen

print(score)  # Gibt den Accuracy-Score aus, also wie oft die Vorhersage richtig war
