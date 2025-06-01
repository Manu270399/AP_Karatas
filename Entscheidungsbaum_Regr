import pandas as pd  # Importiert die Pandas-Bibliothek zum Laden und Bearbeiten von Daten
from sklearn.model_selection import train_test_split  # Importiert Funktion zur Aufteilung der Daten in Trainings- und Testsets
from sklearn import metrics  # Importiert Module zur Bewertung von Modellen (Fehlermetriken)
from xgboost import XGBRegressor  # Importiert den XGBoost-Regressor für Regressionsaufgaben

# Für numerische Werte (Regression)

# Datei laden und erste Zeilen ausgeben
path = "adult-2.csv"  # Dateipfad zur CSV-Datei
data = pd.read_csv(path, delimiter=';')  # CSV-Datei mit Semikolon als Trennzeichen einlesen
print(data.head())  # Gibt die ersten 5 Zeilen des DataFrames aus, um einen Überblick zu bekommen

# Daten vorbereiten

# 1) Zielvariable festlegen
col_name = 'sepal.length'  # Definiert den Namen der Zielvariable (abhängige Variable)

# 2) Kategorische Spalten in numerische Codes umwandeln
conv_num = ['species']  # Liste mit den Spaltennamen, die kategorisch sind und umkodiert werden müssen
data[conv_num] = data[conv_num].astype('category')  # Wandelt diese Spalten in den 'category' Datentyp um
data[conv_num] = data[conv_num].apply(lambda x: x.cat.codes)
# Wendet eine Funktion an, die Kategorien durch numerische Codes ersetzt (z.B. 'setosa' -> 0, 'versicolor' -> 1)

# 3) Input- und Output-Daten trennen
col = data[col_name]  # Speichert die Zielvariable (die Spalte 'sepal.length') in der Variable col
data = data.drop([col_name], axis=1)  # Entfernt die Zielvariable aus den Eingabedaten, um nur Features zu behalten

# Modell aufbauen, trainieren und testen

train_data, test_data, train_col, test_col = train_test_split(
    data, col, test_size=0.2
)
# Teilt die Daten in Trainings- (80%) und Testsets (20%) auf; die Eingabedaten (data) und Zielwerte (col)

xgb = XGBRegressor()  # Initialisiert das XGBoost Regressionsmodell mit Standardparametern

xgb.fit(train_data, train_col)  # Trainiert das Modell anhand der Trainingsdaten und der Zielwerte

predicted_col = xgb.predict(test_data)
# Nutzt das trainierte Modell, um Vorhersagen für die Testdaten zu erstellen

score = metrics.mean_absolute_error(test_col, predicted_col)
# Berechnet den mittleren absoluten Fehler (MAE) zwischen den tatsächlichen und den vorhergesagten Zielwerten

print(score)  # Gibt den berechneten Fehler aus, um die Modellgüte zu bewerten
