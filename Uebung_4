import pandas as pd  # Bibliothek zum Laden und Bearbeiten von Tabellen/Daten
import tensorflow as tf  # Framework für neuronale Netze und Deep Learning
import seaborn as sns  # Bibliothek für statistische Visualisierung (hier nicht genutzt)
from sklearn.model_selection import train_test_split  # Funktion zum Aufteilen von Daten in Training und Test
import matplotlib.pyplot as plt  # Bibliothek zum Erstellen von Diagrammen (hier nicht genutzt)
from sklearn.cluster import KMeans  # K-Means Clustering Algorithmus (hier nicht genutzt)
from yellowbrick.cluster import KElbowVisualizer  # Visualisierungshilfe für Clustering (hier nicht genutzt)

# 1. Daten einlesen
file_path = "adult-2.csv"  # Dateipfad zur CSV-Datei mit Daten
df = pd.read_csv(file_path, delimiter=';')  # CSV-Datei laden, Trennzeichen ist Semikolon

# 2. Datensatz erkunden
print("\nErste 5 Zeilen des DataFrames:")
print(df.head())  # Zeigt die ersten 5 Zeilen der Tabelle an
print("\nLetzte 5 Zeilen des DataFrames:")
print(df.tail())  # Zeigt die letzten 5 Zeilen der Tabelle an

# 3. Zusammenfassung des DataFrames
print("\nAllgemeine Informationen zum DataFrame:")
print(df.info())  # Zeigt Datentypen, Anzahl der Nicht-Null-Werte pro Spalte an
print("\nStatistische Übersicht des DataFrames:")
print(df.describe())  # Zeigt statistische Kennzahlen für numerische Spalten

# 4. Bedingte Auswahl (workclass ist ungleich ?)
df = df[df['workclass'] != '?']  # Filtert alle Zeilen heraus, bei denen 'workclass' den Wert '?' hat
print("\nZeilen mit workclass ist ungleich ?:")
print(df)  # Ausgabe des gefilterten DataFrames

# 5. Nicht-numerische Werte in numerische umwandeln
print("\nNicht-numerische Werte in numerische:")
for col in df.select_dtypes(include='object').columns:  # Schleife über alle Spalten mit Textdaten (object)
    df[col] = df[col].astype('category')  # Konvertiert Spalte in kategorischen Datentyp
    df[col] = df[col].cat.codes  # Wandelt Kategorien in numerische Codes um (z.B. 0,1,2,...)
print(df)  # Ausgabe des DataFrames mit numerischen Werten

# 6. Korrelation berechnen
print("\nKorrelation:")
correlations = df[df.columns].corr(numeric_only=True)  # Berechnet paarweise Korrelationen aller numerischen Spalten
print('Alle Korrelationen:')
print('-' * 30)
correlations_abs_sum = correlations[correlations.columns].abs().sum()  # Summe der Beträge der Korrelationen pro Spalte
print(correlations_abs_sum)  # Ausgabe der Summen
print('Schwächsten Korrelationen:')
print('-' * 30)
print(correlations_abs_sum.nsmallest(5))  # Zeigt 5 Spalten mit der geringsten Gesamtkorrelation an

# 7. Spalte mit geringster Korrelation entfernen
print('#7 Spalte mit geringster Korrelation entfernen:')
df = df.drop(['fnlwgt'], axis=1)  # Entfernt die Spalte 'fnlwgt' aus dem DataFrame
print(df)  # Ausgabe des bereinigten DataFrames

# 8. KNN Klassifikation von income (Künstliches Neuronales Netz für Klassifikation)
print("\nKNN Klassifikaion von income:")

"""Daten vorbereiten."""
col_name = 'income'  # Zielvariable, die vorhergesagt werden soll

col = df[col_name]  # Zielspalte (Label) extrahieren
data = df.drop([col_name], axis=1)  # Alle anderen Spalten als Eingabedaten

"""KNN aufbauen"""

# Daten in Trainings- und Testdaten aufteilen
train_data, test_data, train_col, test_col = train_test_split(data, col, test_size=0.2, random_state=42)

# Modellaufbau: Keras Sequenzielles Modell (mehrschichtiges neuronales Netz)
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(data.shape[1],)))  # Eingabeschicht mit der Anzahl an Features
model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu))  # Verborgene Schicht mit 40 Neuronen und ReLU-Aktivierung
model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))  # Verborgene Schicht mit 80 Neuronen und ReLU-Aktivierung
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))  # Ausgabeschicht mit 2 Neuronen (für 2 Klassen), Softmax für Wahrscheinlichkeiten

# Lernprozess konfigurieren: Optimierer, Verlustfunktion und Metrik
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""Trainieren"""

# Frühes Stoppen, falls sich die Loss-Funktion nicht mehr verbessert (Geduld 3 Epochen)
cb_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Modell mit Trainingsdaten trainieren, 100 Epochen, Validierung mit Testdaten, Callback für Early Stopping
model.fit(train_data, train_col, epochs=100, validation_data=(test_data, test_col), callbacks=[cb_early])
