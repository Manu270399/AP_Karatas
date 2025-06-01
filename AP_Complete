import pandas as pd  # Importiert die Pandas-Bibliothek zur Datenverarbeitung
import seaborn as sns  # Importiert Seaborn für Visualisierungen
from matplotlib import pyplot as plt  # Importiert Matplotlib für Diagramme
from sklearn.cluster import KMeans  # Importiert KMeans für Clustering
from yellowbrick.cluster import KElbowVisualizer  # Importiert Visualisierungstool für KMeans
from sklearn.tree import DecisionTreeClassifier  # Importiert Klassifikator für Entscheidungsbaum
from sklearn.model_selection import train_test_split  # Importiert Funktion zum Aufteilen der Daten
from sklearn.metrics import accuracy_score  # Importiert Genauigkeitsbewertung für Modelle

# 1. Daten einlesen:

# CSV-Datei einlesen (Standard)
df = pd.read_csv("csv_dateien/winequality-red.csv", delimiter=';')  # CSV-Datei mit Semikolon-Trennung laden

# --- OPTIONAL: Excel-Datei einlesen statt CSV ---
# Wenn du eine Excel-Datei verwenden möchtest (z. B. "winequality-red.xlsx"),
# kannst du stattdessen die folgende Zeile aktivieren und die obere auskommentieren:
# df = pd.read_excel("excel_dateien/winequality-red.xlsx")  # Excel-Datei laden (falls vorhanden)

# Hinweis: Stelle sicher, dass openpyxl installiert ist für .xlsx-Dateien
# → Installation mit: pip install openpyxl

# 2. Erste und letzte Zeilen anzeigen
print("\nErste 5 Zeilen des DataFrames:")
print(df.head())  # Gibt die ersten 5 Zeilen aus

print("\nLetzte 5 Zeilen des DataFrames:")
print(df.tail())  # Gibt die letzten 5 Zeilen aus

# 3. Zusammenfassung des Datensatzes
print("\nAllgemeine Informationen zum DataFrame:")
print(df.info())  # Überblick über Spalten, Datentypen und Null-Werte

print("\nStatistische Übersicht des DataFrames:")
print(df.describe())  # Statistische Kennzahlen für numerische Spalten

# 4. Nur ausgewählte Spalten anzeigen
selected_columns = df[['alcohol', 'pH']]  # Wählt die Spalten 'alcohol' und 'pH'
print("\nErste 10 Zeilen der ausgewählten Spalten:")
print(selected_columns.head(10))  # Gibt die ersten 10 Zeilen dieser Spalten aus

# 5. Daten filtern (quality == 8)
filtered_df_8 = df[df['quality'] == 8]  # Filtert alle Zeilen mit Qualität genau 8
print("\nZeilen mit Qualität genau 8:")
print(filtered_df_8)

# 6. Filtern mit mehreren Bedingungen (alcohol > 12.5 & quality >= 7)
filtered_df = df[(df['alcohol'] > 12.5) & (df['quality'] >= 7)]  # Filter mit zwei Bedingungen
print("\nZeilen mit Alkohol > 12.5 und Qualität >= 7:")
print(filtered_df)

# 7. Neue Spalte berechnen
df['density_alcohol_ratio'] = df['density'] / df['alcohol']  # Neue Spalte: Dichte durch Alkohol
print("\nNeue Spalte 'density_alcohol_ratio':")
print(df[['density_alcohol_ratio']].head())  # Zeigt neue Spalte

# 8. Werte in 'quality' umwandeln zu Text
def quality_label(q):  # Funktion zur Umwandlung in Text-Kategorien
    if q == 3:
        return "sehr schlecht"
    elif q == 4:
        return "schlecht"
    elif q == 5:
        return "okay"
    elif q == 6:
        return "gut"
    else:
        return "sehr gut"
df['quality_label'] = df['quality'].apply(quality_label)  # Neue Spalte mit Textlabels
print("\nQualitätswerte umgewandelt in Labels:")
print(df[['quality', 'quality_label']].head())

# 9. Zeilen löschen (pH < 3.0)
df = df[df['pH'] >= 3.0]  # Entfernt alle Zeilen mit zu niedrigem pH-Wert
print("\nDaten nach Entfernen von pH < 3.0:")
print(df.head())

# 10. Spaltenüberschriften anzeigen
print("\nSpaltenüberschriften:")
print(df.columns)

# 11. Spaltenüberschriften ins Deutsche übersetzen
deutsch_columns = {
    'fixed acidity': 'fester Säuregehalt',
    'volatile acidity': 'flüchtiger Säuregehalt',
    'citric acid': 'Zitronensäure',
    'residual sugar': 'Restzucker',
    'chlorides': 'Chloride',
    'free sulfur dioxide': 'freies Schwefeldioxid',
    'total sulfur dioxide': 'Gesamtschwefeldioxid',
    'density': 'Dichte',
    'pH': 'pH-Wert',
    'sulphates': 'Sulfate',
    'alcohol': 'Alkohol',
    'quality': 'Qualität'
}
df.rename(columns=deutsch_columns, inplace=True)  # Benennt die Spalten um
print("\nSpalten nach Umbenennung:")
print(df.head())

# 12. Gruppieren nach Qualitätslabel
grouped = df.groupby('quality_label').mean(numeric_only=True)  # Gruppiert nach Kategorie und berechnet Durchschnitt
print("\nDurchschnittswerte je Qualitätskategorie:")
print(grouped)

# 13. Balkendiagramm zur Gruppierung
print("\nErstelle Balkendiagramm nach Qualität:")
grouped[['Alkohol']].plot(kind='bar', legend=False)  # Erstellt ein Balkendiagramm für Alkohol-Durchschnitt je Kategorie
plt.ylabel("Durchschnittlicher Alkoholgehalt")  # Achsenbeschriftung Y
plt.title("Alkoholgehalt nach Qualitätslabel")  # Titel des Diagramms
plt.xticks(rotation=45)  # Dreht die X-Achsenbeschriftung leicht zur besseren Lesbarkeit
plt.tight_layout()  # Sorgt dafür, dass Achsen und Titel nicht abgeschnitten werden
plt.show()  # Zeigt das Diagramm

# 14. Vorhersage mit Entscheidungsbaum
print("\nEntscheidungsbaum zur Vorhersage von Qualitätslabel:")

# Zielspalte: quality_label (Text-Kategorie)
y = df['quality_label']  # Zielvariable (zu vorhersagen)
X = df.drop(['quality_label', 'Qualität'], axis=1)  # Eingabedaten (alle anderen Spalten)

# Aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% Training, 20% Test

# Modell erstellen und trainieren
model = DecisionTreeClassifier(random_state=0)  # Erstellt ein Entscheidungsbaum-Modell
model.fit(X_train, y_train)  # Training mit den Daten

# Vorhersagen auf Testdaten
y_pred = model.predict(X_test)  # Macht Vorhersagen auf Testdaten

# Genauigkeit berechnen
accuracy = accuracy_score(y_test, y_pred)  # Misst, wie viele Vorhersagen korrekt waren
print(f"Genauigkeit des Entscheidungsbaums: {accuracy:.2f}")  # Gibt Genauigkeit als Zahl zwischen 0 und 1 aus

# 15. Seaborn Scatterplot (nicht prüfungsrelevant, optional)
print("\nScatterplot: Alkohol vs. Qualität")
sns.scatterplot(x=df['Alkohol'], y=df['Qualität'])  # Streudiagramm erstellen
plt.xlabel("Alkoholgehalt")
plt.ylabel("Qualität")
plt.title("Alkohol vs. Qualität")
plt.show()

# 16. KMeans-Clustering (nicht prüfungsrelevant, optional)
print("\nKMeans-Clustering:")

# Zielspalten entfernen
data_unknown = df.drop(['Qualität', 'quality_label'], axis=1)  # Entfernt Zielspalten

# Zeigt Datentypen der Spalten
print(data_unknown.dtypes)

# Visualisierung für optimale Clusterzahl
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 9))
visualizer.fit(data_unknown)
visualizer.show()

# Clustering mit 4 Clustern
kmeans = KMeans(n_clusters=4)
pred = kmeans.fit_predict(data_unknown)

# Cluster als neue Spalte hinzufügen
data_new = pd.concat([df, pd.DataFrame(pred, columns=['label'])], axis=1)
print("\nDaten mit Clusterzuordnung:")
print(data_new)

# Speichern in neue CSV
data_new.to_csv("csv_dateien/data_new.csv")  # Speichert Ergebnis in Datei
