import pandas as pd
import csv

input_file = "data/headlines.csv"
output_file = "data/headlines_fixed.csv"

clean_rows = []

with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.reader(f)
    for row in reader:
        # Élimine les lignes vides ou avec entête répétée
        if not row or row[0].strip() == "Date":
            continue
        # Supprime les points-virgules à la fin si présents
        row = [cell for cell in row if cell.strip() != ""]

        if len(row) >= 3:
            date = row[0].strip()
            label = row[1].strip()
            headlines = " ".join(cell.strip() for cell in row[2:])  # concatène les titres
            clean_rows.append([date, label, headlines])

# Enregistrement du fichier propre
df_clean = pd.DataFrame(clean_rows, columns=["Date", "label", "headlines"])
df_clean = df_clean.iloc[1:]
df_clean.to_csv(output_file, index=False)
print(f"✅ Fichier nettoyé : {output_file}")
