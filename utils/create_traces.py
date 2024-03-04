import os
import csv

# Leggi il file Excel e crea i file di testo
csv_file_path = 'C:/Users/marco/Desktop/plcchallenge24_val_v2/val_metal.csv'  # Sostituisci con il percorso del tuo file Excel
output_folder = 'C:/Users/marco/Desktop/traces_v2'  # Sostituisci con il nome della cartella in cui desideri salvare i file di testo

# Crea la cartella di output se non esiste già
os.makedirs(output_folder, exist_ok=True)

with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) >= 4:
            sequence = row[3].replace(" ", "").strip()
            
            # Ottieni il nome del file senza l'estensione ".wav"
            file_name = row[0].split(".wav")[0]
            
            # Crea il nome del file di testo
            output_file_name = f"{file_name.replace(' ', '_')}.txt"
            output_file_path = os.path.join(output_folder, output_file_name)

            # Scrivi ogni cifra della sequenza su una riga nel file di testo senza spazi tra le righe
            with open(output_file_path, 'w') as output_file:
                output_file.write('\n'.join(sequence))

print("Operazione completata. I file di testo sono stati creati nella cartella:", output_folder)
