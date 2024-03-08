import os
import csv

csv_file_path = 'C:/Users/marco/Desktop/plcchallenge24_val_v2/val_meta.csv' 
output_folder = 'C:/Users/marco/Desktop/traces_v2'  

os.makedirs(output_folder, exist_ok=True)

with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) >= 4:
            sequence = row[3].replace(" ", "").strip()
            
            file_name = row[0].split(".wav")[0]
            
            output_file_name = f"{file_name.replace(' ', '_')}.txt"
            output_file_path = os.path.join(output_folder, output_file_name)

            with open(output_file_path, 'w') as output_file:
                output_file.write('\n'.join(sequence))

print("Saved in:", output_folder)
