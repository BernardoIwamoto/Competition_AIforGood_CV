import pandas as pd
import os

img_folders = ['GID-img-1', 'GID-img-2', 'GID-img-3', 'GID-img-4']
label_folder = 'GID-label'

data_list = []

for folder in img_folders:
    if os.path.exists(folder):
        print(folder)
        files = os.listdir(folder)
        print(files)
        
        for filename in files:
            # tomara que o nome seja igual
            img_path = os.path.join(folder, filename)

            label_path = os.path.join(label_folder, filename)

            #se encontrar a dupla, bota a dupla na lista
            if os.path.exists(label_path):
                data_list.append({
                    'image_path': img_path,
                    'label_path': label_path,
                    'filename': filename,
                    'folder': folder
                })
            
            else:
                print(f"nao encontrou o par da imagem: {img_path}")

# cria o dataframe com as duplas encontradas
df = pd.DataFrame(data_list)

df.to_csv("dataset_map.csv")

print(f"dataframe criado: {len(df)} pares")
