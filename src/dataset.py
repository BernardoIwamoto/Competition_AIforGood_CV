import pandas as pd
from pathlib import Path

script_dir = Path(__file__).parent
base_path = script_dir.parent / "data"

img_folders = ['GID-img-1/1', 'GID-img-2/2', 'GID-img-3/3', 'GID-img-4/4']
label_folder = 'GID-label'

data_list = []

for folder in img_folders:
    img_dir = base_path / folder
    label_dir = base_path / label_folder

    if img_dir.exists():
        for img_file in img_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() == '.jpg':
                label_name = img_file.stem + ".png"
                label_file = label_dir / label_name

                if label_file.exists():
                    data_list.append({
                        'image_path': str(img_file),
                        'label_path': str(label_file),
                        'filename': img_file.name,
                        'folder': folder
                    })

# cria o dataframe com as duplas encontradas
df = pd.DataFrame(data_list)

df.to_csv("dataset_map.csv")

df.head()  

print(f"dataframe criado: {len(df)} pares")
