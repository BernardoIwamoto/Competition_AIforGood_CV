import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = "data/dataset_map.csv"
SEED = 42

df = pd.read_csv(CSV_PATH)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    shuffle=True
)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)

print(f"Treino: {len(train_df)}")
print(f"Validação: {len(val_df)}")