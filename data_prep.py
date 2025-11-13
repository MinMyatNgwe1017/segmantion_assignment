import os
import pandas as pd
import matplotlib.pyplot as plt

xrays_dir = "data/xrays"
masks_dir = "data/masks"
csv_dir = "data/CSVs"
os.makedirs(csv_dir, exist_ok=True)

# sorted list of xray filenames
xray_files = sorted([f for f in os.listdir(xrays_dir) if f.endswith(".png")])

data = []
for f in xray_files:
    xray_path = os.path.join(xrays_dir, f)

    # finding a mask with the same filename
    mask_path = os.path.join(masks_dir, f)
    if os.path.exists(mask_path):
        data.append({"xrays": xray_path, "masks": mask_path})
    else:
        data.append({"xrays": xray_path, "masks": None})
        
dataset_df = pd.DataFrame(data)
dataset_csv_path = os.path.join(csv_dir, "dataset.csv")
dataset_df.to_csv(dataset_csv_path, index=False)
print(f"dataset.csv was created with {len(dataset_df)} entries")

#######
test_df = dataset_df[dataset_df["masks"].isna()]         
labeled_df = dataset_df[~dataset_df["masks"].isna()]     

# 80/20 split for training and validation
#.sample(frac=1) randomly shuffles all rows (100% of them).
#.reset_index(drop=True) removes the old indices and creates new ones (0, 1, 2, …).
labeled_df = labeled_df.sample(frac=1, random_state=42).reset_index(drop=True)
n_train = int(0.8 * len(labeled_df))
train_df = labeled_df.iloc[:n_train]
val_df = labeled_df.iloc[n_train:]

'''
.iloc[] selects rows by position:

[:n_train] → first 80% of rows (train)

[n_train:] → remaining 20% (validation)'''

train_df.to_csv(os.path.join(csv_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(csv_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(csv_dir, "test.csv"), index=False)

print(f"train.csv ({len(train_df)}) / val.csv ({len(val_df)}) / test.csv ({len(test_df)}) created")

counts = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
plt.bar(counts.keys(), counts.values(), color=['blue', 'yellow', 'red'])
plt.title("Dataset Split")
plt.ylabel("Number of Samples")
bar_path = os.path.join(csv_dir, "dataset_split_bar.png")
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.close()  
