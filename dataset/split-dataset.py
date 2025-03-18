import os, json, random, shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))

splited_classes_num = 20

target_classes = random.sample([k for k, _ in json.load(open('../src/mapping.json')).items()], k=splited_classes_num)

train_dir = rf"sub-train-{splited_classes_num}"
val_dir = rf"sub-val-{splited_classes_num}"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

count = 0
for i in os.listdir("./train"):
    if i.split("-")[1] in target_classes:
        shutil.copy(os.path.join("./train", i), 
                    os.path.join(train_dir, i))
        count += 1
print(f"train count: {count}")

count = 0
for i in os.listdir("./val"):
    if i.split("-")[1] in target_classes:
        shutil.copy(os.path.join("./val", i), 
                    os.path.join(val_dir, i))
        count += 1
print(f"val count: {count}")


print("splited classes: \n" + " ".join(target_classes))

