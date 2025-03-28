import os, json, random, shutil

config = json.load(open('../src/config.json'))

os.chdir(os.path.dirname(os.path.abspath(__file__)))


splited_classes_num = 20

target_classes = random.sample([str(i) for i in range(config["train"]["num_classes"])], k=splited_classes_num)

train_dir = rf"train-{splited_classes_num}"
val_dir = rf"val-{splited_classes_num}"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def work(ori_path, sub_path):
    count = 0
    for i in os.listdir(ori_path):
        if i.split("-")[1] in target_classes:
            shutil.copy(os.path.join(ori_path, i), 
                        os.path.join(sub_path, i))
            count += 1
    print(f"{sub_path} count: {count}")

work("./train-256", train_dir)
work("./val-256", val_dir)

print("splited classes: \n" + " ".join(target_classes))


