import os, json, random, shutil

config = json.load(open('../src/config.json'))

os.chdir(os.path.dirname(os.path.abspath(__file__)))

random.seed(7)

splited_classes_num = 50

target_classes = sorted(random.sample([str(i) for i in range(256)], k=splited_classes_num), key=lambda x: int(x))

print("splited classes: \n" + " ".join(target_classes))

class_map = {str(v):str(k) for k,v in enumerate(target_classes)}

print(class_map)

train_dir = rf"train-{splited_classes_num}"
val_dir = rf"val-{splited_classes_num}"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def work(ori_path, sub_path):
    count = 0
    for i in os.listdir(ori_path):
        if i.split("-")[1] in target_classes:
            file_info = i.split("-")
            file_info[1] = class_map[file_info[1]]
            new = "-".join(file_info)
            shutil.copy(os.path.join(ori_path, i), 
                        os.path.join(sub_path, new))
            count += 1
    # print(f"{sub_path} count: {count}")

work("./train-256", train_dir)
work("./val-256", val_dir)