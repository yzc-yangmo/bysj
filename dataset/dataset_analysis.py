import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print(os.getcwd())

def work(train_dir, val_dir):
    train_len = len(os.listdir(train_dir))
    val_len = len(os.listdir(val_dir))
    print(f"train_len: {train_len}, val_len: {val_len}")

    train_df = pd.DataFrame({"file_name": [i for i in os.listdir(train_dir)]})
    train_df["class"] = train_df["file_name"].apply(lambda x: x.split("-")[1])
    val_df = pd.DataFrame({"file_name": [i for i in os.listdir(val_dir)]})
    val_df["class"] = val_df["file_name"].apply(lambda x: x.split("-")[1])

    print(train_df["class"].value_counts().sort_index(), "\n----------------------------------\n", val_df["class"].value_counts().sort_index())

if __name__ == "__main__":
    work(input("train_dir: "), input("val_dir: "))