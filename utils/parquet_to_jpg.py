import pandas as pd
import fastparquet as fp
import os, json
from PIL import Image
import io
import threading


mapping = {v: k for k, v in json.load(open("./src/mapping.json", "r")).items()}


def work(parquet_path, output_dir):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df = fp.ParquetFile(parquet_path).to_pandas()
        # df.columns = Index(['label', 'image.bytes', 'image.path'], dtype='object')
        for index, row in df.iterrows():
            # 输出路径
            output_path = os.path.join(output_dir, f"img-{mapping[row.iloc[0]]}-{row.iloc[2].split('.')[0]}.jpg")
            
            # 保存为图像
            with open(output_path, 'wb') as f:
                f.write(row.iloc[1])
            
            # 打印进度信息
            print(f"[{(index+1)/len(df)*100:.2f}%  {index+1} / {len(df)}]  {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        

if __name__ == '__main__':
    train_parquet_path, train_output_path = r"./dataset/parquets/train.parquet", r"./dataset/train"
    val_parquet_path, val_output_path =     r"./dataset/parquets/validation.parquet", r"./dataset/val"
    work(train_parquet_path, train_output_path)
    work(val_parquet_path, val_output_path)
