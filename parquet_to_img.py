import numpy as np
import pandas as pd
import os
from PIL import Image
import io
import threading

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def to_image(info_dict, output_path):
    try:
        img = Image.open(io.BytesIO(info_dict['bytes']))
        
        if info_dict["path"].split('.')[-1] == "jpg":
            img.save(output_path, "JPEG")
            return
        img.save(output_path, info_dict["path"].split('.')[-1].upper())
    
    except Exception as e:
        print(f"Error: {e}")
        return


def process_parquet(parquet_path, thread_id):
    try:
        df = pd.read_parquet(parquet_path)
        for index, row in enumerate(df.iterrows()):
            # 输出路径
            output_path = os.path.join(OUTPUT_PATH, f"img-{row[1].iloc[1]}-{row[1].iloc[0]["path"]}")
            # 保存为图像
            to_image(row[1].iloc[0], output_path)
            print(f"thread_id:{thread_id}  [{(index+1)/len(df)*100:.2f}%  [{index+1} / {len(df)}]]  {output_path}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    threads = []
    for thread_id, par in enumerate(os.listdir(PARQUET_PATH)):
        parquqet_path = os.path.join(PARQUET_PATH, par)
        t = threading.Thread(target=process_parquet, args=(parquqet_path, thread_id))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()

    print("All processing completed!")
        

if __name__ == '__main__':
    PARQUET_PATH = r"..\parquets"
    OUTPUT_PATH = r"..\imgs"
    main()
