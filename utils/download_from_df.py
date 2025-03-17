from datasets import load_dataset
import pyarrow.parquet as pq
import pyarrow as pa
import os

# 加载数据集
ds = load_dataset("ethz/food101")

# 定义保存路径
current_dir = os.getcwd()
output_dir = os.path.abspath(os.path.join(current_dir, "../dataset/parquets"))
output_dir = "./dataset/parquets"

os.makedirs(output_dir, exist_ok=True)

# 遍历数据集的拆分并转换为 Parquet 格式
for split, dataset in ds.items():
    table = pa.Table.from_pandas(dataset.to_pandas())  # 转换为 PyArrow Table
    file_path = f"{output_dir}/{split}.parquet"
    pq.write_table(table, file_path)
    print(f"Saved {split} split to {file_path}")

print("Dataset conversion completed!")