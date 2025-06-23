import pandas as pd
import os

# 加载你的 CSV 文件路径
csv_path = "D:\数据集分析\MAGIC-main\output_graph_M1-CVE-2015-5122_windows_h2.dot.csv"

# 加载 dot 文件名，用于生成 output 文件名（示例路径）
dot_file = r"D:\数据集分析\Flash-IDS-main\atlas_data\graph_M1-CVE-2015-5122_windows_h2.dot"
filename = os.path.splitext(os.path.basename(dot_file))[0]  # 获取不带扩展名的文件名

# 读取 CSV
df = pd.read_csv(csv_path)

# 确保列顺序正确，列名与原始要求一致
required_cols = ["actorID", "actor_type", "objectID", "object", "action", "timestamp"]
if not all(col in df.columns for col in required_cols):
    raise ValueError("CSV 中缺少必要的列，请确保包含: " + ", ".join(required_cols))

# 将所有列组合成制表符分隔的字符串
lines = df[required_cols].astype(str).agg('\t'.join, axis=1)

# 保存为 .txt 文件
output_txt_path = f"output_{filename}.txt"
with open(output_txt_path, 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line + '\n')

print(f"已保存为：{output_txt_path}")