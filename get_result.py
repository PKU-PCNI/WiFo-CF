import csv
import re
import numpy as np

input_file = 'WiFo_CF_base_test_Q1-Q8.csv'
output_file = 'Output_Q1_Q8_avg.csv'

# 需要求平均的指标列
metric_cols = ['nmse', 'se', 'se_max']

with open(input_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# 用于存储分组数据
# key: (model, Q编号, cr, num_bit)
# value: 对应的 nmse / se / se_max 列表
groups = {}

for row in rows:
    dataset = row['dataset'].strip()

    # 匹配 Q1.1、Q2.4、Q10.3 这种格式
    match = re.match(r'^([QD]\d+)\.\d+$', dataset)
    if not match:
        print(f"跳过无法识别的 dataset: {dataset}")
        continue

    q_group = match.group(1)  # 例如 Q1.1 -> Q1

    key = (
        row['model'].strip(),
        q_group,
        row['cr'].strip(),
        row['num_bit'].strip()
    )

    if key not in groups:
        groups[key] = {col: [] for col in metric_cols}

    for col in metric_cols:
        try:
            groups[key][col].append(float(row[col]))
        except ValueError:
            print(f"跳过非法数值: dataset={dataset}, column={col}, value={row[col]}")

# 写入结果
fieldnames = ['model', 'dataset', 'cr', 'num_bit'] + metric_cols

with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for key, values in groups.items():
        model, q_group, cr, num_bit = key

        output_row = {
            'model': model,
            'dataset': q_group,
            'cr': cr,
            'num_bit': num_bit
        }

        for col in metric_cols:
            output_row[col] = np.mean(values[col]) if len(values[col]) > 0 else ''

        writer.writerow(output_row)

print(f"处理完成，已保存为 '{output_file}'")
