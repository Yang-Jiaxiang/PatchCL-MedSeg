import pandas as pd
from datetime import datetime
import os


def save_loss(total_loss, supervised_loss, contrastive_loss, consistency_loss, filename='/tf/PatchCL-MedSeg-pioneeryj/loss_record.csv'):
    # 创建一个包含损失值和当前时间的字典
    data = {'loss': [total_loss], 'supervised_loss':[supervised_loss], 'contrastive_loss':[contrastive_loss], 'consistency_loss':[consistency_loss], 'time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        # 如果文件不存在，创建一个新的 DataFrame，并保存为 CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Created new file and saved data: {filename}")
    else:
        # 如果文件已存在，加载文件，添加新数据，并保存
        df = pd.read_csv(filename)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(filename, index=False)
        print(f"Appended new data and saved to existing file: {filename}")



def check_loss_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"File {filename} has been deleted.")
    else:
        print(f"File {filename} does not exist in the directory.")

