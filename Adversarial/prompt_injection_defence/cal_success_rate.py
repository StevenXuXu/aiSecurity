import time
import pandas as pd
from tqdm import tqdm


########################## llm based detection (ours) ##########################
# from defences import llm_based_detection
# detect = llm_based_detection.llm_based_detection_3

########################## llm guard (protect ai) ##########################
# from defences import llm_guard
# detect = llm_guard.llm_guard_prompt_injection_detection

########################## llm guard v2 (protect ai) ##########################
from defences import llm_guard_v2
detect = llm_guard_v2.llm_guard_v2

########################## llama guard (meta) ##########################
# from defences import llama_guard
# detect = llama_guard.llama_guard_detect

########################## prompt guard (meta) ##########################
# from defences import prompt_guard
# detect = prompt_guard.prompt_guard_detection

# 提示注入攻击csv文件
df = pd.read_csv('/home/ubuntu/prompt_injection/result.csv')[:1000]
# 正常提示csv文件
# df = pd.read_csv('/home/ubuntu/prompt_injection/normal_result.csv')

# 初始化变量
correct_predictions = 0
total_predictions = len(df)
print('total: ' + str(total_predictions))

start = time.time()
# 遍历每一行，调用test函数并比较结果
for index, row in tqdm(df.iterrows()):
    predicted_label = detect(row['content'])
    actual_label = row['label']
    if predicted_label == actual_label:
        correct_predictions += 1

# 计算分类成功率
success_rate = correct_predictions / total_predictions * 100

end = time.time()
# 输出结果
print(f'分类成功率: {success_rate:.2f}%')
print('time cost: ' + str(end - start))
