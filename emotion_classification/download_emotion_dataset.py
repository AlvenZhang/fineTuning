from datasets import load_dataset

# 指定数据集下载位置
CACHE_DIR = "./emotion_dataset_cache"

# 下载并加载 Emotion 数据集
print(f"开始下载 Emotion 数据集到 {CACHE_DIR}...")
dataset = load_dataset("emotion", cache_dir=CACHE_DIR)
print("数据集下载完成！")

# 查看数据集结构
print("\n数据集结构：")
print(dataset)

# 查看数据样本
print("\n训练集前 3 个样本：")
for i in range(3):
    sample = dataset["train"][i]
    print(f"文本: {sample['text']}")
    print(f"标签: {sample['label']}")
    print()

# 查看标签映射
print("\n标签映射：")
print({0: "anger", 1: "fear", 2: "joy", 3: "love", 4: "sadness", 5: "surprise"})

print(f"\n数据集已成功下载并保存到指定目录：{CACHE_DIR}")
print("你可以在后续的微调脚本中通过设置相同的 cache_dir 参数来使用它。")
print("例如：dataset = load_dataset('emotion', cache_dir='./emotion_dataset_cache')")
