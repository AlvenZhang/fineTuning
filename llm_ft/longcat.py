from transformers import LongformerConfig, LongformerTokenizer, LongformerForTokenClassification  # 注意：需确认模型仓库实际提供的类名

model = LongformerForTokenClassification.from_pretrained("meituan-longcat/LongCat-Flash-Thinking")
tokenizer = LongformerTokenizer.from_pretrained("meituan-longcat/LongCat-Flash-Thinking")

# 自定义 PyTorch 训练循环（需自行处理数据加载、优化器等）
print("123")