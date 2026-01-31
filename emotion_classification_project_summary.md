# Qwen3-0.6B 情感分类微调项目总结

## 项目概述

本项目旨在使用 Qwen3-0.6B 模型进行情感分类任务的微调，将文本分类为六种情感类别：愤怒（anger）、恐惧（fear）、快乐（joy）、爱（love）、悲伤（sadness）和惊讶（surprise）。

## 项目结构

```
emotion_classification/
├── config/
│   └── config.py          # 配置管理
├── data/
│   └── dataset.py         # 数据集处理
├── model/
│   └── model.py           # 模型加载和配置
├── train/
│   └── trainer.py         # 训练配置和执行
├── evaluate/
│   └── evaluator.py       # 模型评估
├── utils/
│   └── utils.py           # 工具函数
├── download_emotion_dataset.py  # 数据集下载脚本
├── fine_tune_qwen_emotion.py    # 微调主脚本
├── fine_tune_qwen_emotion_plan.md  # 项目计划
├── inference_example.py    # 推理示例
└── main.py                # 主执行脚本
```

## 已实现功能

### 1. 配置管理
- 命令行参数解析
- 配置类管理
- 目录结构自动创建

### 2. 数据集处理
- 从本地缓存加载 Emotion 数据集
- 数据分词和预处理
- 标签映射管理
- 数据集统计信息日志

### 3. 模型加载与配置
- 加载 Qwen3-0.6B 预训练模型
- 配置为序列分类任务
- PEFT (LoRA) 高效微调配置

### 4. 训练配置
- 学习率、批次大小等超参数设置
- 训练参数配置
- 评估指标定义

### 5. 模型评估
- 测试集性能评估
- 详细指标计算（准确率、精确率、召回率、F1值）
- 分类报告生成

### 6. 模型保存与部署
- 模型和分词器保存
- 推理示例脚本生成
- 模型卡片创建

## 技术特点

### 1. 高效微调
- 使用 PEFT (Parameter-Efficient Fine-Tuning) 技术
- 配置 LoRA (Low-Rank Adaptation) 参数：r=8, alpha=16
- 减少可训练参数数量，提高训练效率

### 2. 完整的评估体系
- 多维度评估指标
- 详细的分类报告
- 模型性能可视化

### 3. 模块化设计
- 清晰的代码结构
- 职责分离明确
- 易于维护和扩展

### 4. 完整的部署支持
- 推理示例脚本
- 模型卡片文档
- 详细的使用说明

## 数据集信息

- **数据集名称**：Emotion 数据集
- **数据规模**：
  - 训练集：16,000 个样本
  - 验证集：2,000 个样本
  - 测试集：2,000 个样本
- **情感类别**：6 种（anger, fear, joy, love, sadness, surprise）
- **语言**：英语

## 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 2e-4 | 模型训练学习率 |
| 批次大小 | 8 | 训练和评估批次大小 |
| 训练轮数 | 3 | 模型训练轮数 |
| 预热步数 | 500 | 学习率预热步数 |
| 权重衰减 | 0.01 | 正则化参数 |
| 随机种子 | 42 | 结果可重现性 |
| PEFT 配置 | LoRA (r=8, alpha=16) | 高效微调配置 |

## 当前项目状态

### ✅ 已完成
- [x] 项目结构搭建
- [x] 配置管理模块
- [x] 数据集处理模块
- [x] 模型加载与配置模块
- [x] 训练配置模块
- [x] 模型评估模块
- [x] 模型保存与部署模块
- [x] 推理示例脚本
- [x] 模型卡片模板

### ⏳ 待执行
- [ ] 取消注释训练代码并执行训练
- [ ] 评估模型性能
- [ ] 保存最终模型
- [ ] 生成完整的模型卡片和推理示例

## 如何运行

1. **准备环境**：确保安装了所需的依赖包
2. **准备模型**：下载 Qwen3-0.6B 预训练模型到指定路径
3. **准备数据集**：运行 `download_emotion_dataset.py` 下载并缓存数据集
4. **执行训练**：
   - 取消注释 `main.py` 或 `fine_tune_qwen_emotion.py` 中的训练代码部分
   - 运行 `python main.py` 或 `python fine_tune_qwen_emotion.py`

## 预期输出

训练完成后，项目将生成以下文件：

- `fine_tuned_qwen_emotion/`：保存微调后的模型
- `fine_tuned_qwen_emotion/tokenizer/`：保存分词器
- `fine_tuned_qwen_emotion/evaluation_results.txt`：评估结果
- `fine_tuned_qwen_emotion/model_card.md`：模型卡片
- `fine_tuned_qwen_emotion/inference_example.py`：推理示例脚本

## 模型使用示例

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def classify_emotion(text):
    """使用微调后的模型对文本进行情感分类"""
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained("fine_tuned_qwen_emotion")
    model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_qwen_emotion")
    
    # 分词输入文本
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # 预测情感
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    
    # 映射到情感标签
    emotion_labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]
    return emotion_labels[predictions.item()]

# 示例使用
test_sentences = [
    "I'm so happy today!",
    "I feel scared of the dark.",
    "I'm angry about the situation.",
    "I love spending time with my family.",
    "I'm sad that it's over.",
    "I'm surprised by the news!"
]

for sentence in test_sentences:
    emotion = classify_emotion(sentence)
    print(f"Text: {sentence}")
    print(f"Emotion: {emotion}")
    print("-")
```

## 项目优势

1. **高效微调**：使用 PEFT 技术减少计算资源需求
2. **模块化设计**：代码结构清晰，易于维护和扩展
3. **完整的评估体系**：多维度评估模型性能
4. **详细的文档**：包含模型卡片和使用说明
5. **易于部署**：提供推理示例脚本

## 潜在改进方向

1. **超参数优化**：尝试不同的学习率、批次大小等超参数
2. **模型集成**：考虑使用模型集成提高性能
3. **多语言支持**：扩展到其他语言的情感分类
4. **领域适应**：针对特定领域的情感分类进行优化
5. **实时推理**：优化模型以支持实时推理场景

## 结论

本项目已完成所有核心组件的实现，采用了模块化设计和高效微调技术，为 Qwen3-0.6B 模型的情感分类任务做好了充分准备。只需取消注释训练代码并执行训练，即可获得一个性能良好的情感分类模型。

项目结构清晰，代码质量高，为后续的模型部署和应用提供了坚实的基础。