# -*- coding: utf-8 -*-
import os
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

import matplotlib.pyplot as plt
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import (
    BertTokenizer, BertModel, BertPreTrainedModel,
    Trainer, TrainingArguments, AutoConfig
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
import random
from transformers import TrainerCallback
import torch.nn as nn

# ========= 路径配置 =========
DATA_XLSX = r"C:\Users\ASUS\Desktop\test1.xlsx"
LOCAL_MODEL_DIR =  r"C:\Users\ASUS\Desktop\BERT\macbert"
BASE_RESULT_DIR = r"C:\Users\ASUS\Desktop\BERT\result_macbert_new"

OUTPUT_DIR = os.path.join(BASE_RESULT_DIR, "fine_tuned_results")
LOG_DIR    = os.path.join(BASE_RESULT_DIR, "fine_tuned_logs")
MODEL_DIR  = os.path.join(BASE_RESULT_DIR, "fine_tuned_model_bilstm") 
PLOT_TRAIN_LOSS = os.path.join(BASE_RESULT_DIR, "Training Loss-bert-bilstm.png")
PLOT_VAL_LOSS   = os.path.join(BASE_RESULT_DIR, "Validation Loss-bert-bilstm.png")
PLOT_CM         = os.path.join(BASE_RESULT_DIR, "Confusion Matrix-bert-bilstm.png")

os.makedirs(BASE_RESULT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ===== （可选）设为离线 & 仅PyTorch， =====
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ========= 早停回调=========
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.metrics_log = []
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and 'eval_accuracy' in metrics:
            accuracy = metrics['eval_accuracy']
            self.metrics_log.append(accuracy)
            if accuracy > self.threshold:
                print(f"验证集准确率已达到 {accuracy:.4f}，超过阈值 {self.threshold}，触发早停机制")
                control.should_training_stop = True
                control.should_save = True

# ========= 数据加载 =========
def load_data():
    # 如需，先在环境安装：pip install openpyxl
    df_train = pd.read_excel(DATA_XLSX)  # , engine="openpyxl"
    data_list = []
    for _, row in df_train.iterrows():
        data_list.append({
            'text': str(row['test']),
            'label': int(row['target'])
        })
    return data_list

data_list = load_data()
random.shuffle(data_list)

# 8:2 切分
total_size = len(data_list)
train_size = int(total_size * 0.8)
train_data = data_list[:train_size]
valid_data = data_list[train_size:]

dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "valid": Dataset.from_list(valid_data)
})

# ========= 分词器（本地BERT）=========
tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR)

def preprocess_function(examples):
    # 保持你的参数：max_length=16
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=16)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized_datasets["train"]
valid_dataset = tokenized_datasets["valid"]

# ========= 自定义 BERT+BiLSTM 分类模型 =========
class BertBiLSTMForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=6, lstm_hidden=128, lstm_layers=1, dropout_prob=0.1):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(2 * lstm_hidden, num_labels)  # 双向拼接
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state  # (B, L, H)

        lstm_out, (hn, cn) = self.lstm(sequence_output)  # hn: (num_layers*2, B, hidden)
        fw = hn[-2]  # (B, hidden)
        bw = hn[-1]  # (B, hidden)
        feat = torch.cat([fw, bw], dim=1)  # (B, 2*hidden)

        feat = self.dropout(feat)
        logits = self.classifier(feat)  # (B, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

# 从本地 BERT 权重初始化（BERT参数加载，BiLSTM随机初始化）
model = BertBiLSTMForSequenceClassification.from_pretrained(
    LOCAL_MODEL_DIR,
    num_labels=6,       # 保持你的设置
    lstm_hidden=128,    # 可之后自行调参
    lstm_layers=1,
    dropout_prob=0.1
)

# ========= 设备 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ========= 评估指标 =========
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ========= 训练参数=========
fine_tuning_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    logging_dir=LOG_DIR,
    logging_strategy="steps",
    logging_steps=25,
    save_strategy="steps",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_total_limit=3,
    report_to="none",
)

early_stopping = EarlyStoppingCallback(threshold=0.8)  # 如需早停，在 Trainer 里取消注释

# ========= Trainer =========
fine_tuner = Trainer(
    model=model,
    args=fine_tuning_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # callbacks=[early_stopping]  # 需要早停就去掉注释
)

# ========= 训练 =========
fine_tune_output = fine_tuner.train()

# ========= 日志采集与可视化 =========
fine_tune_logs = fine_tuner.state.log_history
fine_tune_train_loss = [log["loss"] for log in fine_tune_logs if "loss" in log]
fine_tune_val_loss = [log["eval_loss"] for log in fine_tune_logs if "eval_loss" in log]
fine_tune_epochs = range(1, len(fine_tune_val_loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(fine_tune_train_loss) + 1), fine_tune_train_loss, label="Train Loss")
plt.xlabel("Steps"); plt.ylabel("Loss"); plt.title("Training Loss")
plt.legend(); plt.grid(); plt.savefig(PLOT_TRAIN_LOSS, dpi=200); plt.show()

plt.figure(figsize=(10, 6))
plt.plot(fine_tune_epochs, fine_tune_val_loss, label="Validation Loss")
plt.xlabel("Steps"); plt.ylabel("Loss"); plt.title("Validation Loss")
plt.legend(); plt.grid(); plt.savefig(PLOT_VAL_LOSS, dpi=200); plt.show()

# ========= 验证集预测、指标与混淆矩阵 =========
all_predictions, all_labels = [], []
for batch in fine_tuner.get_eval_dataloader():
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs["logits"].cpu().numpy()
    predictions = np.argmax(logits, axis=-1)
    all_predictions.extend(predictions)
    all_labels.extend(batch["labels"].cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_predictions)
print("混淆矩阵：\n", conf_matrix)

acc = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average="weighted")
recall = recall_score(all_labels, all_predictions, average="weighted")
f1 = f1_score(all_labels, all_predictions, average="weighted")
print("准确率（Accuracy）:", acc)
print("精确率（Precision）:", precision)
print("召回率（Recall）:", recall)
print("F1值（F1-score）:", f1)
print("\nClassification Report:\n")
print(classification_report(all_labels, all_predictions, digits=4))

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix"); plt.colorbar()
plt.xlabel("Predicted label"); plt.ylabel("True label")
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j],
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
plt.savefig(PLOT_CM, dpi=200); plt.show()

# ========= 保存微调后的模型到你的 result1 目录 =========
fine_tuner.save_model(MODEL_DIR)      # 保存权重+配置
tokenizer.save_pretrained(MODEL_DIR)  # 保存分词器

# ======================== 推理示例 ========================
config = AutoConfig.from_pretrained(MODEL_DIR)
infer_model = BertBiLSTMForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
infer_tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

def fine_tuned_predict(text):
    inputs = infer_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=16)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    infer_model.eval()
    with torch.no_grad():
        outputs = infer_model(**inputs)
    prediction = int(outputs["logits"].argmax(dim=-1).item())
    return prediction

new_text = " 安全管理有缺陷"
print("预测结果：", fine_tuned_predict(new_text))
