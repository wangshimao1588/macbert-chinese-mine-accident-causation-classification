# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from datasets import DatasetDict, Dataset
from transformers import (
    BertTokenizer, BertModel, BertPreTrainedModel,
    Trainer, TrainingArguments, AutoConfig, TrainerCallback
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# ========= 路径配置 =========
DATA_XLSX = r"C:\Users\ASUS\Desktop\BERT\test1.xlsx"
LOCAL_MODEL_DIR = r"C:\Users\ASUS\Desktop\BERT\macbert"
BASE_RESULT_DIR = r"C:\Users\ASUS\Desktop\BERT\result_macbert_10runs"

os.makedirs(BASE_RESULT_DIR, exist_ok=True)

# ===== 离线 & 仅PyTorch =====
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ========= 随机种子设置 =========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========= 早停回调 =========
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
    df = pd.read_excel(DATA_XLSX)
    df = df[['test', 'target']].copy()
    df['test'] = df['test'].astype(str)
    df['target'] = df['target'].astype(int)
    return df

# ========= 分词器 =========
tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=16
    )

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
        self.classifier = nn.Linear(2 * lstm_hidden, num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state

        lstm_out, (hn, cn) = self.lstm(sequence_output)
        fw = hn[-2]
        bw = hn[-1]
        feat = torch.cat([fw, bw], dim=1)

        feat = self.dropout(feat)
        logits = self.classifier(feat)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

# ========= 评估指标 =========
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# ========= 单次实验函数 =========
def run_one_experiment(df, seed, run_idx):
    print("\n" + "=" * 80)
    print(f"开始第 {run_idx} 次随机划分实验，seed = {seed}")
    print("=" * 80)

    set_seed(seed)

    run_dir = os.path.join(BASE_RESULT_DIR, f"run_{run_idx}_seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    output_dir = os.path.join(run_dir, "fine_tuned_results")
    log_dir = os.path.join(run_dir, "fine_tuned_logs")
    model_dir = os.path.join(run_dir, "fine_tuned_model_bilstm")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    plot_train_loss = os.path.join(run_dir, "Training_Loss.png")
    plot_val_loss = os.path.join(run_dir, "Validation_Loss.png")
    plot_cm = os.path.join(run_dir, "Confusion_Matrix.png")
    report_txt = os.path.join(run_dir, "classification_report.txt")

    # ===== 分层随机划分 =====
    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["target"]
    )

    train_data = [{"text": row["test"], "label": row["target"]} for _, row in train_df.iterrows()]
    valid_data = [{"text": row["test"], "label": row["target"]} for _, row in valid_df.iterrows()]

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "valid": Dataset.from_list(valid_data)
    })

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = tokenized_datasets["train"]
    valid_dataset = tokenized_datasets["valid"]

    model = BertBiLSTMForSequenceClassification.from_pretrained(
        LOCAL_MODEL_DIR,
        num_labels=6,
        lstm_hidden=128,
        lstm_layers=1,
        dropout_prob=0.1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        logging_dir=log_dir,
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
        seed=seed,
        data_seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ===== 日志采集与可视化 =====
    logs = trainer.state.log_history
    train_loss = [log["loss"] for log in logs if "loss" in log]
    val_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    val_steps = range(1, len(val_loss) + 1)

    if len(train_loss) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss")
        plt.xlabel("Logging Steps")
        plt.ylabel("Loss")
        plt.title(f"Training Loss (Run {run_idx}, Seed {seed})")
        plt.legend()
        plt.grid()
        plt.savefig(plot_train_loss, dpi=200, bbox_inches="tight")
        plt.close()

    if len(val_loss) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(val_steps, val_loss, label="Validation Loss")
        plt.xlabel("Evaluation Times")
        plt.ylabel("Loss")
        plt.title(f"Validation Loss (Run {run_idx}, Seed {seed})")
        plt.legend()
        plt.grid()
        plt.savefig(plot_val_loss, dpi=200, bbox_inches="tight")
        plt.close()

    # ===== 验证集预测 =====
    all_predictions, all_labels = [], []
    model.eval()

    for batch in trainer.get_eval_dataloader():
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs["logits"].cpu().numpy()
        predictions = np.argmax(logits, axis=-1)

        all_predictions.extend(predictions)
        all_labels.extend(batch["labels"].cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_predictions)

    acc = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
    cls_report = classification_report(all_labels, all_predictions, digits=4)

    print(f"第 {run_idx} 次结果：")
    print("准确率（Accuracy）:", acc)
    print("精确率（Precision）:", precision)
    print("召回率（Recall）:", recall)
    print("F1值（F1-score）:", f1)
    print("\nClassification Report:\n")
    print(cls_report)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"Run {run_idx}, Seed {seed}\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
        f.write(f"F1-score: {f1:.6f}\n\n")
        f.write("Classification Report:\n")
        f.write(cls_report)

    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Run {run_idx}, Seed {seed})")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(
                j, i, conf_matrix[i, j],
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
            )

    plt.savefig(plot_cm, dpi=200, bbox_inches="tight")
    plt.close()

    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    return {
        "run": run_idx,
        "seed": seed,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "accuracy": acc * 100
    }

# ========= 主程序：10次随机划分 =========
if __name__ == "__main__":
    df = load_data()

    seeds = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
    all_results = []

    for idx, seed in enumerate(seeds, start=1):
        result = run_one_experiment(df, seed, idx)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    mean_row = {
        "run": "mean",
        "seed": "-",
        "precision": results_df["precision"].mean(),
        "recall": results_df["recall"].mean(),
        "f1": results_df["f1"].mean(),
        "accuracy": results_df["accuracy"].mean()
    }

    std_row = {
        "run": "std",
        "seed": "-",
        "precision": results_df["precision"].std(ddof=1),
        "recall": results_df["recall"].std(ddof=1),
        "f1": results_df["f1"].std(ddof=1),
        "accuracy": results_df["accuracy"].std(ddof=1)
    }

    summary_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True
    )

    excel_path = os.path.join(BASE_RESULT_DIR, "ten_runs_metrics.xlsx")
    csv_path = os.path.join(BASE_RESULT_DIR, "ten_runs_metrics.csv")
    txt_path = os.path.join(BASE_RESULT_DIR, "ten_runs_summary.txt")

    summary_df.to_excel(excel_path, index=False)
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("10次随机划分实验结果汇总\n")
        f.write("=" * 60 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n" + "=" * 60 + "\n")
        f.write("均值：\n")
        f.write(
            f"Precision = {mean_row['precision']:.2f}%\n"
            f"Recall    = {mean_row['recall']:.2f}%\n"
            f"F1-score  = {mean_row['f1']:.2f}%\n"
            f"Accuracy  = {mean_row['accuracy']:.2f}%\n"
        )
        f.write("标准差：\n")
        f.write(
            f"Precision = {std_row['precision']:.2f}\n"
            f"Recall    = {std_row['recall']:.2f}\n"
            f"F1-score  = {std_row['f1']:.2f}\n"
            f"Accuracy  = {std_row['accuracy']:.2f}\n"
        )

    print("\n所有10次随机划分实验已完成！")
    print(f"汇总Excel已保存到：{excel_path}")
    print(f"汇总CSV已保存到：{csv_path}")
    print(f"汇总TXT已保存到：{txt_path}")

    # ===== 最后一轮模型推理示例 =====
    last_seed = seeds[-1]
    last_run_dir = os.path.join(BASE_RESULT_DIR, f"run_{len(seeds)}_seed_{last_seed}")
    model_dir = os.path.join(last_run_dir, "fine_tuned_model_bilstm")

    infer_model = BertBiLSTMForSequenceClassification.from_pretrained(model_dir).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    infer_tokenizer = BertTokenizer.from_pretrained(model_dir)

    def fine_tuned_predict(text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = infer_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=16
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        infer_model.eval()
        with torch.no_grad():
            outputs = infer_model(**inputs)
        prediction = int(outputs["logits"].argmax(dim=-1).item())
        return prediction

    new_text = "安全管理有缺陷"
    print("推理示例文本：", new_text)
    print("预测结果：", fine_tuned_predict(new_text))