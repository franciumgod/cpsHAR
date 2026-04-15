# 参数使用说明（简化版）

这个项目参数很多，但日常训练你可以只记下面几个核心参数。

## 1. 最小可用命令

```bash
python main.py LightGBM
```

默认会使用：
- `data=raw`
- `feature_engineering=False`
- `feature_domain=time`
- `signal_combo=False`
- `train_with_val=False`

---

## 2. 建议优先使用的核心参数（只记这 7 个）

1. `--data`
- 数据集选择：`raw`、`100`、`200`、`400`、`500`、自定义 pkl 文件名。

2. `--feature_engineering`
- 是否开启时域手工特征增强。`True/False`

3. `--feature_domain`
- 特征域选择：
  - `time`：只用时域特征（默认）
  - `freq`：只用频域特征
  - `time_freq`：时域+频域拼接

4. `--spectrum_method`
- 当 `feature_domain` 包含 `freq` 时生效：
  - `rfft`（默认）
  - `welch_psd`
  - `stft`
  - `dwt`

5. `--use_tsfresh`
- 是否拼接 tsfresh 特征。`True/False`

6. `--signal_combo`
- 是否启用按标签的传感器组合。`True/False`

7. `--train_with_val`
- 是否将 train+val 合并用于最终训练。`True/False`

8. `--output`
- 结果输出目录（图、run_summary.json）。

---

## 3. 常用命令模板

## A. 轻量基线（推荐先跑）

```bash
python main.py --data 500 --feature_engineering False --feature_domain time --signal_combo False --train_with_val False --output output/baseline_500 LightGBM
```

## B. 时域+频域

```bash
python main.py --data 500 --feature_engineering True --feature_domain time_freq --spectrum_method welch_psd --signal_combo True --train_with_val True --output output/time_freq_500 LightGBM
```

## C. 频域 + tsfresh

```bash
python main.py --data 500 --feature_domain freq --spectrum_method stft --use_tsfresh True --signal_combo True --train_with_val True --output output/freq_tsfresh_500 XGBoost
```

---

## 4. 与增强相关（按需加）

- `--sample_augment`：训练样本增强数量（`False/0` 关闭，`True`=1，或整数 N）
- `--augment_method`：`jitter,scaling,rotation,mixup,cutmix,smote,basic`
- `--augment_target`：默认 `multilabel`
- `--tta`：测试时增强次数
- `--tta_method`：TTA 方法，和 `augment_method` 同集合

---

## 5. 训练日志里的关键参数打印

每次运行会自动打印以下关键配置：
- `data`
- `train_with_val`
- `signal_combo`
- `feature_engineering`
- `feature_domain`

方便你确认当前实验开关状态，避免命令太长看漏。
