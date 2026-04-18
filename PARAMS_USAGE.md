# cpsHAR Mainline Usage

这版 `cpsHAR` 已经收口成主线实验入口，默认思路是：

- `train_with_val=True`
- 四折轮转里每次 `3 个实验训练 + 1 个实验测试`
- 特征工程默认开启
- 频域分支只开放 `RFFT`
- 特殊信号组合开关保留
- 额外两轴合成信号开关保留

建议直接在 `conda auto` 环境里运行：

```bash
D:\Miniconda\envs\auto\python.exe main.py XGBoost
```

底层旧实现没有被硬删掉，后面如果你要把 `welch_psd / stft / dwt` 或其他扩展接回来，主要改入口和注册表就行。

## 1. 最小命令

```bash
python main.py XGBoost
```

## 2. 当前主线保留的模型

- `LightGBM`
- `XGBoost`
- `CatBoost`
- `TabM`
- `TabICL`
- `RGF`

说明：

- `LightGBM / XGBoost / CatBoost / RGF / TabM` 已经接入主流程
- `TabICL` 也已经接入主流程，但它依赖预训练 checkpoint
- 当前机器上没有发现本地缓存的 `TabICL` checkpoint，所以跑 `TabICL` 时建议显式提供 `--tabicl_model_path`

## 3. 主要参数

- `model`
  位置参数。模型名，例如 `LightGBM`、`XGBoost`

- `--data`
  数据集选择。可以是 `raw`、步长数字如 `200/500`，或者 `data/` 下的自定义文件名

- `--output`
  输出目录。会保存 `run_summary.json` 和混淆矩阵图

- `--train_with_val`
  默认 `True`。主线协议建议保持开启

- `--signal_combo`
  是否启用按类别定制的特殊信号组合

- `--synthetic_axes`
  是否启用两条合成轴：`Acc.norm` 和 `Gyro.norm`

- `--feature_engineering`
  是否启用当前默认手工特征工程

- `--feature_domain`
  可选：`time`、`freq`、`time_freq`

- `--spectrum_method`
  目前固定只有 `rfft`

## 4. 推荐命令

### XGBoost 主线

```bash
D:\Miniconda\envs\auto\python.exe main.py XGBoost ^
  --data 500 ^
  --feature_domain time_freq ^
  --signal_combo True ^
  --synthetic_axes True ^
  --feature_engineering True ^
  --train_with_val True ^
  --output output/xgb_mainline
```

### LightGBM 主线

```bash
D:\Miniconda\envs\auto\python.exe main.py LightGBM ^
  --data 500 ^
  --feature_domain time_freq ^
  --signal_combo True ^
  --synthetic_axes True ^
  --feature_engineering True ^
  --train_with_val True ^
  --output output/lgb_mainline
```

### CatBoost 主线

```bash
D:\Miniconda\envs\auto\python.exe main.py CatBoost ^
  --data 500 ^
  --feature_domain time_freq ^
  --signal_combo True ^
  --synthetic_axes True ^
  --feature_engineering True ^
  --train_with_val True ^
  --output output/catboost_mainline
```

### RGF 主线

```bash
D:\Miniconda\envs\auto\python.exe main.py RGF ^
  --data 500 ^
  --feature_domain time_freq ^
  --signal_combo True ^
  --synthetic_axes True ^
  --feature_engineering True ^
  --train_with_val True ^
  --output output/rgf_mainline
```

### TabM 主线

```bash
D:\Miniconda\envs\auto\python.exe main.py TabM ^
  --data 500 ^
  --feature_domain time_freq ^
  --signal_combo True ^
  --synthetic_axes True ^
  --feature_engineering True ^
  --train_with_val True ^
  --output output/tabm_mainline
```

### TabICL 主线

```bash
D:\Miniconda\envs\auto\python.exe main.py TabICL ^
  --data 500 ^
  --feature_domain time_freq ^
  --signal_combo True ^
  --synthetic_axes True ^
  --feature_engineering True ^
  --train_with_val True ^
  --tabicl_model_path path\\to\\tabicl-classifier-v2-20260212.ckpt ^
  --output output/tabicl_mainline
```

## 5. 新模型相关参数

- `--rgf_max_leaf`
- `--rgf_algorithm`
- `--rgf_reg_depth`
- `--rgf_l2`
- `--rgf_learning_rate`
- `--rgf_min_samples_leaf`

- `--tabm_max_epochs`
- `--tabm_batch_size`
- `--tabm_learning_rate`
- `--tabm_weight_decay`
- `--tabm_patience`
- `--tabm_validation_fraction`
- `--tabm_arch_type`
- `--tabm_k`
- `--tabm_d_block`
- `--tabm_n_blocks`
- `--tabm_dropout`

- `--tabicl_n_estimators`
- `--tabicl_batch_size`
- `--tabicl_kv_cache`
- `--tabicl_model_path`
- `--tabicl_allow_auto_download`
- `--tabicl_checkpoint_version`
- `--tabicl_device`
- `--tabicl_verbose`

## 6. 输出内容

当前默认只保留这几类图和汇总：

- 每折普通混淆矩阵
- 一个 overall 的纵向二分类混淆矩阵图
- `run_summary.json`

时间线图和每折二分类小图已经从主线输出里拿掉了。

## 7. 日志顺序

运行时日志分两段：

1. 先统一打印四折的预览信息
2. 再进入正式四折训练与评估

这样训练前就能先看全量折信息、比例统计和样本构成，不会把预览和训练日志搅在一起。
