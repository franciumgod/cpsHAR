# CPSHAR 项目阶段性完整报告

报告日期：2026-04-18

## 1. 项目目标

这个项目的核心任务是做多标签工业活动识别（multi-label HAR）。输入是叉车/作业过程中的多传感器时间序列，输出不是单一动作类别，而是 6 个超类标签上的多标签判定，允许一个时间窗口内同时出现多个活动。

从当前仓库结构看，项目的主线已经从“直接端到端深度模型”逐步收敛到“窗口化时序 + 手工特征/频域特征 + One-vs-Rest 梯度提升树”的方案，并围绕以下问题做了系统实验：

1. 数据窗口应该如何构建与抽样。
2. 标签应该如何从细类映射到稳定的超类。
3. 是否需要特征工程、频域特征、类定制信号组合。
4. 是否需要对正样本阈值、负样本构成、增强策略做针对性优化。

## 2. 数据是什么

### 2.1 原始数据形态

原始数据文件是 `data/cps_data_multi_label.pkl`，体量约 653 MB。根据数据分析结果，整套原始逐点数据共有 4 个实验段（experiment 1~4），总计 4,186,812 行采样点。

每个采样点包含：

1. 7 个原始传感器通道：
   `Acc.x, Acc.y, Acc.z, Gyro.x, Gyro.y, Gyro.z, Baro.x`
2. 后续派生的 2 个模长通道：
   `Acc.norm, Gyro.norm`
3. 10 个原始细标签：
   `Driving(straight), Driving(curve), Lifting(raising), Lifting(lowering), Standing, Docking, Forks(entering or leaving front), Forks(entering or leaving side), Wrapping, Wrapping(preparation)`

代码依据：

1. 传感器与标签定义见 `utils/config.py`
2. 模长通道构造见 `data_handler.py` 中 `_add_new_signal`

### 2.2 标签体系

项目并没有直接使用 10 个细标签做最终建模，而是将它们映射为 6 个超类：

1. `Driving(curve)`
2. `Driving(straight)`
3. `Lifting(lowering)`
4. `Lifting(raising)`
5. `Stationary processes`
6. `Turntable wrapping`

其中：

1. `Standing / Docking / Forks(front) / Forks(side) / Wrapping(preparation)` 被合并为 `Stationary processes`
2. `Wrapping` 被映射为 `Turntable wrapping`

这样做的目的很明确：减少细粒度静态动作之间的碎片化，提高标签稳定性，同时把真正容易混淆、业务上更关键的 driving / lifting 类保留下来。

### 2.3 数据规模与分布

原始逐点数据按 4 个实验统计如下：

1. EXP1：1,108,546 行
2. EXP2：941,667 行
3. EXP3：947,353 行
4. EXP4：1,189,246 行

总体标签分布很不平衡：

1. `Stationary processes` 占比最高，原始逐点正样本比例约 43.52%
2. `Driving(curve)` 约 16.54%
3. `Driving(straight)` 约 15.57%
4. `Turntable wrapping` 约 11.16%
5. `Lifting(lowering)` 约 4.29%
6. `Lifting(raising)` 约 3.13%

整体样本组成也说明问题难点明显：

1. 无标签点约 9.55%
2. 单标签点约 86.69%
3. 多标签点约 3.75%

虽然多标签占比不算特别高，但它们高度集中在 driving 和 lifting 的边界组合上，是当前模型最难处理的一类窗口。

## 3. 项目对数据做了哪些处理

### 3.1 清洗与信号补充

数据进入训练前会先做三步基础处理：

1. 删除无关列：`Error, Synchronization, None, transportation, container`
2. 增加 `Acc.norm` 和 `Gyro.norm`
3. 应用超类映射，将 10 个细标签收缩为 6 个建模标签

这是所有后续窗口构造和建模的统一前处理。

### 3.2 窗口化

项目采用 2 秒窗口。由于原始频率是 2000 Hz，因此原始窗口长度是：

1. `window_size = 2 * 2000 = 4000` 个采样点

项目已经准备了多种窗口步长的数据：

1. `step=1`：存成 sample manifest
2. `step=100`
3. `step=200`
4. `step=400`
5. `step=500`

其中当前主线实验几乎都围绕 `step=500` 展开，对应数据文件为：

1. `data/cps_windows_2s_2000hz_step_500.pkl`

这个文件一共包含 8,344 个窗口，分布到 4 个实验中：

1. EXP1：2,210
2. EXP2：1,876
3. EXP3：1,887
4. EXP4：2,371

一个重要结论是：`step=500` 窗口后的总体标签比例与原始逐点数据几乎一致，说明当前窗口采样策略没有明显扭曲总体分布，适合作为主训练集。

### 3.3 窗口标签的定义

窗口标签不是窗口内多数投票，而是采用“窗口末端标签作为硬标签”的方式：

1. `y = label_values[window_end - 1]`

同时，项目额外计算了一个非常关键的辅助量：

1. `label_ratio_pre_ds`

它表示某个标签在整个原始窗口 4000 个点中所占的比例。这个比例不是最终预测目标，但被用于：

1. 分析窗口边界的模糊程度
2. 正样本阈值筛选
3. One-vs-Rest 负样本构成控制
4. 样本统计分析

这一步是当前项目设计里很有价值的地方，因为它把“硬标签”与“窗口纯度”分开保存了。

### 3.4 下采样与归一化

当前主线使用 `preprocess_order = subsample_first`。也就是说：

1. 先从原始 4000 点窗口取样
2. 再对单个窗口做时域下采样

默认下采样方式是 `sliding_window`：

1. 窗长 40
2. 步长 20

这相当于把 2000 Hz 的原始窗口压缩成约 199 个时间步的低频序列，用移动平均保留局部趋势。

完成窗口构造后，项目使用 `MinMaxScaler(-1, 1)` 按通道做缩放，并且只用训练集拟合 scaler，然后应用到验证集与测试集。

### 3.5 数据切分方式

项目使用 4 折轮转式留一实验：

1. Fold1：test=EXP1, val=EXP2, train=EXP3+EXP4
2. Fold2：test=EXP2, val=EXP3, train=EXP1+EXP4
3. Fold3：test=EXP3, val=EXP4, train=EXP1+EXP2
4. Fold4：test=EXP4, val=EXP1, train=EXP2+EXP3

这是当前项目最重要的评估协议。它比随机切分更合理，因为不同 experiment 之间存在明显分布差异。

## 4. 提取了哪些特征

项目的特征工程分为 4 层。

### 4.1 时域基础统计特征

默认时域特征包括：

1. 均值 `mean`
2. 标准差 `std`
3. 最大值 `max`
4. 最小值 `min`

### 4.2 增强时域手工特征

开启 `feature_engineering=True` 后，会继续增加：

1. RMS
2. RMSE（去均值后）
3. 一阶差分均值
4. 多个时滞点特征：lag 1、3、5、10

因此项目当前并不是只做简单统计量，而是在尝试补充短期动态变化和末端时序状态。

### 4.3 频域特征

如果 `feature_domain` 包含 `freq`，项目会先构建频谱，再提取频域统计：

可选频谱方法包括：

1. `rfft`
2. `welch_psd`
3. `stft`
4. `dwt`

在频谱上进一步提取：

1. 频域均值
2. 频域标准差
3. 频域最大值
4. 频域最小值
5. 频域能量
6. 频谱质心
7. 频谱扩散度

### 4.4 tsfresh 特征

项目已经支持用 `tsfresh` 做额外自动特征抽取，并把结果拼接到特征向量后面。但从现有仓库结果看，这条线还没有形成最终可验证结论：

1. `R34_P4_final_best_plus_tsfresh` 目录只有命令和配置，没有完整 `run_summary.json`
2. 说明这部分实验在当前仓库快照里尚未完成或未保留结果

## 5. 用了哪些方法

### 5.1 当前主线模型

当前主线可运行模型主要是两个：

1. `LightGBM`
2. `XGBoost`

它们都采用多标签 One-vs-Rest 训练方式：每个标签独立训练一个二分类器。

### 5.2 类定制信号组合

项目不是所有类别都用全部传感器，而是为不同标签设计了定制通道子集。例如：

1. `Driving(curve)`：`Acc.x, Acc.z, Gyro.norm`
2. `Lifting(lowering)`：`Acc.z, Gyro.x, Baro.x`
3. `Turntable wrapping`：`Acc.z, Gyro.z, Gyro.norm`

这说明项目已经不再把 9 个通道视作完全同质，而是在利用动作先验做“按类选信号”。

### 5.3 正样本阈值筛选

对某些标签，项目允许使用 `label_ratio_pre_ds` 设定正样本阈值。例如：

1. `Lifting(raising):0.1`

它的含义不是改变测试判定阈值，而是在训练某个 OvR 分类器时，只保留窗口内该标签占比超过阈值的正样本。也就是说，项目在主动剔除“末端是正样本，但整个窗口非常不纯”的边界窗口。

### 5.4 负样本构成控制

项目还实现了针对某个 OvR 分类器的负样本重组。例如：

1. 在训练 `Lifting(lowering)` 时，指定负样本中 `Driving(curve)` 的比例为 10%

这一机制的目的，是把分类边界集中到真正容易混淆的类别上，而不是让大量“太容易的负样本”主导训练。

### 5.5 数据增强与 TTA

项目支持多种增强方法：

1. jitter
2. scaling
3. rotation
4. mixup
5. cutmix
6. smote
7. basic 组合增强

测试时也支持 TTA。

不过需要注意：当前仓库中增强脚本是存在的，但增强实验输出目录并不完整保留，因此增强结论不能与主线 LightGBM / XGBoost 结果同等强度地视为“本地已完全复现”。

### 5.6 深度模型探索

仓库 `result/ds_result/` 中还保留了几组深度模型结果：

1. Swin Transformer, time-freq image, focal loss
2. Swin Transformer, time-freq image
3. InceptionTime, time-freq image, BCE

这些结果说明项目曾尝试把 7 路信号转成 time-frequency 图像来做视觉式建模，但当前表现明显落后于树模型主线。

## 6. 得到了什么样的结果

## 6.1 数据分析层面的结论

数据分析已经得出几个稳定结论：

1. `Stationary processes` 是最大类，`Lifting(raising)` 和 `Lifting(lowering)` 是最难的小类。
2. 多标签样本主要集中在 driving 与 lifting 的组合边界上。
3. `step=500` 窗口化后，整体类别比例几乎保持不变，因此这套窗口化策略是合理的。
4. 实验间分布并不完全一致，特别是 EXP2 的多标签比例最高，达到约 6.45%，这会对折间稳定性产生影响。

## 6.2 LightGBM 主线实验结果

当前仓库里最完整、最系统的一条结果链来自 `output/exp_20260415_lgbm_rt_v2/leaderboard.csv`。从该排行榜看：

1. 基线方案（不做特征工程、不做信号组合）`avg_mcc = 0.6856`
2. 只加特征工程后提升到 `0.7536`
3. 只加信号组合提升到 `0.7141`
4. 特征工程 + 信号组合 + train_with_val 后达到 `0.7955`

进一步搜索后，当前这条 LightGBM 主线里最优配置是：

1. `feature_domain = time_freq`
2. `spectrum_method = welch_psd`
3. 其结果为：
   `avg_mcc = 0.8051`
   `avg_macro_f1 = 0.8243`
   `avg_macro_pr_auc = 0.8892`
   `avg_macro_brier = 0.0345`

这说明当前项目最有效的增益主要来自三件事：

1. 手工特征工程
2. 类定制信号组合
3. 时域 + 频域联合特征

另外，几组关键结论也比较明确：

1. 只用频域特征明显变差，说明时域形态信息不能丢。
2. `time_freq` 比纯 `time` 略好且更稳。
3. 在频谱方法中，`welch_psd` 略优于 `stft` 和 `rfft`，`dwt` 在当前设置下不占优。

### 6.3 正样本阈值和负样本重构结果

从系统实验记录看：

1. `Lifting(raising)` 的正样本阈值设为 10% 时优于 20% 和 30%
2. 对 `Lifting(lowering)` 的负样本中注入约 10% `Driving(curve)`，能把 `avg_mcc` 推到约 `0.7982`

这说明项目现在已经抓到了一个关键事实：

1. lifting 类的难点不是“样本太少”这么简单
2. 更核心的问题是边界窗口纯度不高，以及与 driving 类之间的近邻混淆

### 6.4 最新 XGBoost 运行结果

当前工作区最新一次完整 `run_summary.json` 位于 `output/plot/`，对应配置为：

1. `data=500`
2. `train_with_val=True`
3. `signal_combo=True`
4. `feature_engineering=True`
5. `feature_domain=time`
6. `spectrum_method=welch_psd`
7. 无增强、无 tsfresh

该次 XGBoost 结果为：

1. `avg_mcc = 0.7947`
2. `avg_macro_f1 = 0.8125`
3. `avg_macro_pr_auc = 0.8729`
4. `avg_macro_brier = 0.0336`

它和 LightGBM 主线相近，但没有超过前述 LightGBM 最优方案。

### 6.5 Optuna 调参结果

`out/optuna_lgbm/run_summary.json` 显示：

1. Optuna 1000 trials 版 LightGBM 的 `avg_mcc = 0.7926`
2. 没有超过手工实验链上最好的 `0.8051`

这意味着当前性能瓶颈已经不主要在树深、叶子数这类常规超参数上，而更多在：

1. 特征设计
2. 类边界样本选择
3. 负样本构成
4. 数据增强与标签纯度建模

### 6.6 深度模型结果

当前保存在仓库中的深度模型结果如下：

1. Swin + time_freq + instance + 全量训练：`avg_mcc = 0.6940`
2. Swin + time_freq + focal：`avg_mcc = 0.6895`
3. InceptionTime + time_freq + BCE：`avg_mcc = 0.6545`

这说明在当前数据规模、窗口设计和标签形态下，传统树模型配合结构化特征显著优于当前深度视觉化路线。

## 7. 当前结果还有哪些地方需要进一步优化

### 7.1 最难的类别仍然稳定是 Lifting(lowering)

从最新 XGBoost 和 Optuna-LGBM 的按类平均结果看，最弱类都稳定是：

1. `Lifting(lowering)`

而最强类几乎总是：

1. `Turntable wrapping`

这说明模型性能上限不是均匀受限，而是明显被某一类拖住了。

### 7.2 Lifting(raising) 也明显偏弱

`Lifting(raising)` 虽然略好于 `Lifting(lowering)`，但仍明显低于 driving 和 stationary 类。说明 lifting 两类是当前任务的主要短板，且这不是单一 fold 的偶发问题，而是跨 fold 稳定存在的模式。

### 7.3 当前增强链路证据不完整

仓库里有增强脚本 `run_lgbm_rotation_aug_grid.py`，而且脚本设定看起来是接在“当前最佳 time_freq + welch + raising threshold + negative mix”锚点之后继续做 rotation sweep。但当前工作区缺少对应完整输出目录，因此：

1. 增强方向是明确的
2. 但增强收益在本地快照里还不能视为完全闭环

### 7.4 tsfresh 还没有形成闭环结论

项目已经把 `tsfresh` 接入到统一特征提取框架，但最终实验结果未保留完整 summary，因此目前不能回答：

1. tsfresh 是否真正优于手工特征
2. tsfresh 的收益是否只出现在某些类上
3. tsfresh 是否带来了过拟合或训练耗时问题

### 7.5 训练目标仍然只基于窗口末端硬标签

虽然项目已经保存了 `label_ratio_pre_ds`，但最终预测目标仍然是窗口末端的 0/1 标签。这意味着：

1. 大量边界窗口在训练时被强行视为纯正或纯负
2. 模型并没有直接学习“窗口内部成分比例”

这很可能正是 lifting / driving 混淆长期存在的重要原因。

## 8. 下一步优化建议

我建议优先按下面顺序推进。

### 8.1 第一优先级：围绕 lifting 类做边界样本优化

建议动作：

1. 固定当前最优主线：LightGBM + feature_engineering + signal_combo + time_freq + welch_psd
2. 重点继续调 `Lifting(lowering)` 和 `Lifting(raising)` 的正样本阈值
3. 继续做针对 lifting 的负样本构成重配，尤其是与 `Driving(curve)`、`Driving(straight)` 的 hard negative 比例

原因：

1. 当前最弱类高度集中，优化 ROI 最高
2. 现有实验已经证明这条线能带来真实收益

### 8.2 第二优先级：把 label_ratio_pre_ds 真正纳入学习目标

建议动作：

1. 在训练时尝试把 `label_ratio_pre_ds` 用作 soft label
2. 或者至少做双头任务：一个头预测 endpoint label，一个头回归窗口纯度
3. 或在树模型侧把 ratio 衍生成更多样本权重或置信度机制

原因：

1. 当前项目已经计算了这个信息，但使用方式还偏保守
2. 这可能是解决边界窗口误判的最关键一步

### 8.3 第三优先级：补完增强实验闭环

建议动作：

1. 重新运行并保存 `exp_20260415_aug_rotation_grid` 结果
2. 只围绕当前最佳主线做最小必要增强网格，不再大面积扩参
3. 重点验证 rotation 在 `xz` 平面是否稳定优于其他平面

原因：

1. 当前已有脚本说明方向已经明确
2. 但缺少本地完整产物，无法正式纳入结论

### 8.4 第四优先级：补完 tsfresh 收尾实验

建议动作：

1. 只在当前最优主线上追加 `use_tsfresh=True`
2. 明确记录训练时长、内存开销、每折收益
3. 和纯手工特征方案做严格对照

原因：

1. tsfresh 已经接入，工程成本不高
2. 但必须避免“特征更多但收益很小”的低性价比情况

### 8.5 第五优先级：若继续做深度模型，应转回时序原生建模

如果后续还想推进深度学习路线，我不建议继续把当前任务主要做成 time-frequency 图像分类。更值得尝试的是：

1. 直接处理多通道时序的 1D CNN / TCN / Transformer
2. 显式建模多标签与边界窗口纯度
3. 结合 class-balanced loss 或 focal/BCE with soft targets

原因：

1. 当前深度图像化路线明显落后
2. 说明信息转换方式可能损失了时序结构优势

## 9. 总结

到目前为止，这个项目已经完成了从“原始多传感器数据整理”到“稳定的多标签分类主线”之间的大部分关键工作。现阶段最重要的结论可以概括为：

1. 数据侧：`step=500` 的 2 秒窗口化策略合理，且不会明显扭曲原始类别分布。
2. 标签侧：将 10 个细标签映射到 6 个超类是正确方向，减少了静态细粒度标签噪声。
3. 特征侧：手工时域特征、频域补充特征、按类信号组合都有效。
4. 方法侧：当前最强路线是 LightGBM 的 One-vs-Rest 方案，而不是当前保留的深度模型路线。
5. 结果侧：系统实验最优结果约为 `avg_mcc = 0.805`，已经比基础基线 `0.686` 有显著提升。
6. 短板侧：`Lifting(lowering)` 和 `Lifting(raising)` 是持续瓶颈，问题核心在于边界窗口纯度和 driving-lifting 混淆。
7. 下一步：最值得投入的不是继续盲目调超参数，而是继续做边界样本筛选、负样本重构、增强闭环和 soft-label 化。

## 10. 主要依据文件

1. `utils/config.py`
2. `data_handler.py`
3. `build_windowed_sample_manifests.py`
4. `models/LGBM.py`
5. `models/XGB.py`
6. `main.py`
7. `sample_stats.py`
8. `result/dataset_analysis_20260412/dataset_label_analysis.txt`
9. `result/exp_data_profile/exp_raw_vs_step500_summary.json`
10. `output/exp_20260415_lgbm_rt_v2/leaderboard.csv`
11. `output/plot/run_summary.json`
12. `out/optuna_lgbm/run_summary.json`
13. `result/ds_result/*.json`
