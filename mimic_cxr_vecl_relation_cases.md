# MIMIC-CXR VECL 支持-中立-矛盾关系样例

本文档从 VECL 项目中已有的 MIMIC-CXR 图像-报告配对与句子级标签中筛选样例，用于展示 VECL 标签空间下的三类图文关系：支持、中立、矛盾。筛选过程未重新调用 LLM，也未重新抽取标签。

## 数据来源

| 项目 | 路径 |
|---|---|
| 图像-报告配对 | `/hdd/wuwl/project/VECL/Dataset/MIMIC/mimic-cxr-image-report-pair.csv` |
| 报告切句 | `/hdd/wuwl/project/VECL/Dataset/MIMIC/report_sentences.csv` |
| VECL 句子级标签 | `/hdd/wuwl/project/VECL/Dataset/MIMIC/report_sentences_labels.csv` |
| 标签文本映射 | `/hdd/wuwl/project/VECL/Dataset/MIMIC/MIMIC_multi_label_text.json` |
| 标签提示词定义 | `/hdd/wuwl/project/VECL/helper/LLM_prompt.py` |

图像路径按 VECL 训练代码中的规则恢复：将 CSV 中的 `/defaultShare/MIMIC-CXR/` 替换为 `/hdd/wuwl/project/VECL/Dataset/MIMIC-CXR-JPG/2.0.0/`。

## VECL 标签说明

`+` 表示该疾病/征象为阳性发现，`-` 表示该疾病/征象被否定，`25` 表示 No Mention。本样例中用到的标签如下：

| 标签 | 含义 |
|---|---|
| `2+` | 胸腔积液阳性 |
| `2-` | 胸腔积液阴性 |
| `3-` | 气胸阴性 |
| `4+` | 心影增大阳性 |
| `4-` | 心脏/心肺急性异常阴性 |
| `6+` | 肺炎阳性 |
| `7+` | 肺部肿块阳性 |
| `8+` | 肺水肿阳性 |
| `9+` | 肺结节阳性 |
| `13+` | 胸膜增厚/其他胸膜异常阳性 |
| `15+` | 肺实变阳性 |
| `15-` | 肺实变阴性 |
| `16+` | 骨折/陈旧骨折阳性 |
| `16-` | 骨性结构急性异常阴性 |
| `17-` | 纵隔/心纵隔增宽阴性 |
| `20+` | 支持装置/术后夹等阳性 |
| `25` | 未提及上述标签 |

## 样本清单

### 样本 S1：多种阳性肺部异常

| 字段 | 内容 |
|---|---|
| CSV 行号 | `16` |
| 原始图像路径 | `/defaultShare/MIMIC-CXR/files/p10/p10000980/s54935705/6ad819bb-bae74eb9-7b663e90-b8deabd7-57f8054a.jpg` |
| 本地图像路径 | `/hdd/wuwl/project/VECL/Dataset/MIMIC-CXR-JPG/2.0.0/files/p10/p10000980/s54935705/6ad819bb-bae74eb9-7b663e90-b8deabd7-57f8054a.jpg` |
| 视角 | Frontal |
| 报告摘录 | there is mild pulmonary edema with superimposed region of more confluent consolidation in the left upper lung. there are possible small bilateral pleural effusion. moderate cardiomegaly is again seen as well as tortuosityof the descending thoracic aorta. no acute osseous abnormalities. mild pulmonary edema with superimposed left upper lung consolidation potentially more confluent edema versus superimposed infection. |

![S1 X-ray](/hdd/wuwl/project/VECL/Dataset/MIMIC-CXR-JPG/2.0.0/files/p10/p10000980/s54935705/6ad819bb-bae74eb9-7b663e90-b8deabd7-57f8054a.jpg)

| 句子 | VECL 标签 |
|---|---|
| there is mild pulmonary edema with superimposed region of more confluent consolidation in the left upper lung. | `8+` |
| there are possible small bilateral pleural effusion. | `2+` |
| moderate cardiomegaly is again seen as well as tortuosity of the descending thoracic aorta. | `4+` |
| no acute osseous abnormalities. | `16-` |
| There is mild pulmonary edema with superimposed left upper lung consolidation potentially more confluent edema versus superimposed infection. | `8+, 15+` |
| There may be small bilateral pleural effusion. | `2+, 13+` |
| There is moderate cardiomegaly as well as tortuosity of the descending thoracic aorta. | `4+` |

### 样本 S2：否定实变、胸腔积液和气胸，伴结节影与陈旧肋骨异常

| 字段 | 内容 |
|---|---|
| CSV 行号 | `0` |
| 原始图像路径 | `/defaultShare/MIMIC-CXR/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg` |
| 本地图像路径 | `/hdd/wuwl/project/VECL/Dataset/MIMIC-CXR-JPG/2.0.0/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg` |
| 视角 | Frontal |
| 报告摘录 | there is no focal consolidation pleural effusion or pneumothorax. bilateral nodular opacities that most likely represent nipple shadows. the cardiomediastinal silhouette is normal. clips project over the left lung potentially within the breast. chronic deformity of the posterior left sixth and seventh ribs are noted. no acute cardio pulmonary process. |

![S2 X-ray](/hdd/wuwl/project/VECL/Dataset/MIMIC-CXR-JPG/2.0.0/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg)

| 句子 | VECL 标签 |
|---|---|
| there is no focal consolidation pleural effusion or pneumothorax. | `15-, 2-, 3-` |
| bilateral nodular opacities that most likely represent nipple shadows. | `9+` |
| the cardiomediastinal silhouette is normal. | `17-` |
| clips project over the left lung potentially within the breast. | `20+` |
| chronic deformity of the posterior left sixth and seventh ribs are noted. | `16+` |
| no acute cardio pulmonary process. | `4-` |

### 样本 S3：右上肺肺炎或肿块

| 字段 | 内容 |
|---|---|
| CSV 行号 | `14` |
| 原始图像路径 | `/defaultShare/MIMIC-CXR/files/p10/p10000980/s51967283/943486a3-b3fa9ff7-50f5a769-7a62fcbb-f39b6da4.jpg` |
| 本地图像路径 | `/hdd/wuwl/project/VECL/Dataset/MIMIC-CXR-JPG/2.0.0/files/p10/p10000980/s51967283/943486a3-b3fa9ff7-50f5a769-7a62fcbb-f39b6da4.jpg` |
| 视角 | Frontal |
| 报告摘录 | right upper lobe pneumonia or mass. however given right hilar fullness a mass resulting in post-obstructive pneumonia is within the differential. recommend chest ct with intravenous contrast for further assessment. |

![S3 X-ray](/hdd/wuwl/project/VECL/Dataset/MIMIC-CXR-JPG/2.0.0/files/p10/p10000980/s51967283/943486a3-b3fa9ff7-50f5a769-7a62fcbb-f39b6da4.jpg)

| 句子 | VECL 标签 |
|---|---|
| right upper lobe pneumonia or mass. | `6+` |
| however given right hilar fullness a mass resulting in post-obstructive pneumonia is within the differential. | `7+` |
| There is right upper lobe pneumonia or mass. | `6+, 7+` |
| There may be post-obstructive pneumonia within the differential. | `6+` |

## 支持-中立-矛盾关系组合

| 图像样本 | 文本样本 | 关系 | 证据标签 | 解释 |
|---|---|---|---|---|
| S1 | S1 | 支持 | 图像对应报告中包含 `8+`、`15+`、`2+`、`4+` | S1 的原始报告描述肺水肿、左上肺实变、双侧胸腔积液和心影增大，与同一图像配对，属于标准支持关系。 |
| S1 | S2 | 矛盾 | S1 有 `2+`、`15+`；S2 文本有 `2-`、`15-` | S1 图像对应文本提示胸腔积液和实变阳性，而 S2 文本明确否定胸腔积液和实变，因此以 S1 图像配 S2 文本时构成矛盾。 |
| S1 | S3 | 中立 | S1 主要为 `8+`、`15+`、`2+`、`4+`；S3 主要为 `6+`、`7+` | S3 文本描述右上肺肺炎或肿块，没有直接否定 S1 的胸腔积液、肺水肿、实变或心影增大，也不是同一图像的报告，因此可作为中立关系样例。 |
| S2 | S2 | 支持 | 同一样本文本包含 `15-`、`2-`、`3-`、`9+`、`16+`、`20+` | S2 图像与其原始报告配对，报告中对实变、积液、气胸给出否定，同时描述结节影、术后夹和陈旧肋骨异常，属于支持关系。 |
| S2 | S1 | 矛盾 | S2 有 `2-`、`15-`；S1 文本有 `2+`、`15+` | S2 文本否定胸腔积液和实变，而 S1 文本报告这些异常为阳性，反向配对时同样构成矛盾。 |
| S3 | S3 | 支持 | 同一样本文本包含 `6+`、`7+` | S3 图像与其原始报告配对，文本描述右上肺肺炎或肿块，与 VECL 标签一致，属于支持关系。 |

## 可复现筛选规则

1. 从 `mimic-cxr-image-report-pair.csv`、`report_sentences.csv`、`report_sentences_labels.csv` 按行号同步读取样本。
2. 仅保留 `Frontal/Lateral = Frontal` 且本地图像文件存在的样本。
3. 将每个样本的句子级标签拆分为集合，例如 `15-, 2-, 3-` 拆为 `15-`、`2-`、`3-`。
4. 优先选择报告较短、标签极性清晰、具有典型阳性或阴性描述的样本。
5. 支持关系使用同一图像和同一报告文本。
6. 矛盾关系要求至少一个相同疾病编号在两个文本/图像语义中出现相反极性，例如 `2+` 与 `2-`、`15+` 与 `15-`。
7. 中立关系要求主要阳性标签不重叠，且不存在相同疾病编号的相反极性。

