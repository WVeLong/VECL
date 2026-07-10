from pathlib import Path

try:
    from .paths import (
        PROJECT_ROOT, DATASET_ROOT, OPENI_ROOT, CHEXPERT_ROOT,
        CHESTXRAY14_ROOT, CHESTXDET10_ROOT, PADCHEST_ROOT,
        MIMIC_CXR_JPG_ROOT, dataset_path
    )
except ImportError:
    from VECL.paths import (
        PROJECT_ROOT, DATASET_ROOT, OPENI_ROOT, CHEXPERT_ROOT,
        CHESTXRAY14_ROOT, CHESTXDET10_ROOT, PADCHEST_ROOT,
        MIMIC_CXR_JPG_ROOT, dataset_path
    )

# MIMIC constants
MIMIC_DATA_DIR = dataset_path("MIMIC")
ABS_PATH = str(PROJECT_ROOT) + "/"
MIMIC_MASTER_CSV_XH =  "mimic-cxr-image-report-pair.csv"
SENT_Path = 'report_sentences.csv'
LABEL_Path = 'report_sentences_labels.csv'
MIMIC_label = 'MIMIC_multi_label_text.json'

MIMIC_VALID_NUM = 5000
MIMIC_VIEW_COL = "Frontal/Lateral"
MIMIC_PATH_COL = "Path"
MIMIC_SPLIT_COL = "Split"
MIMIC_REPORT_COL = "Report Impression"
MIMIC_LLM_REPORT_COL = "LLM Report Impression"
MIMIC_XH_REPORT_COL = "xinhuo"
MIMIC_LLM_REPORT_V1_COL = "LLM Report v1 Impression"
MIMIC_DataFlag_COL = "Data Flag"
MIMIC_RAMINDEX_COL = "Index"
MIMIC_Original_VIEW_COL = "OriginalView"

PWD_Path = str(MIMIC_CXR_JPG_ROOT) + "/"


# OpenI constants
OpenI_DATA_DIR = str(dataset_path("OpenI")) + "/"
OpenI_ABS_PATH = str(OPENI_ROOT) + "/"
OpenI_TRAIN_CSV = OpenI_DATA_DIR + "openi_multi_label_image.csv"  # train split from train.csv
OpenI_TEST_INPUT_CSV = OpenI_DATA_DIR + "test_input.csv"  # test input split from train.csv
OpenI_TEST_LABEL_CSV = OpenI_DATA_DIR + "test_label.csv"  # test label split from train.csv
OpenI_LABEL_CSV = OpenI_DATA_DIR + "custom.csv"
OpenI_VIEW_COL = "Frontal/Lateral"
OpenI_PATH_COL = "Path"
OpenI_pathologies = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "Fracture",
    "Opacity",
    "Lesion",
    "Calcified Granuloma",
    "Granuloma",
    "No_Finding"
]

# ChestXray14 constants
ChestXray14_DATA_DIR = str(dataset_path("ChestXray14")) + "/"
ChestXray14_ABS_PATH = str(CHESTXRAY14_ROOT.parent) + "/"
ChestXray14_TRAIN_DATA = ChestXray14_DATA_DIR + "test_list.txt"
ChestXray14_TEST_INPUT_CSV = ChestXray14_DATA_DIR + "test_input.csv"  # test input split from train.csv
ChestXray14_TEST_LABEL_CSV = ChestXray14_DATA_DIR + "test_label.csv"  # test label split from train.csv
ChestXray14_VIEW_COL = "Frontal/Lateral"
ChestXray14_PATH_COL = "path"
ChestXray14_pathologies = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Lung Mass',
    'Lung Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural Thickening',
    'Hernia'
]

# ChestXDet10 constants
ChestXDet10_DATA_DIR = str(dataset_path("ChestXDet10")) + "/"
ChestXDet10_ABS_PATH = str(CHESTXDET10_ROOT) + "/"
ChestXDet10_TRAIN_INPUT = ChestXDet10_DATA_DIR + "chestXDet10_test_image.csv"
ChestXDet10_TRAIN_LABEL = ChestXDet10_DATA_DIR + "test.json"
ChestXDet10_TEST_INPUT_CSV = ChestXDet10_DATA_DIR + "test_input.csv"  # test input split from train.csv
ChestXDet10_TEST_LABEL_CSV = ChestXDet10_DATA_DIR + "test_label.csv"  # test label split from train.csv
ChestXDet10_VIEW_COL = "Frontal/Lateral"
ChestXDet10_PATH_COL = "Path"
ChestXDet10_pathologies = [
    'Atelectasis',
    'Calcification',
    'Consolidation',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Fracture',
    'Mass',
    'Nodule',
    'Pneumothorax'
]
# ChestXDet10_pathologies_new = [
#     'Atelectasis',
#     'Consolidation',
#     'Effusion',
#     'Emphysema',
#     'Fibrosis',
#     'Mass',
#     'Nodule',
#     'Pneumothorax'
# ]