import os
from pathlib import Path


def _path_from_env(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default)


PROJECT_ROOT = _path_from_env("VECL_PROJECT_ROOT", Path(__file__).resolve().parents[1])
DATASET_ROOT = _path_from_env("VECL_DATASET_ROOT", PROJECT_ROOT / "Dataset")
MODEL_ROOT = _path_from_env("VECL_MODEL_ROOT", PROJECT_ROOT / "pretrain_model")
CKPT_ROOT = _path_from_env("VECL_CKPT_ROOT", PROJECT_ROOT / "data" / "ckpt")
OUTPUT_ROOT = _path_from_env("VECL_OUTPUT_DIR", PROJECT_ROOT / "data" / "output")

M3AE_VITB16_PATH = _path_from_env("VECL_M3AE_VITB16", MODEL_ROOT / "VITB-16-M3AE_last.ckpt")
BIOCLINICAL_BERT_PATH = os.environ.get("VECL_BIOCLINICAL_BERT", "Laihaoran/BioClinicalMPBERT")
FINAL_CKPT_PATH = _path_from_env("VECL_FINAL_CKPT", MODEL_ROOT / "vecl_miccai_final.ckpt")

OPENI_ROOT = _path_from_env("VECL_OPENI_ROOT", DATASET_ROOT / "OpenI" / "images")
CHEXPERT_ROOT = _path_from_env("VECL_CHEXPERT_ROOT", DATASET_ROOT / "Chexpert" / "images")
CHESTXRAY14_ROOT = _path_from_env("VECL_CHESTXRAY14_ROOT", DATASET_ROOT / "ChestXray14" / "images")
CHESTXDET10_ROOT = _path_from_env("VECL_CHESTXDET10_ROOT", DATASET_ROOT / "ChestXDet10")
PADCHEST_ROOT = _path_from_env("VECL_PADCHEST_ROOT", DATASET_ROOT / "PadChest" / "images")
MIMIC_CXR_JPG_ROOT = _path_from_env("VECL_MIMIC_CXR_JPG_ROOT", DATASET_ROOT / "MIMIC-CXR-JPG" / "2.0.0")


def repo_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)


def dataset_path(*parts):
    return DATASET_ROOT.joinpath(*parts)


def output_path(*parts):
    return OUTPUT_ROOT.joinpath(*parts)


def remap_image_path(path):
    if path is None:
        return None
    p = str(path)
    replacements = {
        "/defaultShare/OpenI/NLMCXR_png/": str(OPENI_ROOT) + "/",
        "/defaultShare/ChestX-Det10-Dataset/": str(CHESTXDET10_ROOT) + "/",
        "/defaultShare/CheXpert/test/": str(CHEXPERT_ROOT) + "/",
        "/defaultShare/PadChest/manualset/": str(PADCHEST_ROOT) + "/",
        "/mnt/nvme_share/wuwl/project/CARZero-main/": str(PROJECT_ROOT) + "/",
    }
    for old, new in replacements.items():
        if p.startswith(old):
            return p.replace(old, new, 1)
    if "ChestX-ray14" in p:
        return str(CHESTXRAY14_ROOT / p[38:])
    return p
