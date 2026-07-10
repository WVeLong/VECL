# Dataset redistribution notes

The Git repository includes lightweight CSV/JSON/TXT metadata where possible. Raw images are never redistributed.

Large MIMIC-CXR report-derived files and pickle caches are not committed because they are large and may be subject to the original PhysioNet credentialed-data terms. Regenerate them from the official MIMIC-CXR/MIMIC-CXR-JPG downloads using the helper scripts, or publish them only if you have confirmed redistribution rights.

Ignored examples:

- Dataset/MIMIC/report_sentences.csv
- Dataset/MIMIC/mimic-cxr-image-report-pair.csv
- Dataset/MIMIC/mimic-cxr-label-LLM_report-xinhuo-chexpertformat.csv
- Dataset/MIMIC/captions*.pickle
- Dataset/MIMIC/retrieval_report_corpus.pickle
- Dataset/MIMIC/chexper_labeler/*.pickle

- Dataset/MIMIC/report_sentences_labels.csv is excluded from Git because it is larger than GitHub's recommended file size; regenerate it from MIMIC report labels when needed.
