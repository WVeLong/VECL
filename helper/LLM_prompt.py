def Category_Classify_Prompt(sent):
    prompt = """
Assume you are an experienced radiologist. Help me identify the correct medical condition label for the given radiology report sentence. Below are the medical condition labels with corresponding numbers and their medical descriptions:

1.Atelectasis: Lung tissue exhibits signs of partial or complete atelectasis, with decreased lung volume and increased localized radiographic density.
2.Pleural Effusion: There is an abnormal accumulation of fluid within the pleural cavity, which is evident on X-ray imaging as a blunted costophrenic angle or the presence of a fluid level.
3.Pneumothorax: There is evidence of free air within the thoracic cavity, resulting in partial or complete atelectasis. This is characterized by a retracted lung edge and an area devoid of lung markings.
4.Cardiomegaly: The cardiac silhouette is enlarged, indicating a size beyond the normal range.
5.Opacity: A lung region demonstrates increased radiopacity, potentially suggestive of inflammatory changes, a tumor, or hemorrhage.
6.Pneumonia: Patchy to diffuse lung infiltrates are observed, frequently associated with air bronchograms.
7.Pulmonary Mass: A lung mass, either well-circumscribed or poorly defined, is present, typically measuring over 3 cm in diameter.
8.Edema: Interstitial or alveolar fluid accumulation in the lungs is evident, characterized by increased and indistinct lung markings, a common finding in cardiogenic pulmonary edema.
9.Lung Nodule: A round or oval opacity within the lung, measuring less than 3 cm in diameter.
10.Lung Infiltration: Patchy or reticular opacities are noted within the lung tissue, suggestive of inflammatory processes or other infiltrative conditions.
11.Fibrosis: Interstitial lung thickening and fibrosis are present, exhibiting a reticular pattern and a honeycomb-like appearance on imaging studies.
12.Emphysema: Overinflation of the lungs with alveolar destruction is observed, manifesting as increased lung lucency and expanded lung fields on the chest X-ray.
13.Pleural Thickening: Pleural layer thickening is evident, appearing as broadened pleural shadows on imaging, often attributed to chronic inflammation or fibrosis.
14.Hernia: Internal organ protrusion through either normal or abnormal openings is observed; in the case of a diaphragmatic hernia, this may present as an abnormal position and contour of the diaphragm on X-ray imaging.
15.Consolidation: The lung alveoli are opacified with fluid or solid material, manifesting as regions of increased density with indistinct borders on imaging. the lung tissue exhibits a solidified appearance, a common finding in cases of pneumonia.
16.Bone Fracture: A discontinuity within the bone structure is observed on X-ray, characterized by a disrupted cortical bone and the presence of fracture lines.
17.Enlarged Cardiomediastinum: An enlargement of the mediastinal shadow or cardiomediastinal silhouette is noted.
18.Pleural Other: Other pleural abnormalities, such as pleural calcifications or plaques, are present, exhibiting specific imaging features that are indicative of the underlying condition.
19.Lung Lesion: An encompassing term for a variety of abnormal imaging findings within the lung, encompassing nodules, masses, infiltrates, and other anomalies.
20.Support Devices: Visualized on imaging are various medical devices, including catheters, stents, prosthetic heart valves, and other implanted or inserted medical apparatus.
21.Abnormal Lesion: A non-specific term denoting any abnormal imaging findings within the lungs, which may encompass a range of presentations such as nodules, masses, opacities, or other anomalies.
22.Lung Granuloma: A small pulmonary nodule, usually measuring less than 3 cm in diameter, frequently exhibiting calcification.
23.Calcified Granuloma: A calcified granuloma is observed, manifesting as a high-density nodule on imaging studies.
24.Tissue Calcification: Calcifications within soft tissue are noted, appearing as areas of increased density on imaging, indicative of calcified spots or plaques.
25.No Mention: None of the above symptoms are mentioned or related, cannot exist with any other labels at the same time.

Examples:

Sentence: there is no focal consolidation pleural effusion or pneumothorax.
Label: 15, 2, 3

Sentence: bilateral nodular opacities that most likely represent nipple shadows.
Label: 9

Sentence: chronic deformity of the posterior left sixth and seventh ribs are noted.
Label: 16

Sentence: the patient shows no signs of free air below the right hemidiaphragm.
Label: 3

Sentence: the imaged upper abdomen shows no remarkable findings.
Label: 25

Sentence: the patient's overall condition is normal.
Label: 25

Special Note:
If the sentence mentions a condition (whether positive or negative), use the corresponding label.
If the sentence describes multiple conditions, except 25(No Mention), output multiple labels, separated by commas ",".
Remember, 25(No Mention) and other labels cannot exist at the same time.

In case you forgot, let me repeat these labels:
1. Atelectasis
2. Pleural Effusion
3. Pneumothorax
4. Cardiomegaly
5. Opacity
6. Pneumonia
7. Pulmonary Mass
8. Edema
9. Lung Nodule
10. Lung Infiltration
11. Fibrosis
12. Emphysema
13. Pleural Thickening
14. Hernia
15. Consolidation
16. Bone Fracture
17. Enlarged Cardiomediastinum
18. Pleural Other
19. Lung Lesion
20. Support Devices
21. Abnormal Lesion
22. Lung Granuloma
23. Calcified Granuloma
24. Tissue Calcification
25. No Mention

Now, please select the most appropriate label for the following sentence and output only the corresponding number(s).
Notice: you only need to output the label (pure numbers), do not output anything else, Directly provide the answer without any reasoning!!!
"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": sent}
    ]
    return messages


def Status_Classify_Prompt(sent):
    prompt = """
Assume you are an experienced radiologist. Please help me determine whether the disease mentioned in the sentence is positive or negative. If it is positive, return +; if it is negative, return -.

Examples:

Sentence: there is no focal consolidation pleural effusion or pneumothorax.
Output: -

Sentence: bilateral nodular opacities that most likely represent nipple shadows.
Label: +

Sentence: chronic deformity of the posterior left sixth and seventh ribs are noted.
Label: +

Sentence: the patient shows no signs of free air below the right hemidiaphragm.
Label: -

Sentence: the imaged upper abdomen shows no remarkable findings.
Label: -

Sentence: the patient's overall condition is normal.
Label: -

Special Note:
Even if multiple diseases are mentioned in the sentence, you only need to return a + or -, because generally, multiple diseases mentioned in a sentence are either all positive or all negative.
If no disease is mentioned, just return -.

Now, please determine whether the disease mentioned in the follow sentence is positive or negative.
Notice: you only need to output + or -, do not output anything else, Directly provide the answer without any reasoning!!!
"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": sent}
    ]
    return messages