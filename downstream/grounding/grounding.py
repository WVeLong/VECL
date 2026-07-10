import argparse
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils import *
from PIL import Image
from VECL import VECL
from VECL.paths import FINAL_CKPT_PATH, CHESTXDET10_ROOT, dataset_path, remap_image_path



def conve_attention_map(ws, batch_size):
    ws = (ws[-4] + ws[-3] + ws[-2] + ws[-1]) / 4
    ws = ws.view(batch_size, ws.shape[1], 14, 14)
    pred_map = ws.detach().cpu().numpy()
    pred_map_expand = torch.from_numpy(pred_map.repeat(16, axis=2).repeat(16, axis=3))  # Final
    return pred_map_expand, pred_map

def process_attention_map_to_img(pred_map, original_size):
    attention_map_resized = torch.nn.functional.interpolate(torch.tensor(pred_map).unsqueeze(0).unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze().numpy()
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())
    y_center, x_center = np.unravel_index(np.argmax(attention_map_resized), attention_map_resized.shape)
    return x_center, y_center, attention_map_resized

def evaluate_prediction(pred_x, pred_y, gt_box, tolerance=20):
    """
    评估预测点是否在真实框内，加入容忍范围
    """
    xmin, ymin, xmax, ymax = gt_box  # 真实坐标的左上角和右下角坐标

    # 扩展边界框
    xmin = max(0, xmin - tolerance)
    ymin = max(0, ymin - tolerance)
    xmax = min(img_width, xmax + tolerance)
    ymax = min(img_height, ymax + tolerance)

    return xmin <= pred_x <= xmax and ymin <= pred_y <= ymax


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default=str(FINAL_CKPT_PATH))
    parser.add_argument('--image_csv', default=str(dataset_path('ChestXDet10', 'chestXDet10_test_image.csv')))
    parser.add_argument('--label_json', default=str(dataset_path('ChestXDet10', 'test.json')))
    args = parser.parse_args()
    CARZero_model = VECL.load_VECL(ckpt_path=args.ckpt_path, device=device)
    img_path = args.image_csv
    df = pd.read_csv(img_path)

    # 读取 JSON 文件
    with open(args.label_json, 'r') as f:
        data = json.load(f)
    box_dict = {i: item['boxes'] for i, item in enumerate(data)}
    syms_dict = {i: item['syms'] for i, item in enumerate(data)}  # 注意这里的键是整数，不是字符串
    file_dict = {i: item['file_name'] for i, item in enumerate(data)}

    length = len(syms_dict)
    hit_count = 0
    le = 0
    s_m = {0: "There is Atelectasis", 1: "There is Calcification", 2: "There is Consolidation", 3: "There is Effusion",
           4: "There is Emphysema", 5: "There is Fibrosis", 6: "There is Fracture", 7: "There is Mass",
           8: "There is Nodule", 9: "There is Pneumothorax"}
    # 使用列表推导来创建一个包含所有单元素字典的列表
    delete_index = []  # 用于记录需要删除的索引
    single_element_dicts = [{key: value} for key, value in s_m.items()]
    for syms_dict_choose in single_element_dicts:
        processed_txt = CARZero_model.process_class_prompts(syms_dict_choose, device)
        key, sm = syms_dict_choose.popitem()
        sm = sm[len('There is '):]
        txts = processed_txt[key]
        for i in range(length):
            if sm in syms_dict[i]:
                le += 1
            for j in range(len(syms_dict[i])):
                if syms_dict[i][j] in sm:
                    index = syms_dict[i].index(syms_dict[i][j])
                    box_d = box_dict[i][index]
                    processed_img = CARZero_model.process_img(remap_image_path(df['Path'].tolist()[i]), device, delete_index, 0, 1)
                    # 获取原图尺寸
                    original_image = Image.open(remap_image_path(df['Path'].tolist()[i]))
                    img_width, img_height = original_image.size
                    with torch.no_grad():
                        CARZero_model.eval()
                        label_img_emb_l, label_img_emb_g = CARZero_model.image_encoder_forward(processed_img)
                        query_emb_l, query_emb_g, _ = CARZero_model.text_encoder_forward(
                            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"])
                        bs = label_img_emb_g.size(0)  # 获取批次数目
                        label_img_emb_l_ = label_img_emb_l.view(label_img_emb_l.size(0), label_img_emb_l.size(1), -1)  # patch_num  dim
                        label_img_emb_l_ = label_img_emb_l_.permute(0, 2, 1)
                        query_emb_l_ = query_emb_l.view(query_emb_l.size(0), query_emb_l.size(1), -1)
                        query_emb_l_ = query_emb_l_.permute(0, 2, 1)  # patch_num b dim # [97, 512, 768]
                        i2t_cls, atten_i2t = CARZero_model.fusion_module(torch.cat([label_img_emb_g.unsqueeze(1), label_img_emb_l_], dim=1), query_emb_g, return_atten=True)
                        atten_i2t = [atten[:, :, 1:] for atten in atten_i2t]
                        atte = atten_i2t
                        pred_map, pred_map_min = conve_attention_map(atte, batch_size=bs)
                        pred_map = pred_map.cpu().numpy()
                        if len(pred_map.shape) > 2:
                            pred_map = pred_map.squeeze()
                        x_pred, y_pred, attention_vis = process_attention_map_to_img(pred_map, (original_image.size[1], original_image.size[0]))
                        if evaluate_prediction(x_pred, y_pred, box_d, tolerance=20):
                            hit_count += 1
        print(f"满足要求的总数:{le} ")
        print(f"hit_count: {hit_count}")
        accuracy = hit_count / le
        print(f"Pointing Game Accuracy about {sm}: {accuracy:.3f}")