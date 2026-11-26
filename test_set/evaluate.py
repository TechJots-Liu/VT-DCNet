import json
from pathlib import Path

import numpy as np

# BASE_DIR = Path(__file__).resolve().parent
PRED_JSON = r"test_predictions.json.json"
GT_JSON = r"test_label.json"


def remap_by_file_name(records, name_to_id):
    remapped = []
    for rec in records:
        rec = dict(rec)
        file_name = rec.get("file_name")
        if file_name is not None:
            rec["image_id"] = name_to_id.get(Path(file_name).name)
        remapped.append(rec)
    return remapped


def match_predictions(pred_cls, true_cls, iou, iouv):
    correct = np.zeros((len(pred_cls), len(iouv)), dtype=bool)
    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    for i, threshold in enumerate(iouv):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def box_iou(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)))
    gx1, gy1, gx2, gy2 = gt.T
    px1, py1, px2, py2 = pred.T
    iw = np.clip(np.minimum(gx2[:, None], px2[None, :]) - np.maximum(gx1[:, None], px1[None, :]), 0, None)
    ih = np.clip(np.minimum(gy2[:, None], py2[None, :]) - np.maximum(gy1[:, None], py1[None, :]), 0, None)
    inter = iw * ih
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    pred_area = (px2 - px1) * (py2 - py1)
    return inter / (gt_area[:, None] + pred_area[None, :] - inter + 1e-16)


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", np.trapz)
    return trapz(np.interp(x, mrec, mpre), x)


def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    x = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        mask = pred_cls == c
        n_l = nt[ci]
        if mask.sum() == 0 or n_l == 0:
            continue

        fpc = (1 - tp[mask]).cumsum(0)
        tpc = tp[mask].cumsum(0)

        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[mask], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[mask], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    best = smooth(f1_curve.mean(0), 0.1).argmax()
    return p_curve[:, best], r_curve[:, best], ap


def main():
    with open(GT_JSON, encoding="utf-8") as f:
        gt = json.load(f)
    with open(PRED_JSON, encoding="utf-8") as f:
        preds = json.load(f)

    name_to_id = {Path(img["file_name"]).name: img["id"] for img in gt["images"]}
    gt_by_image = {img["id"]: {"boxes": [], "cls": []} for img in gt["images"]}
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        entry = gt_by_image[ann["image_id"]]
        entry["boxes"].append([x, y, x + w, y + h])
        entry["cls"].append(int(ann["category_id"]))

    preds = remap_by_file_name(preds, name_to_id)
    pred_by_image = {img["id"]: {"boxes": [], "cls": [], "conf": []} for img in gt["images"]}
    dropped = 0
    for rec in preds:
        image_id = rec.get("image_id")
        if image_id not in pred_by_image:
            dropped += 1
            continue
        x, y, w, h = rec["bbox"]
        entry = pred_by_image[image_id]
        entry["boxes"].append([x, y, x + w, y + h])
        entry["cls"].append(int(rec["category_id"]))
        entry["conf"].append(float(rec["score"]))

    iouv = np.linspace(0.5, 0.95, 10)

    tp_list = []
    conf_list = []
    pred_cls_list = []
    target_cls_list = []

    for image_id, g in gt_by_image.items():
        p = pred_by_image[image_id]
        target_cls_list.extend(g["cls"])

        if len(p["cls"]) == 0:
            continue

        pred_cls = np.array(p["cls"])
        conf = np.array(p["conf"])

        if len(g["cls"]) == 0:
            tp_list.append(np.zeros((len(pred_cls), len(iouv)), dtype=bool))
        else:
            gt_cls = np.array(g["cls"])
            iou = box_iou(np.array(g["boxes"], dtype=float), np.array(p["boxes"], dtype=float))
            tp_list.append(match_predictions(pred_cls, gt_cls, iou, iouv))

        conf_list.extend(conf.tolist())
        pred_cls_list.extend(pred_cls.tolist())

    tp = np.concatenate(tp_list, 0) if tp_list else np.zeros((0, len(iouv)), dtype=bool)
    conf = np.array(conf_list)
    pred_cls = np.array(pred_cls_list)
    target_cls = np.array(target_cls_list)

    p, r, ap = ap_per_class(tp, conf, pred_cls, target_cls)

    mp = float(p.mean()) if len(p) else 0.0
    mr = float(r.mean()) if len(r) else 0.0
    map50 = float(ap[:, 0].mean()) if len(ap) else 0.0
    map75 = float(ap[:, 5].mean()) if len(ap) else 0.0
    map_all = float(ap.mean()) if len(ap) else 0.0

    print("\n" + "=" * 60)
    print(f"  mAP50:        {map50:.6f}")
    print(f"  mAP50-95:     {map_all:.6f}")
    print(f"  mAP75:        {map75:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
