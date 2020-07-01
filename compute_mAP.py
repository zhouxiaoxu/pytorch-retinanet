# coding: utf-8

# 按照置信度降序排序
sorted_ind = np.argsort(-confidence)    # 按照confidence排序，返回的排序的index
BB = BB[sorted_ind, :]   # 按照index排序的顺序，重排BB
image_ids = [image_ids[x] for x in sorted_ind] # 各个预测框的对应图片id

# 遍历预测框，并统计TPs和FPs
nd = len(image_ids)     # 有多少个预测框
tp = np.zeros(nd)       # true positive 数量
fp = np.zeros(nd)       # false postive 数量  precision = tp/(tp+fp)
for d in range(nd):
    R = class_recs[image_ids[d]]
    bb = BB[d, :].astype(float)
    ovmax = -np.inf
    BBGT = R['bbox'].astype(float)  # ground truth

    if BBGT.size > 0:
        # 计算IoU
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
    # 取最大的IoU
    if ovmax > ovthresh:  # 是否大于阈值
        if not R['difficult'][jmax]:  # 非difficult物体
            if not R['det'][jmax]:    # 未被检测
                tp[d] = 1.
                R['det'][jmax] = 1    # 标记已被检测
            else:
                fp[d] = 1.
    else:
        fp[d] = 1.

# 计算precision recall
fp = np.cumsum(fp)
tp = np.cumsum(tp)
rec = tp / float(npos)
# avoid divide by zero in case the first detection matches a difficult
# ground truth
prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
