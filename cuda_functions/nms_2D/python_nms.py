import numpy as np
import torch


def gpu_nms(keep, num_out, boxes, nms_overlap_thresh):
    """
    Pure-Python GPU-compatible NMS fallback.
    - `boxes` is expected to be a tensor of shape (N, 5) and already sorted by score descending.
    - `keep` is a LongTensor (preallocated) where kept indices (into the sorted boxes) will be written.
    - `num_out` is a LongTensor of size 1 where the number of kept indices will be written.
    Returns 1 on success (match C extension behavior).
    """
    if boxes.numel() == 0:
        num_out[0] = 0
        return 1

    if boxes.is_cuda:
        boxes = boxes.cpu()
    boxes = boxes.contiguous().detach()

    x1 = boxes[:, 0].numpy()
    y1 = boxes[:, 1].numpy()
    x2 = boxes[:, 2].numpy()
    y2 = boxes[:, 3].numpy()

    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    N = boxes.size(0)
    suppressed = np.zeros((N,), dtype=np.uint8)

    keep_list = []
    for i in range(N):
        if suppressed[i]:
            continue
        keep_list.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for j in range(i + 1, N):
            if suppressed[j]:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1.0)
            h = max(0.0, yy2 - yy1 + 1.0)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_overlap_thresh:
                suppressed[j] = 1

    num = len(keep_list)
    for idx in range(num):
        keep[idx] = int(keep_list[idx])
    num_out[0] = num
    return 1


def cpu_nms(keep_out, num_out, boxes, order, areas, nms_overlap_thresh):
    """
    CPU NMS compatible with the original C signature.
    - `boxes` shape (N, 4)
    - `order` is a LongTensor of sorted indices (by score desc)
    - `areas` is a FloatTensor of areas per box
    Writes original box indices into `keep_out` and the count into `num_out[0]`.
    """
    if boxes.numel() == 0:
        num_out[0] = 0
        return 1

    if boxes.is_cuda:
        boxes = boxes.cpu()
    if order.is_cuda:
        order = order.cpu()
    if areas.is_cuda:
        areas = areas.cpu()

    boxes = boxes.contiguous().detach()
    order = order.contiguous().detach()
    areas = areas.contiguous().detach()

    boxes_np = boxes.numpy()
    order_np = order.numpy()
    areas_np = areas.numpy()

    N = boxes_np.shape[0]
    suppressed = np.zeros((N,), dtype=np.uint8)

    num_to_keep = 0
    for _i in range(N):
        i = int(order_np[_i])
        if suppressed[i]:
            continue
        keep_out[num_to_keep] = i
        num_to_keep += 1

        ix1 = boxes_np[i, 0]
        iy1 = boxes_np[i, 1]
        ix2 = boxes_np[i, 2]
        iy2 = boxes_np[i, 3]
        iarea = areas_np[i]

        for _j in range(_i + 1, N):
            j = int(order_np[_j])
            if suppressed[j]:
                continue
            xx1 = max(ix1, boxes_np[j, 0])
            yy1 = max(iy1, boxes_np[j, 1])
            xx2 = min(ix2, boxes_np[j, 2])
            yy2 = min(iy2, boxes_np[j, 3])
            w = max(0.0, xx2 - xx1 + 1.0)
            h = max(0.0, yy2 - yy1 + 1.0)
            inter = w * h
            ovr = inter / (iarea + areas_np[j] - inter)
            if ovr >= nms_overlap_thresh:
                suppressed[j] = 1

    num_out[0] = num_to_keep
    return 1


def nms_gpu(dets, thresh):
    """High-level wrapper compatible with pth_nms.nms_gpu.
    Accepts `dets` of shape (N,5) and returns indices of kept boxes (tensor).
    """
    scores = dets[:, 4]
    order = scores.sort(0, descending=True)[1]
    dets_sorted = dets[order].contiguous()

    keep = torch.LongTensor(dets_sorted.size(0))
    num_out = torch.LongTensor(1)
    # call low-level implementation
    gpu_nms(keep, num_out, dets_sorted, thresh)

    # map kept indices back to original ordering
    if num_out[0] == 0:
        return torch.LongTensor([])
    kept = keep[:num_out[0]]
    return order[kept.cuda()].contiguous() if dets.is_cuda else order[kept].contiguous()


def nms_cpu(dets, thresh):
    """High-level wrapper compatible with pth_nms.nms_cpu.
    Returns `keep` tensor containing kept indices (in sorted-order reference).
    """
    dets = dets.cpu().detach()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    cpu_nms(keep, num_out, dets, order, areas, thresh)

    return keep[:num_out[0]]
