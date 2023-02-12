import torch
import torch.nn.functional as F

def downscale_label_ratio(gt,
                          trg_h, trg_w,
                          min_ratio,
                          n_classes,
                          ignore_index=255):
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    ignore_substitute = n_classes

    out = gt.clone()  # otw. next line would modify original gt
    out[out == ignore_index] = ignore_substitute
    out = F.one_hot(
        out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
    assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
    out = F.interpolate(out.float(), size=(trg_h, trg_w), mode="bilinear", align_corners=True)

    gt_ratio, out = torch.max(out, dim=1, keepdim=True)
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out