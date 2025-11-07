# data_reader/pathmnist_extractor.py
import os
import numpy as np

def _one_hot(n_classes, idx):
    v = np.zeros((n_classes,), dtype=np.float32)
    v[int(idx)] = 1.0
    return v

def _to_float01(x):
    # medmnist gives uint8 [0..255], shape (H,W,C)
    return (x.astype(np.float32) / 255.0)

def pathmnist_extract_all(file_path=os.path.dirname(__file__)):
    """
    Returns:
      train_image: [Ntr, 2352] float32 in [0,1]
      train_label: [Ntr, 9]    one-hot float32
      test_image:  [Nte, 2352] float32
      test_label:  [Nte, 9]    one-hot
      train_label_orig: list[int] of original labels for server index logic
    """
    import medmnist
    PathMNIST = getattr(medmnist.dataset, 'PathMNIST')
    n_classes = 9

    ds_tr  = PathMNIST(split='train', download=True)
    ds_val = PathMNIST(split='val',   download=True)
    ds_te  = PathMNIST(split='test',  download=True)

    # concat train + val for training
    tr_imgs = np.concatenate([ds_tr.imgs, ds_val.imgs], axis=0)  # (Ntr,H,W,C)
    tr_lbls = np.concatenate([ds_tr.labels, ds_val.labels], axis=0).reshape(-1)  # (Ntr,)

    te_imgs = ds_te.imgs
    te_lbls = ds_te.labels.reshape(-1)

    # normalize & flatten
    tr_imgs = _to_float01(tr_imgs).reshape((tr_imgs.shape[0], -1)).astype(np.float32)  # [Ntr, 2352]
    te_imgs = _to_float01(te_imgs).reshape((te_imgs.shape[0], -1)).astype(np.float32)  # [Nte, 2352]

    # one-hot
    tr_oh = np.stack([_one_hot(n_classes, i) for i in tr_lbls], axis=0)
    te_oh = np.stack([_one_hot(n_classes, i) for i in te_lbls], axis=0)

    # keep original (int) labels for server’s index helpers
    tr_lbls_orig = tr_lbls.astype(np.int32).tolist()

    return tr_imgs, tr_oh, te_imgs, te_oh, tr_lbls_orig