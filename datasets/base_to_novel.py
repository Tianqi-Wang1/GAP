# datasets/base_to_novel.py

import math
from dassl.data.datasets import Datum


def _clone_datum(item, new_label):
    kwargs = {
        "impath": item.impath,
        "label": int(new_label),
        "classname": item.classname,
    }

    if hasattr(item, "domain"):
        kwargs["domain"] = item.domain

    return Datum(**kwargs)


def subsample_classes(*datasets, subsample="all"):
    """
    CoOp/CLIP-SVD style base-to-novel split.

    all  : keep all classes
    base : first ceil(C/2) classes after sorting labels
    new  : remaining classes

    The labels are remapped to 0, ..., C_sub-1.
    """
    assert subsample in ["all", "base", "new"]

    if subsample == "all":
        return datasets if len(datasets) > 1 else datasets[0]

    all_labels = sorted({
        item.label
        for dataset in datasets
        for item in dataset
    })

    num_classes = len(all_labels)
    split = math.ceil(num_classes / 2)

    if subsample == "base":
        selected = all_labels[:split]
    else:
        selected = all_labels[split:]

    relabeler = {old_label: new_label for new_label, old_label in enumerate(selected)}

    new_datasets = []
    for dataset in datasets:
        new_dataset = []
        for item in dataset:
            if item.label in relabeler:
                new_dataset.append(_clone_datum(item, relabeler[item.label]))
        new_datasets.append(new_dataset)

    print(
        f">> SUBSAMPLE_CLASSES={subsample}: "
        f"{len(selected)}/{num_classes} classes"
    )

    return new_datasets if len(new_datasets) > 1 else new_datasets[0]