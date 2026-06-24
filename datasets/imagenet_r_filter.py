import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

from .imagenet import ImageNet


TO_BE_IGNORED = ["README.txt"]


@DATASET_REGISTRY.register()
class ImageNetRFilter(DatasetBase):
    """ImageNet-A with 1000-way classifier construction but 200-way filtered evaluation."""

    dataset_dir = "imagenet-rendition"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "imagenet-r")

        # 1. Read ImageNet-A's own 200-class names.
        ar_text_file = os.path.join(self.dataset_dir, "classnames.txt")
        ar_classnames = ImageNet.read_classnames(ar_text_file)

        # 2. Read full ImageNet-1K class names.
        imagenet_text_file = os.path.join(root, "imagenet", "classnames.txt")
        imagenet_classnames = ImageNet.read_classnames(imagenet_text_file)

        self.full_folders = sorted(imagenet_classnames.keys())
        self.full_classnames = [
            imagenet_classnames[folder] for folder in self.full_folders
        ]

        # 3. Data labels are still 0..199, following TaskRes filtered evaluation.
        data = self.read_data(ar_classnames)

        super().__init__(train_x=data, test=data)

        # 4. Force model construction to use 1000 ImageNet classnames.
        self._classnames = self.full_classnames
        self._lab2cname = {
            i: cname for i, cname in enumerate(self.full_classnames)
        }
        self._num_classes = len(self.full_classnames)

    def read_data(self, classnames):
        folders = listdir_nohidden(self.image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]

        items = []

        for label, folder in enumerate(folders):
            classname = classnames[folder]
            imnames = listdir_nohidden(os.path.join(self.image_dir, folder))

            for imname in imnames:
                impath = os.path.join(self.image_dir, folder, imname)
                item = Datum(
                    impath=impath,
                    label=label,          # 0..199
                    classname=classname,  # A/R class name
                )
                items.append(item)

        return items