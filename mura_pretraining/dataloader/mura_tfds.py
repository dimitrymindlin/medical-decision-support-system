"""kaggleChestXrayImages dataset."""

import tensorflow_datasets as tfds
import os
import pandas as pd


class MuraImages(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.3')
    RELEASE_NOTES = {
        '1.0.3': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="",
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'name': tfds.features.Text(),  # patient id
                'image': tfds.features.Image(),
                'image_num': tfds.features.Text(),  # image number of a patient
                'label': tfds.features.ClassLabel(names=['negative', 'positive']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://www.kaggle.com/cjinny/mura-v11',
            citation='',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_kaggle_data(competition_or_dataset='cjinny/mura-v11')

        return {
            'train': self._generate_examples(os.path.join(path, 'MURA-v1.1/train')),
            'test': self._generate_examples(os.path.join(path, 'MURA-v1.1/valid'))
        }

    def _generate_examples(self, path):
        """Yields examples."""
        body_parts = ["XR_WRIST"]

        # Read the input data out of the source files
        root = "/".join(path.split("/")[:-2])  # ../.. to get to root dataloader folder
        if "train" in path:
            csv_path = root + "/MURA-v1.1/train_image_paths.csv"
        else:
            csv_path = root + "/MURA-v1.1/valid_image_paths.csv"

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            for row in d:
                img_path = str(row, encoding='utf-8').strip()
                if img_path.split('/')[2] in body_parts:
                    # And yield (key, feature_dict)
                    patient_id = img_path.split("/")[-3].replace("patient", "")
                    yield img_path, {
                        'name': patient_id,  # patient id ## ex. 0692
                        'image': root + "/" + img_path,
                        'image_num': img_path.split("/")[-1].split(".")[0].replace("image", ""),
                        # image count for patient
                        'label': img_path.split('_')[-1].split('/')[0],
                    }
