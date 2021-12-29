import tensorflow_datasets as tfds


class WristXrayImages(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.1')
    RELEASE_NOTES = {
        '1.0.1': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="",
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(),
                'label': tfds.features.ClassLabel(names=['normal', 'fracture']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://data.mendeley.com/datasets/xbdsnzr8ct/1',
            citation='',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        #path = dl_manager.download_kaggle_data(competition_or_dataset='cjinny/mura-v11')
        path = dl_manager.download_and_extract("https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/xbdsnzr8ct-1.zip")

        return {
            'train': self._generate_examples(path=path / 'Wrist Fracture'),
        }

    def _generate_examples(self, path):
        """Yields examples."""

        # Read the input data out of the source files
        #root = "/".join(path.split("/")[:-2])  # ../.. to get to root dataloader folder
        normal_path = path / "Normal"
        fracture_path = path / "Fracture"

        for img_path in normal_path.glob('*.jpg'):
            # Yields (key, example)
            yield img_path.name, {
                'image': img_path,
                'label': 'normal',
            }

        for img_path in fracture_path.glob('*.jpg'):
            # Yields (key, example)
            yield img_path.name, {
                'image': img_path,
                'label': 'fracture',
            }
