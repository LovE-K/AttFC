# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from PIL import Image
from torch.utils.data import Dataset


def opencv_rgb_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageDataset(Dataset):
    def __init__(self, root, anno_file, img_count, transform, image_loader=opencv_rgb_loader):
        """
        Dataset for training
        :param root: root directory of the dataset
        :param anno_file: annotation json file
        """
        super(ImageDataset, self).__init__()

        self.root = root
        self.transform = transform
        self.image_loader = image_loader

        cache_file = os.path.join(root, anno_file)
        print('use anno file: {}'.format(cache_file))

        sequence_list = []
        sub_seq_list = []
        data_type = root.split('\\')[-1]
        if data_type.startswith("WF"):
            id_type = '0_0_0000000'
            with open(cache_file, 'r') as f:
                for line_number, line in enumerate(f, start=1):
                    if line.split('\\')[1] == id_type:
                        sub_seq_list.append(line)
                    else:
                        sequence_list.append(sub_seq_list)
                        sub_seq_list = [line]
                        id_type = line.split('\\')[1]
                    if line_number == img_count:
                        sequence_list.append(sub_seq_list)
        else:
            with open(cache_file, 'r') as f:
                sub_count = 0
                for line_number, line in enumerate(f, start=1):
                    if int(line.split('\\')[1]) == sub_count:
                        sub_seq_list.append(line)
                    else:
                        sequence_list.append(sub_seq_list)
                        sub_seq_list = [line]
                        sub_count += 1
                    if line_number == img_count:
                        sequence_list.append(sub_seq_list)

        print('has {} ids'.format(len(sequence_list)))
        self.sequence_list = sequence_list

        # build index to seq id list
        item_list = []
        for seq_id, seq in enumerate(self.sequence_list):
            for frame_id in range(len(seq)):
                item_list.append((seq_id, frame_id))
        self.item_list = item_list
        print('has {} images'.format(len(item_list)))

    def __len__(self):
        return self.get_num_sequences()

    def __getitem__(self, index):
        """This function won't be used. Use get_frames instead."""
        return None

    def get_num_images(self):
        return len(self.item_list)

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        heads = set([p.split('\\')[0] for p in self.sequence_list[seq_id]])
        if len(heads) == 1:
            dataset_name = list(heads)[0]
        else:
            raise NotImplementedError
        return {'seq_len': len(self.sequence_list[seq_id]),
                'set_id': seq_id,
                'dataset': dataset_name}

    def _get_frame(self, sequence, frame_id):
        frame_path = os.path.join(self.root, sequence[frame_id])
        img = self.image_loader(frame_path.strip())
        sample = self.transform(img)
        return sample

    def get_frames(self, seq_id, frame_ids):
        sequence = self.sequence_list[seq_id]
        frame_list = [self._get_frame(sequence, f) for f in frame_ids]
        return frame_list
