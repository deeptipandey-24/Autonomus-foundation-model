import os
import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.nuscenes import NuScenes

dataroot = os.path.abspath('./data/nuscenes')

class NuScenesNoMap(NuScenes):
    def __make_reverse_index__(self, verbose):
        token2ind = {}
        for table in self.table_names:
            if table == 'map':
                continue
            token2ind[table] = {}
            for ind, member in enumerate(getattr(self, table)):
                token2ind[table][member['token']] = ind
        self.token2ind = token2ind

nusc = NuScenesNoMap(version='v1.0-mini', dataroot=dataroot, verbose=True)

from collections import defaultdict
sample_to_data = defaultdict(list)
for sd in nusc.sample_data:
    sample_to_data[sd['sample_token']].append(sd)

first_sample = nusc.sample[0]
sample_token = first_sample['token']
print(f"Sample token: {sample_token}")

related = sample_to_data[sample_token]
print(f"\nFound {len(related)} sample_data records for this sample")

cam_sd = None
for sd in related:
    if 'CAM_FRONT' in sd['filename'] and sd['is_key_frame']:
        cam_sd = sd
        break

if cam_sd is None:
    for sd in related:
        if 'CAM_FRONT' in sd['filename']:
            cam_sd = sd
            break

if cam_sd:
    img_path = os.path.join(dataroot, cam_sd['filename'])
    print(f"\nImage path: {img_path}")
    print(f"File exists: {os.path.exists(img_path)}")

    img = Image.open(img_path)
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.title(f"CAM_FRONT — sample {sample_token[:8]}...")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("Image rendered successfully")
else:
    print(" No CAM_FRONT found. Available filenames:")
    for sd in related[:10]:
        print(" ", sd['filename'])
