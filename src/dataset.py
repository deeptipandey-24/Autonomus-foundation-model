import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


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
        self._token2ind = token2ind


class NuScenesMultimodalDataset(Dataset):
    N_POINTS = 4096
    IMG_SIZE  = 224

    def __init__(self, dataroot, version='v1.0-mini', split='train'):
        self.dataroot = dataroot
        self.nusc = NuScenesNoMap(version=version, dataroot=dataroot, verbose=False)

        self._sd_index = defaultdict(dict)
        for sd in self.nusc.sample_data:
            parts = sd['filename'].replace('\\', '/').split('/')
            if len(parts) >= 2:
                channel = parts[1]
                existing = self._sd_index[sd['sample_token']].get(channel)
                if existing is None or (sd['is_key_frame'] and not existing['is_key_frame']):
                    self._sd_index[sd['sample_token']][channel] = sd

        n_scenes  = len(self.nusc.scene)
        split_idx = max(1, int(n_scenes * 0.8))
        scenes = self.nusc.scene[:split_idx] if split == 'train' else self.nusc.scene[split_idx:]

        self.samples = []
        for scene in scenes:
            token = scene['first_sample_token']
            while token:
                sample = self.nusc.get('sample', token)
                ch = self._sd_index.get(token, {})
                if 'CAM_FRONT' in ch and 'LIDAR_TOP' in ch:
                    self.samples.append(token)
                token = sample['next']

        print(f'[Dataset] split={split} | {len(self.samples)} samples from {len(scenes)} scenes')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token    = self.samples[idx]
        sample   = self.nusc.get('sample', token)
        channels = self._sd_index[token]
        camera   = self._load_camera(channels['CAM_FRONT'])
        lidar    = self._load_lidar(channels['LIDAR_TOP'])
        ego, can = self._load_ego(channels['LIDAR_TOP'], sample)
        return {'camera': camera, 'lidar': lidar, 'ego': ego, 'can': can, 'token': token}

    def _load_camera(self, sd):
        img = Image.open(os.path.join(self.dataroot, sd['filename'])).convert('RGB')
        img = img.resize((self.IMG_SIZE, self.IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_lidar(self, sd):
        pts = np.fromfile(os.path.join(self.dataroot, sd['filename']), dtype=np.float32).reshape(-1, 5)
        pts = self._fps(pts[:, :4], self.N_POINTS)
        return torch.from_numpy(pts)

    @staticmethod
    def _fps(pts, k):
        n = len(pts)
        if n <= k:
            return np.concatenate([pts, np.zeros((k-n, pts.shape[1]), dtype=np.float32)])
        chosen = np.zeros(k, dtype=np.int32)
        dist   = np.full(n, np.inf, dtype=np.float32)
        chosen[0] = np.random.randint(n)
        for i in range(1, k):
            d = np.sum((pts[:, :3] - pts[chosen[i-1], :3])**2, axis=1)
            dist = np.minimum(dist, d)
            chosen[i] = np.argmax(dist)
        return pts[chosen]

    def _load_ego(self, lidar_sd, sample):
        ep  = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        t   = np.array(ep['translation'], dtype=np.float32)
        rpy = np.array(Quaternion(ep['rotation']).yaw_pitch_roll[::-1], dtype=np.float32)
        ego = torch.from_numpy(np.concatenate([t, rpy]))
        vel = np.zeros(6, dtype=np.float32)
        if sample['next']:
            nc = self._sd_index.get(sample['next'], {})
            if 'LIDAR_TOP' in nc:
                sd2  = nc['LIDAR_TOP']
                ep2  = self.nusc.get('ego_pose', sd2['ego_pose_token'])
                t2   = np.array(ep2['translation'], dtype=np.float32)
                rpy2 = np.array(Quaternion(ep2['rotation']).yaw_pitch_roll[::-1], dtype=np.float32)
                dt   = max((sd2['timestamp'] - lidar_sd['timestamp']) / 1e6, 1e-6)
                vel  = np.concatenate([t2-t, rpy2-rpy]) / dt
        return ego, torch.from_numpy(vel)

if __name__ == '__main__':
    dataroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'nuscenes'))
    ds = NuScenesMultimodalDataset(dataroot)
    item = ds[0]
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: {v.shape}')
        else:
            print(f'  {k}: {v}')