import mmcv
from torch.utils.data import Dataset
from loguru import logger
class DemoDataset(Dataset):
    def __init__(self, ann_file_path, frame_num=8):
        """
        Args:
            ann_file_path (str): Path to the annotation file.
            frame_num (int): Number of frames to consider.
        """
        self.ann_file_path = ann_file_path
        self.frame_num = frame_num
        self.load_clips()
    
    def load_clips(self):
        data = mmcv.load(self.ann_file_path)
        data_infos = list(sorted(data["infos"], key=lambda x: x["timestamp"]))
        self.metadata = data["metadata"]
        self.clip_infos = self.build_clips(data_infos, data["scene_tokens"])
        logger.info(f"Loaded {len(self.clip_infos)} clips from {self.ann_file_path}")

    def build_clips(self, data_infos, scene_tokens):
        self.token_data_dict = {item['token']: item for item in data_infos}
        all_clips = []
        for sid, scene in enumerate(scene_tokens):
            first_frames = range(0, len(scene) - self.frame_num, self.frame_num)
            for start in first_frames:
                clip = [self.token_data_dict[scene[i]] for i in range(start, start + self.frame_num)]
                all_clips.append(clip)
        return all_clips

    def __len__(self):
        return len(self.clip_infos)

    def __getitem__(self, idx):
        return self.clip_infos[idx]
    

if __name__ == "__main__":
    dataset = DemoDataset("data/nuscenes/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl", frame_num=8)
    print(f"Total clips: {len(dataset)}")