# SuperDistillation
SuperDistillation
## Install
- Depth Anything V3
```
git submodule add https://github.com/ByteDance-Seed/Depth-Anything-3.git super_distillation/sub_modules/Depth-Anything-3
git submodule update --init --recursive
```

```
pip install xformers torch\>=2 torchvision
pip install -e super_distillation/sub_modules/Depth-Anything-3 # Basic
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head
pip install -e ".[app]" # Gradio, python>=3.10
pip install -e ".[all]" # ALL
```

- SparseBEV
```
<!-- openmim -->
pip install openmim #[y]

<!-- mmcv -->
cd super_distillation/sub_modules/mmcv-1.6.0
MMCV_WITH_OPS=1 pip install -e .

<!-- mmdet -->
pip install mmdet==2.28.2

<!-- mmsegmentation -->
pip install https://github.com/open-mmlab/mmsegmentation/archive/refs/tags/v0.30.0.zip

<!-- mmdet3d -->
cd super_distillation/sub_modules/mmdetection3d-1.0.0rc6
pip install -v . --no-build-isolation
```

- GaussianFormer