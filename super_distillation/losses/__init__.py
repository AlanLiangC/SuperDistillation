from mmengine.registry import Registry
SD_LOSS = Registry('sd_loss')

from .multi_loss import MultiLoss
from .occupancy_loss import OccupancyLoss
from .bce_loss import BinaryCrossEntropyLoss, PixelDistributionLoss
