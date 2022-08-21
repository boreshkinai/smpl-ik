from smplik.geometry.skeleton import Skeleton
from typing import Optional, List


class DataComponents:
    def __init__(self, skeleton: Skeleton, contact_joints: Optional[List] = None):
        self.skeleton = skeleton
        if contact_joints is None:
            contact_joints = []
        self.contact_joints = [j for j in contact_joints if j in skeleton.all_joints]
