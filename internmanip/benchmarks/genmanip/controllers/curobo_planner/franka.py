from .base import CuroboPlanner


class CuroboFrankaPlanner(CuroboPlanner):
    def __init__(self, robot_cfg, robot_prim_path):
        super().__init__(robot_cfg, robot_prim_path)
        self.ordered_js_names = [
            'panda_joint1',
            'panda_joint2',
            'panda_joint3',
            'panda_joint4',
            'panda_joint5',
            'panda_joint6',
            'panda_joint7',
        ]
        self.dof_len = 7
