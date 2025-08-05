from .base import CuroboPlanner


class CuroboPiperPlanner(CuroboPlanner):
    def __init__(self, robot_cfg, robot_prim_path, type='default'):
        super().__init__(robot_cfg, robot_prim_path)
        self.type = type
        if self.type == 'default':
            self.ordered_js_names = [
                'joint1',
                'joint2',
                'joint3',
                'joint4',
                'joint5',
                'joint6',
            ]
        elif self.type == 'left':
            self.ordered_js_names = [
                'fl_joint1',
                'fl_joint2',
                'fl_joint3',
                'fl_joint4',
                'fl_joint5',
                'fl_joint6',
            ]
        elif self.type == 'right':
            self.ordered_js_names = [
                'fr_joint1',
                'fr_joint2',
                'fr_joint3',
                'fr_joint4',
                'fr_joint5',
                'fr_joint6',
            ]
        else:
            raise ValueError(f'Invalid type: {self.type}')
        self.raw_js_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
        ]
