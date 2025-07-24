import os
import lmdb
import pickle


class ReplayEpisodesAgent():
    """Genmanip demonstration replay agent
    
    Functionality:
    1. Loads pre-recorded action sequences from LMDB storage
    2. Supports multiple action space representations
    3. Manages per-environment episode tracking

    Parameter Specifications:
    :param dataset_path: (str) genmanip dataset path
    :param action_type: (str) format of return action
        - Options:
            "joint_action": Combined arm+gripper joint values
            "arm_gripper_action_1": Separate arm/gripper control
            "arm_gripper_action_2": Arm control + binary gripper
            "eef_action_1": End-effector pose + gripper values
            "eef_action_2": End-effector pose + binary gripper
    """

    def __init__(self, dataset_path, action_type):
        self.dataset_path = dataset_path
        self.action_type = action_type
        assert self.action_type in \
            ["joint_action", "arm_gripper_action_1", "arm_gripper_action_2", "eef_action_1", "eef_action_2"], \
            f"Invalid action self.action_type: {self.action_type}."

        self.action_manager = {}

    def get_next_action(self, obs):
        all_env_action = []

        for env_id, term in enumerate(obs):
            if not term:
                all_env_action.append([])
                continue

            if env_id not in self.action_manager:
                self.action_manager[env_id] = {}
            
            cur_episode_name = self.action_manager[env_id].get("cur_episode_name", None)
            if cur_episode_name is None or \
                cur_episode_name != term["franka_robot"]["metric"]["episode_name"]:
                self.load_next_episode_lmdb(env_id, term)

            action = self.read_next_action(env_id)
            all_env_action.append(action)
        
        return all_env_action
    
    def load_next_episode_lmdb(self, env_id, obs_info):
        task_name = obs_info["franka_robot"]["metric"]["task_name"]
        episode_name = obs_info["franka_robot"]["metric"]["episode_name"]
        lmdb_path = os.path.join(self.dataset_path, task_name, episode_name, "lmdb")
        lmdb_env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with lmdb_env.begin(write=False) as txn:
            self.action_manager[env_id]["arm_action"] = pickle.loads(txn.get(b"arm_action"))
            self.action_manager[env_id]["gripper_action"] = pickle.loads(txn.get(b"gripper_action"))
            self.action_manager[env_id]["gripper_close"] = pickle.loads(txn.get(b"gripper_close"))
            self.action_manager[env_id]["ee_pose_action"] = pickle.loads(txn.get(b"ee_pose_action"))

        self.action_manager[env_id]["cur_episode_name"] = episode_name
        self.action_manager[env_id]["step_idx"] = 0

        print("current_env_id: {} | episode_name: {}".format(env_id, episode_name))

    def read_next_action(self, env_id):
        arm_action = self.action_manager[env_id]["arm_action"]
        gripper_action = self.action_manager[env_id]["gripper_action"]
        gripper_close = self.action_manager[env_id]["gripper_close"]
        ee_pose_action = self.action_manager[env_id]["ee_pose_action"]
        step_idx = min(self.action_manager[env_id]["step_idx"], len(self.action_manager[env_id]["arm_action"]) - 1)
        self.action_manager[env_id]["step_idx"] = step_idx + 1

        if self.action_type=="joint_action":
            action = list(arm_action[step_idx]) + list(gripper_action[step_idx])

        if self.action_type=="arm_gripper_action_1":
            action = {
                "arm_action": list(arm_action[step_idx]),
                "gripper_action": list(gripper_action[step_idx])
            }

        if self.action_type=="arm_gripper_action_2":
            action = {
                "arm_action": list(arm_action[step_idx]),
                "gripper_action": gripper_close[step_idx]
            }
        
        if self.action_type=="eef_action_1":
            action = {
                "eef_position": list(ee_pose_action[step_idx][0]),
                "eef_orientation": list(ee_pose_action[step_idx][1]),
                "gripper_action": list(gripper_action[step_idx])
            }

        if self.action_type=="eef_action_2":
            action = {
                "eef_position": list(ee_pose_action[step_idx][0]),
                "eef_orientation": list(ee_pose_action[step_idx][1]),
                "gripper_action": gripper_close[step_idx]
            }

        return action