from typing import List, Tuple, Union, Literal
from pydantic import BaseModel, field_validator, ValidationError


class ArmGripperAction(BaseModel):
    arm_action: List[float]
    gripper_action: Union[List[float], int]
    
    @field_validator('arm_action')
    def check_arm(cls, v):
        if len(v) != 7:
            raise ValueError("Arm action must be 1x7 list")
        
        return v
    
    @field_validator('gripper_action')
    def check_gripper(cls, v):
        if isinstance(v, list):
            if len(v) != 2 and len(v)!=6:
                raise ValueError("Gripper action list must be 1x2(panda) or 1x9(robotiq)")
        elif isinstance(v, int):
            if v not in (-1, 1):
                raise ValueError("Gripper action num must be -1 or 1")
        else:
            raise ValueError("Gripper action type must be list or int")
        
        return v


class EEfAction(BaseModel):
    eef_position: List[float]
    eef_orientation: List[float]
    gripper_action: Union[List[float], int]
    
    @field_validator('eef_position')
    def check_position(cls, v):
        if len(v) != 3:
            raise ValueError("EEF position must be 1*3 list")
        
        return v
    
    @field_validator('eef_orientation')
    def check_orientation(cls, v):
        if len(v) != 4:
            raise ValueError("EEF orientation must be 1*4 list")
        
        return v
    
    @field_validator('gripper_action')
    def check_gripper(cls, v):
        if isinstance(v, list):
            if len(v) != 2 and len(v)!=6:
                raise ValueError("Gripper action list must be 1x2(panda) or 1x9(robotiq)")
        elif isinstance(v, int):
            if v not in (-1, 1):
                raise ValueError("Gripper action num must be -1 or 1")
        else:
            raise ValueError("Gripper action type must be list or int")
        
        return v
    

def validate_action(action) -> Tuple[Literal['joint', 'arm_gripper', 'eef'], Union[list, ArmGripperAction, EEfAction]]:
    try:
        if isinstance(action, list):
            if len(action) != 9 and len(action) != 13:
                raise ValueError("Joint action must be 1x9(panda) or 1x13(robotiq) list")
            return 'joint', action
        
        elif isinstance(action, dict):
            if all(k in action for k in ('arm_action', 'gripper_action')):
                arm_gripper_action = ArmGripperAction(**action)
                return 'arm_gripper', arm_gripper_action
            elif all(k in action for k in ('eef_position', 'eef_orientation', 'gripper_action')):
                eef_action = EEfAction(**action)
                return 'eef', eef_action
        
        raise ValueError("Invalid action structure")
    
    except (ValidationError, ValueError) as e:
        print("="*50,
              "\nAction supports following types:\n"
              "1. joint action (List): List[float] with length 1x9(panda) or 1x13(robotiq)\n"
              "2. arm_gripper_action (Dict):\n"
              "{'arm_action': 1x7 List[float],\n"
              "'gripper_action': 1x2(panda) or 1x9(robotiq) List[float] or int -1/1 (open/close)}\n"
              "3. eef_gripper_action (Dict):\n"
              "{'eef_position': 1x3 List[float],\n"
              "'eef_orientation' 1x4 List[float] (quaternion),\n"
              "'gripper_action': 1x2(panda) or 1x9(robotiq) List[float] or int -1/1 (open/close)}\n",
              "="*50)
        
        raise e
