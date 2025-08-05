from typing import List, Literal, Tuple, Union

from pydantic import BaseModel, ValidationError, field_validator


class SingleArmGripperAction(BaseModel):
    arm_action: List[float]
    gripper_action: Union[List[float], int]

    @field_validator('arm_action')
    def check_arm(cls, v):
        if len(v) != 6:
            raise ValueError('[Aloha split] Single arm action must be 1x6 list')

        return v

    @field_validator('gripper_action')
    def check_gripper(cls, v):
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError('[Aloha split] Single gripper action list must be 1x2')
        elif isinstance(v, int):
            if v not in (-1, 1):
                raise ValueError(
                    '[Aloha split] Single gripper action num must be -1(open) or 1(close)'
                )
        else:
            raise ValueError('[Aloha split] Single gripper action type must be list or int')

        return v


class ArmGripperAction(BaseModel):
    left_arm_gripper_action: SingleArmGripperAction
    right_arm_gripper_action: SingleArmGripperAction


class SingleEEfAction(BaseModel):
    eef_position: List[float]
    eef_orientation: List[float]
    gripper_action: Union[List[float], int]

    @field_validator('eef_position')
    def check_position(cls, v):
        if len(v) != 3:
            raise ValueError('[Aloha split] EEf position must be 1*3 list')

        return v

    @field_validator('eef_orientation')
    def check_orientation(cls, v):
        if len(v) != 4:
            raise ValueError(
                '[Aloha split] EEf orientation must be 1*4 list (quaternion: (w, x, y, z))'
            )

        return v

    @field_validator('gripper_action')
    def check_gripper(cls, v):
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError('[Aloha split] Single gripper action list must be 1x2')
        elif isinstance(v, int):
            if v not in (-1, 1):
                raise ValueError(
                    '[Aloha split] Single gripper action num must be -1(open) or 1(close)'
                )
        else:
            raise ValueError('[Aloha split] Single gripper action type must be list or int')

        return v


class EEfAction(BaseModel):
    left_eef_action: SingleEEfAction
    right_eef_action: SingleEEfAction


def validate_action(
    action,
) -> Tuple[Literal['arm_gripper', 'eef'], Union[ArmGripperAction, EEfAction]]:
    validata_keys_1 = (
        'left_arm_action',
        'left_gripper_action',
        'right_arm_action',
        'right_gripper_action',
    )
    validata_keys_2 = (
        'left_eef_position',
        'left_eef_orientation',
        'left_gripper_action',
        'right_eef_position',
        'right_eef_orientation',
        'right_gripper_action',
    )

    try:
        if not isinstance(action, dict):
            raise ValueError('[Aloha split] Action type must be dict')

        if all(k in action for k in validata_keys_1):
            arm_gripper_action = ArmGripperAction(
                left_arm_gripper_action=SingleArmGripperAction(
                    arm_action=action['left_arm_action'],
                    gripper_action=action['left_gripper_action'],
                ),
                right_arm_gripper_action=SingleArmGripperAction(
                    arm_action=action['right_arm_action'],
                    gripper_action=action['right_gripper_action'],
                ),
            )
            return 'arm_gripper', arm_gripper_action

        elif all(k in action for k in validata_keys_2):
            eef_action = EEfAction(
                left_eef_action=SingleEEfAction(
                    eef_position=action['left_eef_position'],
                    eef_orientation=action['left_eef_orientation'],
                    gripper_action=action['left_gripper_action'],
                ),
                right_eef_action=SingleEEfAction(
                    eef_position=action['right_eef_position'],
                    eef_orientation=action['right_eef_orientation'],
                    gripper_action=action['right_gripper_action'],
                ),
            )
            return 'eef', eef_action

        raise ValueError('[Aloha split] Invalid action structure')

    except (ValidationError, ValueError) as e:
        print(
            '=' * 50,
            '\n[Aloha split] Action supports following types:\n'
            '1. arm_gripper_action (Dict):\n'
            "{'left_arm_action': 1x6 List[float],\n"
            "'left_gripper_action': 1x2 List[float] or int -1/1 (open/close),\n"
            "'right_arm_action': 1x6 List[float],\n"
            "'right_gripper_action': 1x2 List[float] or int -1/1 (open/close)}\n"
            '2. eef_gripper_action (Dict):\n'
            "{'left_eef_position': 1x3 List[float],\n"
            "'left_eef_orientation' 1x4 List[float] (quaternion: (w, x, y, z)),\n"
            "'left_gripper_action': 1x2 List[float] or int -1/1 (open/close),\n",
            "'right_eef_position': 1x3 List[float],\n"
            "'right_eef_orientation' 1x4 List[float] (quaternion: (w, x, y, z)),\n"
            "'right_gripper_action': 1x2 List[float] or int -1/1 (open/close)}\n",
            '=' * 50,
        )

        raise e
