import copy

import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions import Distribution
from rlkit.envs.wrappers import NormalizedBoxEnv
import metaworld.envs.mujoco.env_dict as _env_dict
from gym.wrappers.time_limit import TimeLimit




criteria_code_dict = {
    "button-press-topdown-v2": "success = obj_to_target <= 0.024",
    "basketball-v2": """success = float(obj_to_target <= self.TARGET_RADIUS)
                near_object = float(tcp_to_obj <= 0.05)
                grasp_success = float((tcp_open > 0) and (obj[2] - 0.03 > self.obj_init_pos[2]))""",
    "push-back-v2":  """success = float(target_to_obj <= 0.07)
                near_object = float(tcp_to_obj <= 0.03)
                grasp_success = float(self.touching_object  and (tcp_opened > 0) and (obj[2] - 0.02 > self.obj_init_pos[2]))""",
    "dial-turn-v2": "success = float(target_to_obj <= self.TARGET_RADIUS) "
                    "near_object = float(tcp_to_obj <= 0.01)",
    "hand-insert-v2": """success =float(obj_to_target <= 0.05)
            near_object = float(tcp_to_obj <= 0.03),
            grasp_success = float( self.touching_main_object and (tcp_open > 0) and (obj[2] - 0.02 > self.obj_init_pos[2]))""",
    "pick-out-of-hole-v2": """success = float(obj_to_target <= 0.07)
            near_object = float(tcp_to_obj <= 0.03)
            grasp_success = float(grasp_success)""",
    "hammer-v2": "grasp_success = reward_grab >= 0.5",
    "coffee-pull-v2": """success = float(obj_to_target <= 0.07)
            near_object = float(tcp_to_obj <= 0.03)
            grasp_success = float(self.touching_object and (tcp_open > 0))""",
    "reach-v2": "success = float(reach_dist <= 0.05)",
    "peg-unplug-side-v2":"""success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)""",
    "soccer-v2": """success = float(target_to_obj <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(
            self.touching_object
            and (tcp_opened > 0)
            and (obj[2] - 0.02 > self.obj_init_pos[2])
        )""",
    "coffee-push-v2": """success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(self.touching_object and (tcp_open > 0))""",
    "peg-insert-side-v2": """grasp_success = float(
        tcp_to_obj < 0.02
        and (tcp_open > 0)
        and (obj[2] - 0.01 > self.obj_init_pos[2]))
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        """,
     "sweep-into-v2": """grasp_success = float(self.touching_main_object and (tcp_opened > 0))
        success = float(target_to_obj <= 0.05)
        near_object= float(tcp_to_obj <= 0.03),
    """
}
task_description_dict = {
"button-press-topdown-v2": "Press a button from the top",
"basketball-v2": "Dunk the basketball into the basket",
"push-back-v2": "first grab the target and then place it at the target point.",
"dial-turn-v2": "Rotate a dial 180 degrees.",
"hand-insert-v2": "Insert the gripper into a hole.",
"pick-out-of-hole-v2": "Pick up a puck from a hole.",
"hammer-v2": "Hammer a screw on the wall.",
"coffee-pull-v2": "Pull a mug from a coffee machine",
"reach-v2": "reach a goal position.",
"peg-unplug-side-v2":"Unplug a peg sideways",
"soccer-v2": "Kick a soccer into the goal.",
"coffee-push-v2": "Push the coffee cup to the target point.",
"peg-insert-side-v2":"Picking up a stick and inserting it into a square hole in the wall",
"sweep-into-v2": "Sweep a puck into a hole"}
input_dict = {
    "button-press-topdown-v2": """List = ["tcp_center", "action", "_obj_to_target_init","obs",  "init_tcp","_target_pos"]""",
    "basketball-v2": """["init_tcp", "obj_init_pos", "TARGET_RADIUS","_target_pos", "tcp_center", "left_pad",
                    "right_pad", "obs", "action"]""",
    "push-back-v2": """List = ["init_tcp", "obj_init_pos",  "TARGET_RADIUS", "OBJ_RADIUS" ,
                    "init_right_pad", "init_left_pad", "_target_pos", "tcp_center", "left_pad", "right_pad", "obs", "action"]""",
    "dial-turn-v2": """List = ["TARGET_RADIUS", "init_tcp", "pos_objects",
                    "_target_pos", "tcp_center","obj", "obs","action"]""",
    "hand-insert-v2": """List = ["TARGET_RADIUS", "init_tcp", "obj_init_pos",
                    "_target_pos", "tcp_center","left_pad","right_pad", "obs", "action"]""",
    "pick-out-of-hole-v2": """List = ["init_tcp", "obj_init_pos", 
                    "_target_pos", "tcp_center",
                    "left_pad",
                    "right_pad", "obs",
                    "action", "TARGET_RADIUS"]""",
    "hammer-v2": """List = ["init_tcp", "obj_init_pos",
                    "_target_pos", "tcp_center",
                    "left_pad", "NailSlideJoint_qpos", "HAMMER_HANDLE_LENGTH",
                    "right_pad", "obs", "action]""",
    "coffee-pull-v2": """dict = ["init_tcp", "obj_init_pos",
                "_target_pos", "tcp_center",
                "left_pad",
                "right_pad", "obs",
                "action"]""",
    "reach-v2": """List = ["hand_init_pos", "init_right_pad", "init_left_pad", "obs", "action"]""",
    "peg-unplug-side-v2":"""List = ["init_tcp", "obj_init_pos",
                "_target_pos", "tcp_center",
                "left_pad",
                "right_pad", "obs",
                "action"]""",
    "soccer-v2": """list = ["tcp_center",  "_target_pos", "obj_init_pos","action", "obs",
                "TARGET_RADIUS","left_pad", "right_pad","init_tcp",
                "OBJ_RADIUS","init_left_pad", "init_right_pad"]""",
    "coffee-push-v2": """dict = ["init_tcp", "obj_init_pos", "obj_init_angle",
            "hand_init_pos",
            "init_right_pad", "init_left_pad",
            "_target_pos", "tcp_center",
            "left_pad",
            "right_pad", "obs",
            "action"]""",
    "sweep-into-v2": """list = ["init_tcp", "obj_init_pos",
            "obj_init_angle",
            "init_right_pad", "init_left_pad",
            "_target_pos", "OBJ_RADIUS", "tcp_center",
            "objHeight",
            "maxPushDist",
            "left_pad", "left_pad", "right_pad", "obs","action"}""",
    "peg-insert-side-v2": """list = ["obj_init_pos", "peg_head_pos_init",
            "_target_pos",
            "tcp_center", "obj_head",
            "obj", "TARGET_RADIUS", "brc_col_box_1",
            "tlc_col_box_1",
            "brc_col_box_2",
            "tlc_col_box_2" , "left_pad", "right_pad", "obs",
            "action": action}"""

}
parents_function_dict = {
"button-press-topdown-v2": """def _gripper_caging_reward():
    pass""",
    "basketball-v2": """def _gripper_caging_reward(
        leftpad,
        rightpad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj_pos,
        obj_radius,
        pad_success_thresh,
        object_reach_radius,
        xz_thresh,
        desired_gripper_effort=1.0,
        high_density=False,
        medium_density=False,
):
    '''Reward for agent grasping obj.
    Args:
        action(np.ndarray): (4,) array representing the action
            delta(x), delta(y), delta(z), gripper_effort
        obj_pos(np.ndarray): (3,) array representing the obj x,y,z
        obj_radius(float):radius of object's bounding sphere
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        object_reach_radius(float): successful distance of gripper center
            to the object.
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.
    '''
    if high_density and medium_density:
        raise ValueError("Can only be either high_density or medium_density")
    # MARK: Left-right gripper information for caging reward----------------
    left_pad = leftpad
    right_pad = rightpad

    # get current positions of left and right pads (Y axis)
    pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
    # compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
    # compare *current* pad positions with *initial* obj position (Y axis)
    pad_to_objinit_lr = np.abs(pad_y_lr - obj_init_pos[1])

    # Compute the left/right caging rewards. This is crucial for success,
    # yet counterintuitive mathematically because we invented it
    # accidentally.
    #
    # Before touching the object, pad_to_obj_lr ("x") is always separated
    # from caging_lr_margin ("the margin") by some small number,
    # pad_success_thresh.
    #
    # When far away from the object:
    #       x = margin + pad_success_thresh
    #       --> Thus x is outside the margin, yielding very small reward.
    #           Here, any variation in the reward is due to the fact that
    #           the margin itself is shifting.
    # When near the object (within pad_success_thresh):
    #       x = pad_success_thresh - margin
    #       --> Thus x is well within the margin. As long as x > obj_radius,
    #           it will also be within the bounds, yielding maximum reward.
    #           Here, any variation in the reward is due to the gripper
    #           moving *too close* to the object (i.e, blowing past the
    #           obj_radius bound).
    #
    # Therefore, before touching the object, this is very nearly a binary
    # reward -- if the gripper is between obj_radius and pad_success_thresh,
    # it gets maximum reward. Otherwise, the reward very quickly falls off.
    #
    # After grasping the object and moving it away from initial position,
    # x remains (mostly) constant while the margin grows considerably. This
    # penalizes the agent if it moves *back* toward obj_init_pos, but
    # offers no encouragement for leaving that position in the first place.
    # That part is left to the reward functions of individual environments.
    caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
    caging_lr = [
        reward_utils.tolerance(
            pad_to_obj_lr[i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[i],  # "margin" in the description above
            sigmoid="long_tail",
        )
        for i in range(2)
    ]
    caging_y = reward_utils.hamacher_product(*caging_lr)

    # MARK: X-Z gripper information for caging reward-----------------------
    tcp = tcp_center
    xz = [0, 2]

    # Compared to the caging_y reward, caging_xz is simple. The margin is
    # constant (something in the 0.3 to 0.5 range) and x shrinks as the
    # gripper moves towards the object. After picking up the object, the
    # reward is maximized and changes very little
    caging_xz_margin = np.linalg.norm(obj_init_pos[xz] - init_tcp[xz])
    caging_xz_margin -= xz_thresh
    caging_xz = reward_utils.tolerance(
        np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
        bounds=(0, xz_thresh),
        margin=caging_xz_margin,  # "margin" in the description above
        sigmoid="long_tail",
    )

    # MARK: Closed-extent gripper information for caging reward-------------
    gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
    )

    # MARK: Combine components----------------------------------------------
    caging = reward_utils.hamacher_product(caging_y, caging_xz)
    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

    if high_density:
        return  caging_and_gripping, caging
    if medium_density:
        tcp = tcp_center
        tcp_to_obj = np.linalg.norm(obj_pos - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - init_tcp)
        # Compute reach reward
        # - We subtract object_reach_radius from the margin so that the
        #   reward always starts with a value of 0.1
        reach_margin = abs(tcp_to_obj_init - object_reach_radius)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=reach_margin,
            sigmoid="long_tail",
        )
        return caging_and_gripping, reach""",
"push-back-v2": """
def _gripper_caging_reward(tcp_center, left_pad, right_pad,init_left_pad, init_right_pad, obj_init_pos, init_tcp, action, obj_position, obj_radius):
    pad_success_margin = 0.05
    grip_success_margin = obj_radius + 0.003
    x_z_success_margin = 0.01
    tcp = tcp_center
    delta_object_y_left_pad = left_pad[1] - obj_position[1]
    delta_object_y_right_pad = obj_position[1] - right_pad[1]
    right_caging_margin = abs(
        abs(obj_position[1] - init_right_pad[1]) - pad_success_margin
    )
    left_caging_margin = abs(
        abs(obj_position[1] - init_left_pad[1]) - pad_success_margin
    )
    right_caging = reward_utils.tolerance(
        delta_object_y_right_pad,
        bounds=(obj_radius, pad_success_margin),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_caging = reward_utils.tolerance(
        delta_object_y_left_pad,
        bounds=(obj_radius, pad_success_margin),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )
    right_gripping = reward_utils.tolerance(
        delta_object_y_right_pad,
        bounds=(obj_radius, grip_success_margin),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_gripping = reward_utils.tolerance(
        delta_object_y_left_pad,
        bounds=(obj_radius, grip_success_margin),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )
    assert right_caging >= 0 and right_caging <= 1 and left_caging >= 0 and left_caging <= 1

    y_caging = reward_utils.hamacher_product(right_caging, left_caging)
    y_gripping = reward_utils.hamacher_product(right_gripping, left_gripping)

    assert y_caging >= 0 and y_caging <= 1

    tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
    obj_position_x_z = np.copy(obj_position) + np.array(
        [0.0, -obj_position[1], 0.0]
    )
    tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
    init_obj_x_z = obj_init_pos + np.array([0.0, -obj_init_pos[1], 0.0])
    init_tcp_x_z = init_tcp + np.array([0.0, -init_tcp[1], 0.0])

    tcp_obj_x_z_margin = (
        np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
    )
    x_z_caging = reward_utils.tolerance(
        tcp_obj_norm_x_z,
        bounds=(0, x_z_success_margin),
        margin=tcp_obj_x_z_margin,
        sigmoid="long_tail",
    )

    assert right_caging >= 0 and right_caging <= 1
    gripper_closed = min(max(0, action[-1]), 1)
    assert gripper_closed >= 0 and gripper_closed <= 1
    caging = reward_utils.hamacher_product(y_caging, x_z_caging)
    assert caging >= 0 and caging <= 1

    if caging > 0.95:
        gripping = y_gripping
    else:
        gripping = 0.0
    assert gripping >= 0 and gripping <= 1

    caging_and_gripping = (caging + gripping) / 2
    assert caging_and_gripping >= 0 and caging_and_gripping <= 1

    return caging_and_gripping""",
"dial-turn-v2": """def _gripper_caging_reward():
    pass""",
"hand-insert-v2": """def _gripper_caging_reward(
        leftpad,
        rightpad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj_pos,
        obj_radius,
        pad_success_thresh,
        object_reach_radius,
        xz_thresh,
        desired_gripper_effort=1.0,
        high_density=False,
        medium_density=False,
):
    '''Reward for agent grasping obj.
    Args:
        action(np.ndarray): (4,) array representing the action
            delta(x), delta(y), delta(z), gripper_effort
        obj_pos(np.ndarray): (3,) array representing the obj x,y,z
        obj_radius(float):radius of object's bounding sphere
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        object_reach_radius(float): successful distance of gripper center
            to the object.
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.
    '''
    if high_density and medium_density:
        raise ValueError("Can only be either high_density or medium_density")
    # MARK: Left-right gripper information for caging reward----------------
    left_pad = leftpad
    right_pad = rightpad

    # get current positions of left and right pads (Y axis)
    pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
    # compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
    # compare *current* pad positions with *initial* obj position (Y axis)
    pad_to_objinit_lr = np.abs(pad_y_lr - obj_init_pos[1])

    # Compute the left/right caging rewards. This is crucial for success,
    # yet counterintuitive mathematically because we invented it
    # accidentally.
    #
    # Before touching the object, pad_to_obj_lr ("x") is always separated
    # from caging_lr_margin ("the margin") by some small number,
    # pad_success_thresh.
    #
    # When far away from the object:
    #       x = margin + pad_success_thresh
    #       --> Thus x is outside the margin, yielding very small reward.
    #           Here, any variation in the reward is due to the fact that
    #           the margin itself is shifting.
    # When near the object (within pad_success_thresh):
    #       x = pad_success_thresh - margin
    #       --> Thus x is well within the margin. As long as x > obj_radius,
    #           it will also be within the bounds, yielding maximum reward.
    #           Here, any variation in the reward is due to the gripper
    #           moving *too close* to the object (i.e, blowing past the
    #           obj_radius bound).
    #
    # Therefore, before touching the object, this is very nearly a binary
    # reward -- if the gripper is between obj_radius and pad_success_thresh,
    # it gets maximum reward. Otherwise, the reward very quickly falls off.
    #
    # After grasping the object and moving it away from initial position,
    # x remains (mostly) constant while the margin grows considerably. This
    # penalizes the agent if it moves *back* toward obj_init_pos, but
    # offers no encouragement for leaving that position in the first place.
    # That part is left to the reward functions of individual environments.
    caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
    caging_lr = [
        reward_utils.tolerance(
            pad_to_obj_lr[i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[i],  # "margin" in the description above
            sigmoid="long_tail",
        )
        for i in range(2)
    ]
    caging_y = reward_utils.hamacher_product(*caging_lr)

    # MARK: X-Z gripper information for caging reward-----------------------
    tcp = tcp_center
    xz = [0, 2]

    # Compared to the caging_y reward, caging_xz is simple. The margin is
    # constant (something in the 0.3 to 0.5 range) and x shrinks as the
    # gripper moves towards the object. After picking up the object, the
    # reward is maximized and changes very little
    caging_xz_margin = np.linalg.norm(obj_init_pos[xz] - init_tcp[xz])
    caging_xz_margin -= xz_thresh
    caging_xz = reward_utils.tolerance(
        np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
        bounds=(0, xz_thresh),
        margin=caging_xz_margin,  # "margin" in the description above
        sigmoid="long_tail",
    )

    # MARK: Closed-extent gripper information for caging reward-------------
    gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
    )

    # MARK: Combine components----------------------------------------------
    caging = reward_utils.hamacher_product(caging_y, caging_xz)
    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

    if high_density:
        return  caging_and_gripping, caging
    if medium_density:
        tcp = tcp_center
        tcp_to_obj = np.linalg.norm(obj_pos - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - init_tcp)
        # Compute reach reward
        # - We subtract object_reach_radius from the margin so that the
        #   reward always starts with a value of 0.1
        reach_margin = abs(tcp_to_obj_init - object_reach_radius)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=reach_margin,
            sigmoid="long_tail",
        )
        return caging_and_gripping, reach""",
"pick-out-of-hole-v2":'''
def _gripper_caging_reward(
        leftpad,
        rightpad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj_pos,
        obj_radius,
        pad_success_thresh,
        object_reach_radius,
        xz_thresh,
        desired_gripper_effort=1.0,
        high_density=False,
        medium_density=False,
):
    """Reward for agent grasping obj.
    Args:
        action(np.ndarray): (4,) array representing the action
            delta(x), delta(y), delta(z), gripper_effort
        obj_pos(np.ndarray): (3,) array representing the obj x,y,z
        obj_radius(float):radius of object's bounding sphere
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        object_reach_radius(float): successful distance of gripper center
            to the object.
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.
    """
    if high_density and medium_density:
        raise ValueError("Can only be either high_density or medium_density")
    # MARK: Left-right gripper information for caging reward----------------
    left_pad = leftpad
    right_pad = rightpad

    # get current positions of left and right pads (Y axis)
    pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
    # compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
    # compare *current* pad positions with *initial* obj position (Y axis)
    pad_to_objinit_lr = np.abs(pad_y_lr - obj_init_pos[1])

    # Compute the left/right caging rewards. This is crucial for success,
    # yet counterintuitive mathematically because we invented it
    # accidentally.
    #
    # Before touching the object, pad_to_obj_lr ("x") is always separated
    # from caging_lr_margin ("the margin") by some small number,
    # pad_success_thresh.
    #
    # When far away from the object:
    #       x = margin + pad_success_thresh
    #       --> Thus x is outside the margin, yielding very small reward.
    #           Here, any variation in the reward is due to the fact that
    #           the margin itself is shifting.
    # When near the object (within pad_success_thresh):
    #       x = pad_success_thresh - margin
    #       --> Thus x is well within the margin. As long as x > obj_radius,
    #           it will also be within the bounds, yielding maximum reward.
    #           Here, any variation in the reward is due to the gripper
    #           moving *too close* to the object (i.e, blowing past the
    #           obj_radius bound).
    #
    # Therefore, before touching the object, this is very nearly a binary
    # reward -- if the gripper is between obj_radius and pad_success_thresh,
    # it gets maximum reward. Otherwise, the reward very quickly falls off.
    #
    # After grasping the object and moving it away from initial position,
    # x remains (mostly) constant while the margin grows considerably. This
    # penalizes the agent if it moves *back* toward obj_init_pos, but
    # offers no encouragement for leaving that position in the first place.
    # That part is left to the reward functions of individual environments.
    caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
    caging_lr = [
        reward_utils.tolerance(
            pad_to_obj_lr[i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[i],  # "margin" in the description above
            sigmoid="long_tail",
        )
        for i in range(2)
    ]
    caging_y = reward_utils.hamacher_product(*caging_lr)

    # MARK: X-Z gripper information for caging reward-----------------------
    tcp = tcp_center
    xz = [0, 2]

    # Compared to the caging_y reward, caging_xz is simple. The margin is
    # constant (something in the 0.3 to 0.5 range) and x shrinks as the
    # gripper moves towards the object. After picking up the object, the
    # reward is maximized and changes very little
    caging_xz_margin = np.linalg.norm(obj_init_pos[xz] - init_tcp[xz])
    caging_xz_margin -= xz_thresh
    caging_xz = reward_utils.tolerance(
        np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
        bounds=(0, xz_thresh),
        margin=caging_xz_margin,  # "margin" in the description above
        sigmoid="long_tail",
    )

    # MARK: Closed-extent gripper information for caging reward-------------
    gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
    )

    # MARK: Combine components----------------------------------------------
    caging = reward_utils.hamacher_product(caging_y, caging_xz)
    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

    if high_density:
        return  caging_and_gripping, caging
    if medium_density:
        tcp = tcp_center
        tcp_to_obj = np.linalg.norm(obj_pos - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - init_tcp)
        # Compute reach reward
        # - We subtract object_reach_radius from the margin so that the
        #   reward always starts with a value of 0.1
        reach_margin = abs(tcp_to_obj_init - object_reach_radius)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=reach_margin,
            sigmoid="long_tail",
        )
        return caging_and_gripping, reach
'''  ,
"hammer-v2":'''
def _gripper_caging_reward(
        leftpad,
        rightpad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj_pos,
        obj_radius,
        pad_success_thresh,
        object_reach_radius,
        xz_thresh,
        desired_gripper_effort=1.0,
        high_density=False,
        medium_density=False,
):
    """Reward for agent grasping obj.
    Args:
        action(np.ndarray): (4,) array representing the action
            delta(x), delta(y), delta(z), gripper_effort
        obj_pos(np.ndarray): (3,) array representing the obj x,y,z
        obj_radius(float):radius of object's bounding sphere
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        object_reach_radius(float): successful distance of gripper center
            to the object.
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.
    """
    if high_density and medium_density:
        raise ValueError("Can only be either high_density or medium_density")
    # MARK: Left-right gripper information for caging reward----------------
    left_pad = leftpad
    right_pad = rightpad

    # get current positions of left and right pads (Y axis)
    pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
    # compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
    # compare *current* pad positions with *initial* obj position (Y axis)
    pad_to_objinit_lr = np.abs(pad_y_lr - obj_init_pos[1])

    # Compute the left/right caging rewards. This is crucial for success,
    # yet counterintuitive mathematically because we invented it
    # accidentally.
    #
    # Before touching the object, pad_to_obj_lr ("x") is always separated
    # from caging_lr_margin ("the margin") by some small number,
    # pad_success_thresh.
    #
    # When far away from the object:
    #       x = margin + pad_success_thresh
    #       --> Thus x is outside the margin, yielding very small reward.
    #           Here, any variation in the reward is due to the fact that
    #           the margin itself is shifting.
    # When near the object (within pad_success_thresh):
    #       x = pad_success_thresh - margin
    #       --> Thus x is well within the margin. As long as x > obj_radius,
    #           it will also be within the bounds, yielding maximum reward.
    #           Here, any variation in the reward is due to the gripper
    #           moving *too close* to the object (i.e, blowing past the
    #           obj_radius bound).
    #
    # Therefore, before touching the object, this is very nearly a binary
    # reward -- if the gripper is between obj_radius and pad_success_thresh,
    # it gets maximum reward. Otherwise, the reward very quickly falls off.
    #
    # After grasping the object and moving it away from initial position,
    # x remains (mostly) constant while the margin grows considerably. This
    # penalizes the agent if it moves *back* toward obj_init_pos, but
    # offers no encouragement for leaving that position in the first place.
    # That part is left to the reward functions of individual environments.
    caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
    caging_lr = [
        reward_utils.tolerance(
            pad_to_obj_lr[i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[i],  # "margin" in the description above
            sigmoid="long_tail",
        )
        for i in range(2)
    ]
    caging_y = reward_utils.hamacher_product(*caging_lr)

    # MARK: X-Z gripper information for caging reward-----------------------
    tcp = tcp_center
    xz = [0, 2]

    # Compared to the caging_y reward, caging_xz is simple. The margin is
    # constant (something in the 0.3 to 0.5 range) and x shrinks as the
    # gripper moves towards the object. After picking up the object, the
    # reward is maximized and changes very little
    caging_xz_margin = np.linalg.norm(obj_init_pos[xz] - init_tcp[xz])
    caging_xz_margin -= xz_thresh
    caging_xz = reward_utils.tolerance(
        np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
        bounds=(0, xz_thresh),
        margin=caging_xz_margin,  # "margin" in the description above
        sigmoid="long_tail",
    )

    # MARK: Closed-extent gripper information for caging reward-------------
    gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
    )

    # MARK: Combine components----------------------------------------------
    caging = reward_utils.hamacher_product(caging_y, caging_xz)
    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

    if high_density:
        return  caging_and_gripping, caging
    if medium_density:
        tcp = tcp_center
        tcp_to_obj = np.linalg.norm(obj_pos - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - init_tcp)
        # Compute reach reward
        # - We subtract object_reach_radius from the margin so that the
        #   reward always starts with a value of 0.1
        reach_margin = abs(tcp_to_obj_init - object_reach_radius)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=reach_margin,
            sigmoid="long_tail",
        )
        return caging_and_gripping, reach
'''  ,
"coffee-pull-v2":'''
def _gripper_caging_reward(
        leftpad,
        rightpad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj_pos,
        obj_radius,
        pad_success_thresh,
        object_reach_radius,
        xz_thresh,
        desired_gripper_effort=1.0,
        high_density=False,
        medium_density=False,
):
    """Reward for agent grasping obj.
    Args:
        action(np.ndarray): (4,) array representing the action
            delta(x), delta(y), delta(z), gripper_effort
        obj_pos(np.ndarray): (3,) array representing the obj x,y,z
        obj_radius(float):radius of object's bounding sphere
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        object_reach_radius(float): successful distance of gripper center
            to the object.
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.
    """
    if high_density and medium_density:
        raise ValueError("Can only be either high_density or medium_density")
    # MARK: Left-right gripper information for caging reward----------------
    left_pad = leftpad
    right_pad = rightpad

    # get current positions of left and right pads (Y axis)
    pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
    # compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
    # compare *current* pad positions with *initial* obj position (Y axis)
    pad_to_objinit_lr = np.abs(pad_y_lr - obj_init_pos[1])

    # Compute the left/right caging rewards. This is crucial for success,
    # yet counterintuitive mathematically because we invented it
    # accidentally.
    #
    # Before touching the object, pad_to_obj_lr ("x") is always separated
    # from caging_lr_margin ("the margin") by some small number,
    # pad_success_thresh.
    #
    # When far away from the object:
    #       x = margin + pad_success_thresh
    #       --> Thus x is outside the margin, yielding very small reward.
    #           Here, any variation in the reward is due to the fact that
    #           the margin itself is shifting.
    # When near the object (within pad_success_thresh):
    #       x = pad_success_thresh - margin
    #       --> Thus x is well within the margin. As long as x > obj_radius,
    #           it will also be within the bounds, yielding maximum reward.
    #           Here, any variation in the reward is due to the gripper
    #           moving *too close* to the object (i.e, blowing past the
    #           obj_radius bound).
    #
    # Therefore, before touching the object, this is very nearly a binary
    # reward -- if the gripper is between obj_radius and pad_success_thresh,
    # it gets maximum reward. Otherwise, the reward very quickly falls off.
    #
    # After grasping the object and moving it away from initial position,
    # x remains (mostly) constant while the margin grows considerably. This
    # penalizes the agent if it moves *back* toward obj_init_pos, but
    # offers no encouragement for leaving that position in the first place.
    # That part is left to the reward functions of individual environments.
    caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
    caging_lr = [
        reward_utils.tolerance(
            pad_to_obj_lr[i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[i],  # "margin" in the description above
            sigmoid="long_tail",
        )
        for i in range(2)
    ]
    caging_y = reward_utils.hamacher_product(*caging_lr)

    # MARK: X-Z gripper information for caging reward-----------------------
    tcp = tcp_center
    xz = [0, 2]

    # Compared to the caging_y reward, caging_xz is simple. The margin is
    # constant (something in the 0.3 to 0.5 range) and x shrinks as the
    # gripper moves towards the object. After picking up the object, the
    # reward is maximized and changes very little
    caging_xz_margin = np.linalg.norm(obj_init_pos[xz] - init_tcp[xz])
    caging_xz_margin -= xz_thresh
    caging_xz = reward_utils.tolerance(
        np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
        bounds=(0, xz_thresh),
        margin=caging_xz_margin,  # "margin" in the description above
        sigmoid="long_tail",
    )

    # MARK: Closed-extent gripper information for caging reward-------------
    gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
    )

    # MARK: Combine components----------------------------------------------
    caging = reward_utils.hamacher_product(caging_y, caging_xz)
    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

    if high_density:
        return  caging_and_gripping, caging
    if medium_density:
        tcp = tcp_center
        tcp_to_obj = np.linalg.norm(obj_pos - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - init_tcp)
        # Compute reach reward
        # - We subtract object_reach_radius from the margin so that the
        #   reward always starts with a value of 0.1
        reach_margin = abs(tcp_to_obj_init - object_reach_radius)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=reach_margin,
            sigmoid="long_tail",
        )
        return caging_and_gripping, reach
'''  ,
"reach-v2": """
def _gripper_caging_reward():
    pass""",
"peg-unplug-side-v2":'''
def _gripper_caging_reward(
        leftpad,
        rightpad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj_pos,
        obj_radius,
        pad_success_thresh,
        object_reach_radius,
        xz_thresh,
        desired_gripper_effort=1.0,
        high_density=False,
        medium_density=False,
):
    """Reward for agent grasping obj.
    Args:
        action(np.ndarray): (4,) array representing the action
            delta(x), delta(y), delta(z), gripper_effort
        obj_pos(np.ndarray): (3,) array representing the obj x,y,z
        obj_radius(float):radius of object's bounding sphere
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        object_reach_radius(float): successful distance of gripper center
            to the object.
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.
    """
    if high_density and medium_density:
        raise ValueError("Can only be either high_density or medium_density")
    # MARK: Left-right gripper information for caging reward----------------
    left_pad = leftpad
    right_pad = rightpad

    # get current positions of left and right pads (Y axis)
    pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
    # compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
    # compare *current* pad positions with *initial* obj position (Y axis)
    pad_to_objinit_lr = np.abs(pad_y_lr - obj_init_pos[1])

    # Compute the left/right caging rewards. This is crucial for success,
    # yet counterintuitive mathematically because we invented it
    # accidentally.
    #
    # Before touching the object, pad_to_obj_lr ("x") is always separated
    # from caging_lr_margin ("the margin") by some small number,
    # pad_success_thresh.
    #
    # When far away from the object:
    #       x = margin + pad_success_thresh
    #       --> Thus x is outside the margin, yielding very small reward.
    #           Here, any variation in the reward is due to the fact that
    #           the margin itself is shifting.
    # When near the object (within pad_success_thresh):
    #       x = pad_success_thresh - margin
    #       --> Thus x is well within the margin. As long as x > obj_radius,
    #           it will also be within the bounds, yielding maximum reward.
    #           Here, any variation in the reward is due to the gripper
    #           moving *too close* to the object (i.e, blowing past the
    #           obj_radius bound).
    #
    # Therefore, before touching the object, this is very nearly a binary
    # reward -- if the gripper is between obj_radius and pad_success_thresh,
    # it gets maximum reward. Otherwise, the reward very quickly falls off.
    #
    # After grasping the object and moving it away from initial position,
    # x remains (mostly) constant while the margin grows considerably. This
    # penalizes the agent if it moves *back* toward obj_init_pos, but
    # offers no encouragement for leaving that position in the first place.
    # That part is left to the reward functions of individual environments.
    caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
    caging_lr = [
        reward_utils.tolerance(
            pad_to_obj_lr[i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[i],  # "margin" in the description above
            sigmoid="long_tail",
        )
        for i in range(2)
    ]
    caging_y = reward_utils.hamacher_product(*caging_lr)

    # MARK: X-Z gripper information for caging reward-----------------------
    tcp = tcp_center
    xz = [0, 2]

    # Compared to the caging_y reward, caging_xz is simple. The margin is
    # constant (something in the 0.3 to 0.5 range) and x shrinks as the
    # gripper moves towards the object. After picking up the object, the
    # reward is maximized and changes very little
    caging_xz_margin = np.linalg.norm(obj_init_pos[xz] - init_tcp[xz])
    caging_xz_margin -= xz_thresh
    caging_xz = reward_utils.tolerance(
        np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
        bounds=(0, xz_thresh),
        margin=caging_xz_margin,  # "margin" in the description above
        sigmoid="long_tail",
    )

    # MARK: Closed-extent gripper information for caging reward-------------
    gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
    )

    # MARK: Combine components----------------------------------------------
    caging = reward_utils.hamacher_product(caging_y, caging_xz)
    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

    if high_density:
        return  caging_and_gripping, caging
    if medium_density:
        tcp = tcp_center
        tcp_to_obj = np.linalg.norm(obj_pos - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - init_tcp)
        # Compute reach reward
        # - We subtract object_reach_radius from the margin so that the
        #   reward always starts with a value of 0.1
        reach_margin = abs(tcp_to_obj_init - object_reach_radius)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=reach_margin,
            sigmoid="long_tail",
        )
        return caging_and_gripping, reach
'''  ,
"soccer-v2": """def _gripper_caging_reward(action, obj_init_pos, left_pad,  right_pad, init_left_pad, init_right_pad, tcp_center,init_tcp, obj_position, obj_radius):
    pad_success_margin = 0.05
    grip_success_margin = obj_radius + 0.01
    x_z_success_margin = 0.005

    delta_object_y_left_pad = left_pad[1] - obj_position[1]
    delta_object_y_right_pad = obj_position[1] - right_pad[1]
    right_caging_margin = abs(
        abs(obj_position[1] - init_right_pad[1]) - pad_success_margin
    )
    left_caging_margin = abs(
        abs(obj_position[1] - init_left_pad[1]) - pad_success_margin
    )

    right_caging = reward_utils.tolerance(
        delta_object_y_right_pad,
        bounds=(obj_radius, pad_success_margin),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_caging = reward_utils.tolerance(
        delta_object_y_left_pad,
        bounds=(obj_radius, pad_success_margin),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )

    right_gripping = reward_utils.tolerance(
        delta_object_y_right_pad,
        bounds=(obj_radius, grip_success_margin),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_gripping = reward_utils.tolerance(
        delta_object_y_left_pad,
        bounds=(obj_radius, grip_success_margin),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )

    assert right_caging >= 0 and right_caging <= 1
    assert left_caging >= 0 and left_caging <= 1

    y_caging = reward_utils.hamacher_product(right_caging, left_caging)
    y_gripping = reward_utils.hamacher_product(right_gripping, left_gripping)

    assert y_caging >= 0 and y_caging <= 1

    tcp_xz = tcp_center + np.array([0.0, -tcp_center[1], 0.0])
    obj_position_x_z = np.copy(obj_position) + np.array(
        [0.0, -obj_position[1], 0.0]
    )
    tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
    init_obj_x_z = obj_init_pos + np.array([0.0, -obj_init_pos[1], 0.0])
    init_tcp_x_z = init_tcp + np.array([0.0, -init_tcp[1], 0.0])

    tcp_obj_x_z_margin = (
        np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
    )
    x_z_caging = reward_utils.tolerance(
        tcp_obj_norm_x_z,
        bounds=(0, x_z_success_margin),
        margin=tcp_obj_x_z_margin,
        sigmoid="long_tail",
    )

    assert right_caging >= 0 and right_caging <= 1
    gripper_closed = min(max(0, action[-1]), 1)
    assert gripper_closed >= 0 and gripper_closed <= 1
    caging = reward_utils.hamacher_product(y_caging, x_z_caging)
    assert caging >= 0 and caging <= 1

    if caging > 0.95:
        gripping = y_gripping
    else:
        gripping = 0.0
    assert gripping >= 0 and gripping <= 1

    caging_and_gripping = (caging + gripping) / 2
    assert caging_and_gripping >= 0 and caging_and_gripping <= 1

    return caging_and_gripping""",
"coffee-push-v2":'''
def _gripper_caging_reward(
        leftpad,
        rightpad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj_pos,
        obj_radius,
        pad_success_thresh,
        object_reach_radius,
        xz_thresh,
        desired_gripper_effort=1.0,
        high_density=False,
        medium_density=False,
):
    """Reward for agent grasping obj.
    Args:
        action(np.ndarray): (4,) array representing the action
            delta(x), delta(y), delta(z), gripper_effort
        obj_pos(np.ndarray): (3,) array representing the obj x,y,z
        obj_radius(float):radius of object's bounding sphere
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        object_reach_radius(float): successful distance of gripper center
            to the object.
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.
    """
    if high_density and medium_density:
        raise ValueError("Can only be either high_density or medium_density")
    # MARK: Left-right gripper information for caging reward----------------
    left_pad = leftpad
    right_pad = rightpad

    # get current positions of left and right pads (Y axis)
    pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
    # compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
    # compare *current* pad positions with *initial* obj position (Y axis)
    pad_to_objinit_lr = np.abs(pad_y_lr - obj_init_pos[1])

    # Compute the left/right caging rewards. This is crucial for success,
    # yet counterintuitive mathematically because we invented it
    # accidentally.
    #
    # Before touching the object, pad_to_obj_lr ("x") is always separated
    # from caging_lr_margin ("the margin") by some small number,
    # pad_success_thresh.
    #
    # When far away from the object:
    #       x = margin + pad_success_thresh
    #       --> Thus x is outside the margin, yielding very small reward.
    #           Here, any variation in the reward is due to the fact that
    #           the margin itself is shifting.
    # When near the object (within pad_success_thresh):
    #       x = pad_success_thresh - margin
    #       --> Thus x is well within the margin. As long as x > obj_radius,
    #           it will also be within the bounds, yielding maximum reward.
    #           Here, any variation in the reward is due to the gripper
    #           moving *too close* to the object (i.e, blowing past the
    #           obj_radius bound).
    #
    # Therefore, before touching the object, this is very nearly a binary
    # reward -- if the gripper is between obj_radius and pad_success_thresh,
    # it gets maximum reward. Otherwise, the reward very quickly falls off.
    #
    # After grasping the object and moving it away from initial position,
    # x remains (mostly) constant while the margin grows considerably. This
    # penalizes the agent if it moves *back* toward obj_init_pos, but
    # offers no encouragement for leaving that position in the first place.
    # That part is left to the reward functions of individual environments.
    caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
    caging_lr = [
        reward_utils.tolerance(
            pad_to_obj_lr[i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[i],  # "margin" in the description above
            sigmoid="long_tail",
        )
        for i in range(2)
    ]
    caging_y = reward_utils.hamacher_product(*caging_lr)

    # MARK: X-Z gripper information for caging reward-----------------------
    tcp = tcp_center
    xz = [0, 2]

    # Compared to the caging_y reward, caging_xz is simple. The margin is
    # constant (something in the 0.3 to 0.5 range) and x shrinks as the
    # gripper moves towards the object. After picking up the object, the
    # reward is maximized and changes very little
    caging_xz_margin = np.linalg.norm(obj_init_pos[xz] - init_tcp[xz])
    caging_xz_margin -= xz_thresh
    caging_xz = reward_utils.tolerance(
        np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
        bounds=(0, xz_thresh),
        margin=caging_xz_margin,  # "margin" in the description above
        sigmoid="long_tail",
    )

    # MARK: Closed-extent gripper information for caging reward-------------
    gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
    )

    # MARK: Combine components----------------------------------------------
    caging = reward_utils.hamacher_product(caging_y, caging_xz)
    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

    if high_density:
        return  caging_and_gripping, caging
    if medium_density:
        tcp = tcp_center
        tcp_to_obj = np.linalg.norm(obj_pos - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - init_tcp)
        # Compute reach reward
        # - We subtract object_reach_radius from the margin so that the
        #   reward always starts with a value of 0.1
        reach_margin = abs(tcp_to_obj_init - object_reach_radius)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=reach_margin,
            sigmoid="long_tail",
        )
        return caging_and_gripping, reach
'''  ,
"sweep-into-v2": """def _gripper_caging_reward(action, obj_position, obj_radius, tcp_center, left_pad, right_pad, init_tcp, init_left_pad, init_right_pad, obj_init_pos):
    pad_success_margin = 0.05
    grip_success_margin = obj_radius + 0.005
    x_z_success_margin = 0.01

    tcp = tcp_center

    delta_object_y_left_pad = left_pad[1] - obj_position[1]
    delta_object_y_right_pad = obj_position[1] - right_pad[1]
    right_caging_margin = abs(
        abs(obj_position[1] - init_right_pad[1]) - pad_success_margin
    )
    left_caging_margin = abs(
        abs(obj_position[1] - init_left_pad[1]) - pad_success_margin
    )

    right_caging = reward_utils.tolerance(
        delta_object_y_right_pad,
        bounds=(obj_radius, pad_success_margin),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_caging = reward_utils.tolerance(
        delta_object_y_left_pad,
        bounds=(obj_radius, pad_success_margin),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )

    right_gripping = reward_utils.tolerance(
        delta_object_y_right_pad,
        bounds=(obj_radius, grip_success_margin),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_gripping = reward_utils.tolerance(
        delta_object_y_left_pad,
        bounds=(obj_radius, grip_success_margin),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )

    assert right_caging >= 0 and right_caging <= 1
    assert left_caging >= 0 and left_caging <= 1

    y_caging = reward_utils.hamacher_product(right_caging, left_caging)
    y_gripping = reward_utils.hamacher_product(right_gripping, left_gripping)

    assert y_caging >= 0 and y_caging <= 1

    tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
    obj_position_x_z = np.copy(obj_position) + np.array(
        [0.0, -obj_position[1], 0.0]
    )
    tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
    init_obj_x_z = obj_init_pos + np.array([0.0, -obj_init_pos[1], 0.0])
    init_tcp_x_z = init_tcp + np.array([0.0, -init_tcp[1], 0.0])

    tcp_obj_x_z_margin = (
            np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
    )
    x_z_caging = reward_utils.tolerance(
        tcp_obj_norm_x_z,
        bounds=(0, x_z_success_margin),
        margin=tcp_obj_x_z_margin,
        sigmoid="long_tail",
    )

    assert right_caging >= 0 and right_caging <= 1
    gripper_closed = min(max(0, action[-1]), 1)
    assert gripper_closed >= 0 and gripper_closed <= 1
    caging = reward_utils.hamacher_product(y_caging, x_z_caging)
    assert caging >= 0 and caging <= 1

    if caging > 0.95:
        gripping = y_gripping
    else:
        gripping = 0.0
    assert gripping >= 0 and gripping <= 1

    caging_and_gripping = (caging + gripping) / 2
    assert caging_and_gripping >= 0 and caging_and_gripping <= 1

    return caging_and_gripping""",
"peg-insert-side-v2":'''def _gripper_caging_reward(
        leftpad,
        rightpad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj_pos,
        obj_radius,
        pad_success_thresh,
        object_reach_radius,
        xz_thresh,
        desired_gripper_effort=1.0,
        high_density=False,
        medium_density=False,
):
    """Reward for agent grasping obj.
    Args:
        action(np.ndarray): (4,) array representing the action
            delta(x), delta(y), delta(z), gripper_effort
        obj_pos(np.ndarray): (3,) array representing the obj x,y,z
        obj_radius(float):radius of object's bounding sphere
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        object_reach_radius(float): successful distance of gripper center
            to the object.
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.
    """
    if high_density and medium_density:
        raise ValueError("Can only be either high_density or medium_density")
    # MARK: Left-right gripper information for caging reward----------------
    left_pad = leftpad
    right_pad = rightpad

    # get current positions of left and right pads (Y axis)
    pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
    # compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
    # compare *current* pad positions with *initial* obj position (Y axis)
    pad_to_objinit_lr = np.abs(pad_y_lr - obj_init_pos[1])

    # Compute the left/right caging rewards. This is crucial for success,
    # yet counterintuitive mathematically because we invented it
    # accidentally.
    #
    # Before touching the object, pad_to_obj_lr ("x") is always separated
    # from caging_lr_margin ("the margin") by some small number,
    # pad_success_thresh.
    #
    # When far away from the object:
    #       x = margin + pad_success_thresh
    #       --> Thus x is outside the margin, yielding very small reward.
    #           Here, any variation in the reward is due to the fact that
    #           the margin itself is shifting.
    # When near the object (within pad_success_thresh):
    #       x = pad_success_thresh - margin
    #       --> Thus x is well within the margin. As long as x > obj_radius,
    #           it will also be within the bounds, yielding maximum reward.
    #           Here, any variation in the reward is due to the gripper
    #           moving *too close* to the object (i.e, blowing past the
    #           obj_radius bound).
    #
    # Therefore, before touching the object, this is very nearly a binary
    # reward -- if the gripper is between obj_radius and pad_success_thresh,
    # it gets maximum reward. Otherwise, the reward very quickly falls off.
    #
    # After grasping the object and moving it away from initial position,
    # x remains (mostly) constant while the margin grows considerably. This
    # penalizes the agent if it moves *back* toward obj_init_pos, but
    # offers no encouragement for leaving that position in the first place.
    # That part is left to the reward functions of individual environments.
    caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
    caging_lr = [
        reward_utils.tolerance(
            pad_to_obj_lr[i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[i],  # "margin" in the description above
            sigmoid="long_tail",
        )
        for i in range(2)
    ]
    caging_y = reward_utils.hamacher_product(*caging_lr)

    # MARK: X-Z gripper information for caging reward-----------------------
    tcp = tcp_center
    xz = [0, 2]

    # Compared to the caging_y reward, caging_xz is simple. The margin is
    # constant (something in the 0.3 to 0.5 range) and x shrinks as the
    # gripper moves towards the object. After picking up the object, the
    # reward is maximized and changes very little
    caging_xz_margin = np.linalg.norm(obj_init_pos[xz] - init_tcp[xz])
    caging_xz_margin -= xz_thresh
    caging_xz = reward_utils.tolerance(
        np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
        bounds=(0, xz_thresh),
        margin=caging_xz_margin,  # "margin" in the description above
        sigmoid="long_tail",
    )

    # MARK: Closed-extent gripper information for caging reward-------------
    gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
    )

    # MARK: Combine components----------------------------------------------
    caging = reward_utils.hamacher_product(caging_y, caging_xz)
    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

    if high_density:
        caging_and_gripping = (caging_and_gripping + caging) / 2
    if medium_density:
        tcp = tcp_center
        tcp_to_obj = np.linalg.norm(obj_pos - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - init_tcp)
        # Compute reach reward
        # - We subtract object_reach_radius from the margin so that the
        #   reward always starts with a value of 0.1
        reach_margin = abs(tcp_to_obj_init - object_reach_radius)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=reach_margin,
            sigmoid="long_tail",
        )
        caging_and_gripping = (caging_and_gripping + reach) / 2

    return caging_and_gripping
'''}
reward_function_dict = {
"button-press-topdown-v2": """def compute_reward(action, obs, tcp_center, _target_pos, init_tcp, _obj_to_target_init):
    del action
    obj = obs[4:7]
    tcp = tcp_center
    tcp_to_obj = np.linalg.norm(obj - tcp)
    tcp_to_obj_init = np.linalg.norm(obj - init_tcp)
    obj_to_target = abs(_target_pos[2] - obj[2])
    tcp_closed = 1 - obs[3]
    near_button = reward_utils.tolerance(
        tcp_to_obj,
        bounds=(0, 0.01),
        margin=tcp_to_obj_init,
        sigmoid="long_tail",
    )
    button_pressed = reward_utils.tolerance(
        obj_to_target,
        bounds=(0, 0.005),
        margin=_obj_to_target_init,
        sigmoid="long_tail",
    )
    reward = 5 * reward_utils.hamacher_product(tcp_closed, near_button)
    if tcp_to_obj <= 0.03:
        reward += 5 * button_pressed
    return (reward, tcp_to_obj, obs[3], obj_to_target, near_button, button_pressed)""",
"basketball-v2": """def compute_reward(action, obs, _target_pos, obj_init_pos, TARGET_RADIUS, tcp_center, left_pad, right_pad, init_tcp):
    obj = obs[4:7]
    # Force target to be slightly above basketball hoop
    target = _target_pos
    target[2] = 0.3

    # Emphasize Z error
    scale = np.array([1.0, 1.0, 2.0])
    target_to_obj = (obj - target) * scale
    target_to_obj = np.linalg.norm(target_to_obj)
    target_to_obj_init = (obj_init_pos - target) * scale
    target_to_obj_init = np.linalg.norm(target_to_obj_init)

    in_place = reward_utils.tolerance(
        target_to_obj,
        bounds=(0, TARGET_RADIUS),
        margin=target_to_obj_init,
        sigmoid="long_tail",
    )
    tcp_opened = obs[3]
    tcp_to_obj = np.linalg.norm(obj - tcp_center)

    caging_and_gripping, caging = _gripper_caging_reward(
        left_pad,
        right_pad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj,
        object_reach_radius=0.01,
        obj_radius=0.025,
        pad_success_thresh=0.06,
        xz_thresh=0.005,
        high_density=True,
    )
    object_grasped = (caging_and_gripping + caging)/2.0

    if (
            tcp_to_obj < 0.035
            and tcp_opened > 0
            and obj[2] - 0.01 > obj_init_pos[2]
    ):
        object_grasped = 1
    reward = reward_utils.hamacher_product(object_grasped, in_place)

    if (
            tcp_to_obj < 0.035
            and tcp_opened > 0
            and obj[2] - 0.01 > obj_init_pos[2]
    ):
        reward += 1.0 + 5.0 * in_place
    if target_to_obj < TARGET_RADIUS:
        reward = 10.0
    return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)""",
"push-back-v2": """def compute_reward(action, obs, tcp_center, _target_pos, obj_init_pos, TARGET_RADIUS, OBJ_RADIUS, left_pad, right_pad, init_left_pad, init_right_pad, init_tcp):
    obj = obs[4:7]
    tcp_opened = obs[3]
    tcp_to_obj = np.linalg.norm(obj - tcp_center)
    target_to_obj = np.linalg.norm(obj - _target_pos)
    target_to_obj_init = np.linalg.norm(obj_init_pos - _target_pos)
    in_place = reward_utils.tolerance(
        target_to_obj,
        bounds=(0, TARGET_RADIUS),
        margin=target_to_obj_init,
        sigmoid="long_tail",
    )
    object_grasped = _gripper_caging_reward(tcp_center, left_pad, right_pad, init_left_pad, init_right_pad, obj_init_pos, init_tcp, action, obj, OBJ_RADIUS)
    reward = reward_utils.hamacher_product(object_grasped, in_place)
    if (
            (tcp_to_obj < 0.01)
            and (0 < tcp_opened < 0.55)
            and (target_to_obj_init - target_to_obj > 0.01)
    ):
        reward += 1.0 + 5.0 * in_place
    if target_to_obj < TARGET_RADIUS:
        reward = 10.0
    return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)""",
"dial-turn-v2": """
def compute_reward(action, obs, obj, tcp_center, _target_pos, pos_objects, TARGET_RADIUS, init_tcp):
    dial_push_position = pos_objects + np.array([0.05, 0.02, 0.09])
    tcp = tcp_center
    target = _target_pos

    target_to_obj = obj - target
    target_to_obj = np.linalg.norm(target_to_obj)
    target_to_obj_init = dial_push_position - target
    target_to_obj_init = np.linalg.norm(target_to_obj_init)

    in_place = reward_utils.tolerance(
        target_to_obj,
        bounds=(0, TARGET_RADIUS),
        margin=abs(target_to_obj_init - TARGET_RADIUS),
        sigmoid="long_tail",
    )

    dial_reach_radius = 0.005
    tcp_to_obj = np.linalg.norm(dial_push_position - tcp)
    tcp_to_obj_init = np.linalg.norm(dial_push_position - init_tcp)
    reach = reward_utils.tolerance(
        tcp_to_obj,
        bounds=(0, dial_reach_radius),
        margin=abs(tcp_to_obj_init - dial_reach_radius),
        sigmoid="gaussian",
    )
    gripper_closed = min(max(0, action[-1]), 1)

    reach = reward_utils.hamacher_product(reach, gripper_closed)
    tcp_opened = 0
    object_grasped = reach

    reward = 10 * reward_utils.hamacher_product(reach, in_place)
    return (
        reward[0],
        tcp_to_obj,
        tcp_opened,
        target_to_obj,
        object_grasped,
        in_place,
    )""" ,
"hand-insert-v2": """def compute_reward(action, obs, _target_pos, obj_init_pos, TARGET_RADIUS, tcp_center, left_pad, right_pad, init_tcp):
    obj = obs[4:7]

    target_to_obj = np.linalg.norm(obj - _target_pos)
    target_to_obj_init = np.linalg.norm(obj_init_pos - _target_pos)

    in_place = reward_utils.tolerance(
        target_to_obj,
        bounds=(0, TARGET_RADIUS),
        margin=target_to_obj_init,
        sigmoid="long_tail",
    )
    caging_and_gripping, caging = _gripper_caging_reward(
        left_pad,
        right_pad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj,
        object_reach_radius=0.01,
        obj_radius=0.015,
        pad_success_thresh=0.05,
        xz_thresh=0.005,
        high_density=True,
    )

    object_grasped = (caging_and_gripping + caging)/2.0
    reward = reward_utils.hamacher_product(object_grasped, in_place)

    tcp_opened = obs[3]
    tcp_to_obj = np.linalg.norm(obj - tcp_center)

    if tcp_to_obj < 0.02 and tcp_opened > 0:
        reward += 1.0 + 7.0 * in_place
    if target_to_obj < TARGET_RADIUS:
        reward = 10.0
    return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)""",
"pick-out-of-hole-v2": """
def compute_reward(action, obs, tcp_center, _target_pos, obj_init_pos, TARGET_RADIUS, left_pad, right_pad, init_tcp):
    obj = obs[4:7]
    gripper = tcp_center

    obj_to_target = np.linalg.norm(obj - _target_pos)
    tcp_to_obj = np.linalg.norm(obj - gripper)
    in_place_margin = np.linalg.norm(obj_init_pos - _target_pos)

    threshold = 0.03
    # floor is a 3D funnel centered on the initial object pos
    radius = np.linalg.norm(gripper[:2] - obj_init_pos[:2])
    if radius <= threshold:
        floor = 0.0
    else:
        floor = 0.015 * np.log(radius - threshold) + 0.15
    # prevent the hand from running into cliff edge by staying above floor
    above_floor = (
        1.0
        if gripper[2] >= floor
        else reward_utils.tolerance(
            max(floor - gripper[2], 0.0),
            bounds=(0.0, 0.01),
            margin=0.02,
            sigmoid="long_tail",
        )
    )
    caging_and_gripping , caging = _gripper_caging_reward(
        left_pad,
        right_pad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj,
        object_reach_radius=0.01,
        obj_radius=0.015,
        pad_success_thresh=0.02,
        xz_thresh=0.03,
        desired_gripper_effort=0.1,
        high_density=True,
    )
    object_grasped = (caging_and_gripping + caging)/2.0
    in_place = reward_utils.tolerance(
        obj_to_target, bounds=(0, 0.02), margin=in_place_margin, sigmoid="long_tail"
    )
    reward = reward_utils.hamacher_product(object_grasped, in_place)

    near_object = tcp_to_obj < 0.04
    pinched_without_obj = obs[3] < 0.33
    lifted = obj[2] - 0.02 > obj_init_pos[2]
    # Increase reward when properly grabbed obj
    grasp_success = near_object and lifted and not pinched_without_obj
    if grasp_success:
        reward += 1.0 + 5.0 * reward_utils.hamacher_product(in_place, above_floor)
    # Maximize reward on success
    if obj_to_target < TARGET_RADIUS:
        reward = 10.0

    return (
        reward,
        tcp_to_obj,
        grasp_success,
        obj_to_target,
        object_grasped,
        in_place,
    )""",
"hammer-v2": """
def compute_reward(action, obs, HAMMER_HANDLE_LENGTH, _target_pos, NailSlideJoint_qpos, left_pad, right_pad, tcp_center, obj_init_pos, init_tcp):
    hand = obs[:3]
    hammer = obs[4:7]
    hammer_head = hammer + np.array([0.16, 0.06, 0.0])
    # `_gripper_caging_reward` assumes that the target object can be
    # approximated as a sphere. This is not true for the hammer handle, so
    # to avoid re-writing the `_gripper_caging_reward` we pass in a
    # modified hammer position.
    # This modified position's X value will perfect match the hand's X value
    # as long as it's within a certain threshold
    hammer_threshed = hammer.copy()
    threshold = HAMMER_HANDLE_LENGTH / 2.0
    if abs(hammer[0] - hand[0]) < threshold:
        hammer_threshed[0] = hand[0]
    # Ideal laid-down wrench has quat [1, 0, 0, 0]
    # Rather than deal with an angle between quaternions, just approximate:
    ideal = np.array([1.0, 0.0, 0.0, 0.0])
    error = np.linalg.norm(obs[7:11] - ideal)
    reward_quat = max(1.0 - error / 0.4, 0.0)
    caging_and_gripping, caging = _gripper_caging_reward(
        left_pad,
        right_pad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        hammer_threshed,
        object_reach_radius=0.01,
        obj_radius=0.015,
        pad_success_thresh=0.02,
        xz_thresh=0.01,
        high_density=True,
    )
    reward_grab =  (caging_and_gripping + caging) / 2.0
    pos_error = _target_pos - hammer_head
    a = 0.1  # Relative importance of just *trying* to lift the hammer
    b = 0.9  # Relative importance of hitting the nail
    lifted = hammer_head[2] > 0.02
    reward_in_place = a * float(lifted) + b * reward_utils.tolerance(
        np.linalg.norm(pos_error),
        bounds=(0, 0.02),
        margin=0.2,
        sigmoid="long_tail",
    )
    reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
    # Override reward on success. We check that reward is above a threshold
    # because this env's success metric could be hacked easily
    success = NailSlideJoint_qpos > 0.09
    if success and reward > 5.0:
        reward = 10.0
    return (
        reward,
        reward_grab,
        reward_quat,
        reward_in_place,
        success,
    )""",
"coffee-pull-v2": """def compute_reward(action, obs, _target_pos, obj_init_pos, tcp_center,left_pad, right_pad, init_tcp):
    obj = obs[4:7]
    target = _target_pos.copy()

    # Emphasize X and Y errors
    scale = np.array([2.0, 2.0, 1.0])
    target_to_obj = (obj - target) * scale
    target_to_obj = np.linalg.norm(target_to_obj)
    target_to_obj_init = (obj_init_pos - target) * scale
    target_to_obj_init = np.linalg.norm(target_to_obj_init)

    in_place = reward_utils.tolerance(
        target_to_obj,
        bounds=(0, 0.05),
        margin=target_to_obj_init,
        sigmoid="long_tail",
    )
    tcp_opened = obs[3]
    tcp_to_obj = np.linalg.norm(obj - tcp_center)

    caging_and_gripping, reach = _gripper_caging_reward(
        left_pad,
        right_pad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj,
        object_reach_radius=0.04,
        obj_radius=0.02,
        pad_success_thresh=0.05,
        xz_thresh=0.05,
        desired_gripper_effort=0.7,
        medium_density=True,
    )
    object_grasped = (caging_and_gripping + reach)/2.0
    reward = reward_utils.hamacher_product(object_grasped, in_place)

    if tcp_to_obj < 0.04 and tcp_opened > 0:
        reward += 1.0 + 5.0 * in_place
    if target_to_obj < 0.05:
        reward = 10.0
    return (
        reward,
        tcp_to_obj,
        tcp_opened,
        np.linalg.norm(obj - target),  # recompute to avoid `scale` above
        object_grasped,
        in_place,
    )""",
"reach-v2": """def compute_reward(actions, obs, tcp_center,_target_pos, hand_init_pos):
    _TARGET_RADIUS = 0.05
    tcp = tcp_center
    # obj = obs[4:7]
    # tcp_opened = obs[3]
    target = _target_pos

    tcp_to_target = np.linalg.norm(tcp - target)
    # obj_to_target = np.linalg.norm(obj - target)

    in_place_margin = np.linalg.norm(hand_init_pos - target)
    in_place = reward_utils.tolerance(
        tcp_to_target,
        bounds=(0, _TARGET_RADIUS),
        margin=in_place_margin,
        sigmoid="long_tail",
    )
    reward = 10* in_place
    return [reward, tcp_to_target, in_place]""",
"peg-unplug-side-v2": """def compute_reward(action, obs, tcp_center,_target_pos, obj_init_pos, left_pad, right_pad, init_tcp):
    tcp = tcp_center
    obj = obs[4:7]
    tcp_opened = obs[3]
    target = _target_pos
    tcp_to_obj = np.linalg.norm(obj - tcp)
    obj_to_target = np.linalg.norm(obj - target)
    pad_success_margin = 0.05
    object_reach_radius = 0.01
    x_z_margin = 0.005
    obj_radius = 0.025

    caging_and_gripping, caging = _gripper_caging_reward(
        left_pad,
        right_pad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj,
        object_reach_radius=object_reach_radius,
        obj_radius=obj_radius,
        pad_success_thresh=pad_success_margin,
        xz_thresh=x_z_margin,
        desired_gripper_effort=0.8,
        high_density=True,
    )
    object_grasped = (caging_and_gripping + caging) / 2
    in_place_margin = np.linalg.norm(obj_init_pos - target)

    in_place = reward_utils.tolerance(
        obj_to_target,
        bounds=(0, 0.05),
        margin=in_place_margin,
        sigmoid="long_tail",
    )
    grasp_success = tcp_opened > 0.5 and (obj[0] - obj_init_pos[0] > 0.015)

    reward = 2 * object_grasped

    if grasp_success and tcp_to_obj < 0.035:
        reward = 1 + 2 * object_grasped + 5 * in_place

    if obj_to_target <= 0.05:
        reward = 10.0

    return (
        reward,
        tcp_to_obj,
        tcp_opened,
        obj_to_target,
        object_grasped,
        in_place,
        float(grasp_success),
    )""",
"soccer-v2": """def compute_reward(tcp_center,_target_pos, obj_init_pos, action, obs, TARGET_RADIUS, left_pad,  right_pad, init_left_pad, init_right_pad, init_tcp, OBJ_RADIUS):
    obj = obs[4:7]
    tcp_opened = obs[3]
    x_scaling = np.array([3.0, 1.0, 1.0])
    tcp_to_obj = np.linalg.norm(obj - tcp_center)
    target_to_obj = np.linalg.norm((obj - _target_pos) * x_scaling)
    target_to_obj_init = np.linalg.norm((obj - obj_init_pos) * x_scaling)

    in_place = reward_utils.tolerance(
        target_to_obj,
        bounds=(0, TARGET_RADIUS),
        margin=target_to_obj_init,
        sigmoid="long_tail",
    )

    goal_line = _target_pos[1] - 0.1
    if obj[1] > goal_line and abs(obj[0] - _target_pos[0]) > 0.10:
        in_place = np.clip(
            in_place - 2 * ((obj[1] - goal_line) / (1 - goal_line)), 0.0, 1.0)

    object_grasped = _gripper_caging_reward(action, obj_init_pos, left_pad,  right_pad, init_left_pad, init_right_pad, tcp_center,init_tcp, obj, OBJ_RADIUS)

    reward = (3 * object_grasped) + (6.5 * in_place)

    if target_to_obj < TARGET_RADIUS:
        reward = 10.0
    return (
        reward,
        tcp_to_obj,
        tcp_opened,
        np.linalg.norm(obj - _target_pos),
        object_grasped,
        in_place,
    )""",
"coffee-push-v2":"""
def compute_reward(action, obs, _target_pos, obj_init_pos, tcp_center, left_pad, right_pad, init_tcp):
    obj = obs[4:7]

    # Emphasize X and Y errors
    scale = np.array([2.0, 2.0, 1.0])
    target_to_obj = (obj - _target_pos) * scale
    target_to_obj = np.linalg.norm(target_to_obj)
    target_to_obj_init = (obj_init_pos - _target_pos) * scale
    target_to_obj_init = np.linalg.norm(target_to_obj_init)

    in_place = reward_utils.tolerance(
        target_to_obj,
        bounds=(0, 0.05),
        margin=target_to_obj_init,
        sigmoid="long_tail",
    )
    tcp_opened = obs[3]
    tcp_to_obj = np.linalg.norm(obj - tcp_center)

    caging_and_gripping_reward, reach_reward  = _gripper_caging_reward(
        left_pad,
        right_pad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj,
        object_reach_radius=0.04,
        obj_radius=0.02,
        pad_success_thresh=0.05,
        xz_thresh=0.05,
        desired_gripper_effort=0.7,
        medium_density=True,
    )
    #caging_and_gripping, reach = reward_utils.hamacher_product(object_grasped, in_place)
    object_grasped = (caging_and_gripping_reward + reach_reward)/2.0
    denominator = object_grasped + in_place - (object_grasped * in_place)
    object_grasped_reward = ((object_grasped * object_grasped) / denominator) if denominator > 0 else 0
    reward = object_grasped_reward

    if tcp_to_obj < 0.04 and tcp_opened > 0:
        reward += 1.0 + 5.0 * in_place
    if target_to_obj < 0.05:
        reward = 10.0
    return (
        reward,
        tcp_to_obj,
        tcp_opened,
        np.linalg.norm(obj - _target_pos),  # recompute to avoid `scale` above
        object_grasped,
        in_place,
    )"""
    ,
    "peg-insert-side-v2": """def compute_reward(init_tcp, left_pad, right_pad, action, obs, tcp_center, obj_head, obj_init_pos, peg_head_pos_init,
                   _target_pos, TARGET_RADIUS, bottom_right_corner_collision_box_1, bottom_right_corner_collision_box_2,
                   top_left_corner_collision_box_1, top_left_corner_collision_box_2):
    obj = obs[4:7]
    tcp_opened = obs[3]
    tcp_to_obj = np.linalg.norm(obj - tcp_center)
    scale = np.array([1.0, 2.0, 2.0])
    #  force agent to pick up object then insert
    obj_to_target = np.linalg.norm((obj_head - _target_pos) * scale)

    in_place_margin = np.linalg.norm((peg_head_pos_init - _target_pos) * scale)
    in_place = reward_utils.tolerance(
        obj_to_target,
        bounds=(0, TARGET_RADIUS),
        margin=in_place_margin,
        sigmoid="long_tail",
    )
    ip_orig = in_place
    brc_col_box_1 = bottom_right_corner_collision_box_1
    tlc_col_box_1 = top_left_corner_collision_box_1

    brc_col_box_2 = bottom_right_corner_collision_box_2
    tlc_col_box_2 = top_left_corner_collision_box_2
    collision_box_bottom_1 = reward_utils.rect_prism_tolerance(
        curr=obj_head, one=tlc_col_box_1, zero=brc_col_box_1
    )
    collision_box_bottom_2 = reward_utils.rect_prism_tolerance(
        curr=obj_head, one=tlc_col_box_2, zero=brc_col_box_2
    )
    collision_boxes = reward_utils.hamacher_product(
        collision_box_bottom_2, collision_box_bottom_1
    )
    in_place = reward_utils.hamacher_product(in_place, collision_boxes)

    pad_success_margin = 0.03
    object_reach_radius = 0.01
    x_z_margin = 0.005
    obj_radius = 0.0075

    object_grasped = _gripper_caging_reward(
        left_pad,
        right_pad,
        tcp_center,
        obj_init_pos,
        init_tcp,
        action,
        obj,
        object_reach_radius=object_reach_radius,
        obj_radius=obj_radius,
        pad_success_thresh=pad_success_margin,
        xz_thresh=x_z_margin,
        high_density=True,
    )
    if (
            tcp_to_obj < 0.08
            and (tcp_opened > 0)
            and (obj[2] - 0.01 > obj_init_pos[2])
    ):
        object_grasped = 1.0
    in_place_and_object_grasped = reward_utils.hamacher_product(
        object_grasped, in_place
    )
    reward = in_place_and_object_grasped

    if (
            tcp_to_obj < 0.08
            and (tcp_opened > 0)
            and (obj[2] - 0.01 > obj_init_pos[2])
    ):
        reward += 1.0 + 5 * in_place

    if obj_to_target <= 0.07:
        reward = 10.0

    return [
        reward,
        tcp_to_obj,
        tcp_opened,
        obj_to_target,
        object_grasped,
        in_place,
        collision_boxes,
        ip_orig,
    ]"""
}


import sys
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
from sac import sac_agent

import random
import time
from replay_buffer import replay_buffer
import pickle
class Worker(mp.Process):
    def __init__(self, replaybuffer, p_id , buffer_size, memory_path , model_path, target_update_interval, seed, parameters, queue, model, env_name, env, eval_env, args):
        super(Worker, self).__init__()
        self.p_id = p_id
        self.memory_path = memory_path
        self.parameters = parameters
        self.model_path = model_path
        self.target_update_interval = target_update_interval
        self.queue = queue
        self.model = model
        self.env = env
        self.env_name = env_name
        self.eval_env = eval_env
        self.args =args
        self.sac_trainer = sac_agent(self.model, self.env_name, self.env, self.eval_env, self.args, None)
        self.seed = seed
        self.All_buffer = copy.deepcopy(replaybuffer)

        t1 = time.time()
        print("load net time cost ", time.time()-t1)
        self.start_to_elite = False
        self.target_index = None
        self.dynamic_weight = 1.0
        self.current_step = 0
        temp_agent = sac_agent(self.model, self.env_name, self.env, self.eval_env, self.args, None)
        self.elite_actor = temp_agent.actor_net
        self.elite_q1 = temp_agent.qf1
        self.elite_q2 = temp_agent.qf2

        self.constraint_kl = False

    def run(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        while True:
            while self.p_id not in self.parameters:
                time.sleep(0.0001)


            t1 = time.time()
            main_process_buffer_size, train_num, added_data, next_idx, update_model_flag, log_alpha , best_index, reward_list, global_best_one  = self.parameters[self.p_id]
            del self.parameters[self.p_id]
            t2 = time.time()

            if update_model_flag and self.p_id not in best_index:
                print("Update ", self.p_id, " To Best ",best_index)
                print(self.p_id, "worker before", self.sac_trainer.log_alpha)
                print("worker get log_alpha", log_alpha, " best index", best_index)
                self.sac_trainer.actor_net.load_state_dict(torch.load(self.model_path + "/best_actor_net.pth"))
                if self.p_id != 5:
                    self.constraint_kl = True
                    self.start_to_elite = True
                    self.elite_actor.load_state_dict(torch.load(self.model_path + "/best_actor_net.pth"))
                    self.elite_q1.load_state_dict(torch.load(self.model_path + "/best_qf1.pth"))
                    self.elite_q2.load_state_dict(torch.load(self.model_path + "/best_qf2.pth"))

                    self.sac_trainer.qf1.load_state_dict(torch.load(self.model_path + "/best_qf1.pth"))
                    self.sac_trainer.qf2.load_state_dict(torch.load(self.model_path + "/best_qf2.pth"))
                    self.sac_trainer.target_qf1.load_state_dict(torch.load(self.model_path + "/best_target_qf1.pth"))
                    self.sac_trainer.target_qf2.load_state_dict(torch.load(self.model_path + "/best_target_qf2.pth"))
                    self.sac_trainer.log_alpha.data = copy.deepcopy(log_alpha)

                    self.sac_trainer.actor_optim.load_state_dict(torch.load(self.model_path + "/best_actor_optim.pth"))
                    self.sac_trainer.qf1_optim.load_state_dict(torch.load(self.model_path + "/best_qf1_optim.pth"))
                    self.sac_trainer.qf2_optim.load_state_dict(torch.load(self.model_path + "/best_qf2_optim.pth"))
                print("Done ....")

            if update_model_flag:
                if self.p_id in best_index:
                    self.constraint_kl = False
                    self.start_to_elite = False
            t3 = time.time()

            if update_model_flag and self.p_id not in best_index:
                assert  len(self.All_buffer.storge) == len(reward_list)
                sys.stdout.write(f"assert equal????? {len(self.All_buffer.storge)} == {len(reward_list)}\n")
                print("Relabel all data .....", len(self.All_buffer.storge), " ", len(reward_list))
                for index_index, data in enumerate(self.All_buffer.storge):
                    org_data, obs, action, _, obs_, done = data
                    self.All_buffer.storge[index_index]= ([], obs, action, reward_list[index_index], obs_, done )
                print("Relabel Done .....")
            t4 = time.time()


            for data in added_data:
                self.All_buffer.add(*data)

            assert  main_process_buffer_size == len(self.All_buffer.storge)

          
            self.All_buffer.next_idx = next_idx
            mean_actor_fau = []
            mean_q1_fau = []
            mean_q2_fau = []

            t5 = time.time()
            # load memory
            if train_num > 0 :
                train_time_cost_list =[]
                for _ in range(train_num):
                    temp_weight = 1.0
                    qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss, actor_fau, q1_fau, q2_fau, kl_loss, q1_kl_loss, q2_kl_loss , one_train_time_cost_list= self.sac_trainer._update_newtork(self.All_buffer, self.p_id, self.target_index, self.dynamic_weight,False, self.constraint_kl, temp_weight, self.elite_actor, self.elite_q1, self.elite_q2)

                    train_time_cost_list.append(one_train_time_cost_list)
                    mean_actor_fau.append(actor_fau)
                    mean_q1_fau.append(q1_fau)
                    mean_q2_fau.append(q2_fau)
                    # update the target network
                    if _ % self.target_update_interval == 0:
                        self.sac_trainer._update_target_network(self.sac_trainer.target_qf1, self.sac_trainer.qf1)
                        self.sac_trainer._update_target_network(self.sac_trainer.target_qf2, self.sac_trainer.qf2)
                total_train_time = np.sum(train_time_cost_list)
                train_sub_space_time = np.mean(train_time_cost_list, axis=0)/train_num

                t6 = time.time()

                time_cost_list = [t6-t5, t5-t4, t4-t3, t3-t2, t2-t1]
                self.queue.put((self.p_id, self.sac_trainer.actor_net.state_dict(), self.sac_trainer.qf1.state_dict(), self.sac_trainer.qf2.state_dict(),  self.sac_trainer.target_qf1.state_dict(), self.sac_trainer.target_qf2.state_dict(),  self.sac_trainer.actor_optim.state_dict(),  self.sac_trainer.qf1_optim.state_dict(),  self.sac_trainer.qf2_optim.state_dict(), qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss, self.sac_trainer.log_alpha.data,  np.mean(mean_actor_fau), np.mean(mean_q1_fau), np.mean(mean_q2_fau),kl_loss,  q1_kl_loss, q2_kl_loss, time_cost_list, total_train_time, train_sub_space_time))


def make_metaworld_env(cfg, seed):
    env_name = cfg.env_name
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()

    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)

    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)
"""
the tanhnormal distributions from rlkit may not stable

"""
class tanh_normal(Distribution):
    def __init__(self, normal_mean, normal_std, epsilon=1e-6, cuda=False):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.cuda = cuda
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        sample_mean = torch.zeros(self.normal_mean.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
        sample_std = torch.ones(self.normal_std.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
        z = (self.normal_mean + self.normal_std * Normal(sample_mean, sample_std).sample())
        z.requires_grad_()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

# get action_infos
class get_action_info:
    def __init__(self, pis, cuda=False):
        self.mean, self.std = pis
        self.dist = tanh_normal(normal_mean=self.mean, normal_std=self.std, cuda=cuda)

    # select actions
    def select_actions(self, exploration=True, reparameterize=True):
        if exploration:
            if reparameterize:
                actions, pretanh = self.dist.rsample(return_pretanh_value=True)
                return actions, pretanh
            else:
                actions = self.dist.sample()
        else:
            actions = torch.tanh(self.mean)
        return actions

    def get_log_prob(self, actions, pre_tanh_value):
        log_prob = self.dist.log_prob(actions, pre_tanh_value=pre_tanh_value)
        return log_prob.sum(dim=1, keepdim=True)

# env wrapper
class env_wrapper:
    def __init__(self, env, args):
        self._env = env
        self.args = args
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self):
        self.timesteps = 0
        obs = self._env.reset()
        return obs

    def step(self, action):
        # revise the correct action range
        obs, reward, done, info = self._env.step(action)
        # increase the timesteps
        self.timesteps += 1
        if self.timesteps >= self.args.episode_length:
            done = True
        return obs, reward, done, info
    
    def render(self):
        """
        to be Implemented during execute the demo
        """
        self._env.render()

    def seed(self, seed):
        """
        set environment seeds
        """
        self._env.seed(seed)

# record the reward info of the dqn experiments
class reward_recorder:
    def __init__(self, history_length=100):
        self.history_length = history_length
        # the empty buffer to store rewards
        self.buffer = [0.0]
        self._episode_length = 1

    # add rewards
    def add_rewards(self, reward):
        self.buffer[-1] += reward

    # start new episode
    def start_new_episode(self):
        if self.get_length >= self.history_length:
            self.buffer.pop(0)
        # append new one
        self.buffer.append(0.0)
        self._episode_length += 1

    # get length of buffer
    @property
    def get_length(self):
        return len(self.buffer)

    @property
    def mean(self):
        return np.mean(self.buffer)

    # get the length of total episodes
    @property
    def num_episodes(self):
        return self._episode_length
