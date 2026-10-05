def encode_rotation(delta_pose):
    return compact_axis_angle_from_quaternion(delta_pose.q)
