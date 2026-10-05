def decode_rotation(delta_rot):
    return euler_angles_to_matrix(delta_rot, "XYZ")
