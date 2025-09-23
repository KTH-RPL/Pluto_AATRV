from scipy.spatial.transform import Rotation as R

# Example: 90° rotation about z-axis
R_matrix = [
    [0.9510566,  0.3090167,  0.0000000],
    [-0.3090167,  0.9510566,  0.0000000],
    [0.0000000,  0.0000000,  1.0000000]
]

r = R.from_matrix(R_matrix)
rpy = r.as_euler('xyz', degrees=False)  # 'xyz' -> roll, pitch, yaw

print("Roll, Pitch, Yaw (in degrees):", rpy)

'''import numpy as np
from scipy.spatial.transform import Rotation as R

# Original quaternion [x, y, z, w]
q = [0.9998805501718303, -0.005631361428549012, -0.0033964292086203895, 0.013987044904833533]

# Convert to rotation object
rot = R.from_quat(q)

# Get Euler angles (in radians), using 'xyz' convention
euler = rot.as_euler('xyz', degrees=True)
print("Original Euler angles (roll, pitch, yaw):", euler)

# Add 180° to pitch
euler[1] += 180.0

# Wrap pitch back into [-180, 180] if needed
euler[1] = (euler[1] + 180) % 360 - 180

# Convert back to quaternion
rot_new = R.from_euler('xyz', euler, degrees=True)
q_new = rot_new.as_quat()

print("New quaternion [x, y, z, w]:", q_new)'''

