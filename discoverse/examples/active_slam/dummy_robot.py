import cv2
import glfw
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import random

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig


##########################################################################

output_dir = 'output_images'

def generate_camera_lookat_pairs(cam_pos_range, lookat_pos_ranges, num_samples=300):
    """
    生成随机相机位置 (cam-pos) 和对应的观察目标位置 (lookat-pos) 组合。

    :param cam_pos_range: 一个字典，定义相机位置的 x, y, z 范围，例如：
                         {"x": (x_min, x_max), "y": (y_min, y_max), "z": (z_min, z_max)}
    :param lookat_pos_ranges: 一个列表，每个元素是字典，定义 lookat 位置的范围，例如：
                              [{"x": (x_min, x_max), "y": (y_min, y_max), "z": (z_min, z_max)}, {...}, ...]
    :param num_samples: 生成的 (cam-pos, lookat-pos) 组合数量
    :return: 一个包含 (cam-pos, lookat-pos) 组合的列表
    """
    pairs = []

    for _ in range(num_samples):
        # 随机生成相机位置 cam-pos
        cam_x = np.random.uniform(*cam_pos_range["x"])
        cam_y = np.random.uniform(*cam_pos_range["y"])
        cam_z = np.random.uniform(*cam_pos_range["z"])
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # 随机选择一个 lookat 范围
        lookat_range = random.choice(lookat_pos_ranges)

        # 从选择的 lookat 范围中采样 lookat 位置
        lookat_x = np.random.uniform(*lookat_range["x"])
        lookat_y = np.random.uniform(*lookat_range["y"])
        lookat_z = np.random.uniform(*lookat_range["z"])
        lookat_pos = np.array([lookat_x, lookat_y, lookat_z])

        # 组合成一对
        pairs.append((cam_pos, lookat_pos))

    return pairs

# 相机位置范围  0.1 0.5 0.947
cam_pos_range = {
    "x": (-0.7, 0.6),
    "y": (0.2, 0.7),
    "z": (0.6, 1.4)
}

# 多个 lookat 位置范围  book:-0.4,0.98,0.947  ||||  arm:0.3 0.92 0.71  |||||  cabinet: 0.915 0.58 0.01
lookat_pos_ranges = [
    {"x": (-0.3, -0.5), "y": (0.98, 0.99), "z": (0.907, 1.0)},
    {"x": (0.3, 0.35), "y": (0.92, 0.93), "z": (0.76, 0.82)},
    {"x": (0.915, 0.916), "y": (0.38, 0.78), "z": (0.45, 0.95)}
]

# 生成 300 组 (cam-pos, lookat-pos)
pos_pairs = generate_camera_lookat_pairs(cam_pos_range, lookat_pos_ranges, num_samples=300)


##########################################################################

class DummyRobotConfig(BaseConfig):
    y_up = False
    robot_name = "dummy_robot"

class DummyRobot(SimulatorBase):
    Tmat_Zup2Yup = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1]
    ])
    move_step_ratio = 1.0
    pitch_joint_id = 0
    def __init__(self, config: DummyRobotConfig):
        super().__init__(config)

    def updateControl(self, action):
        move_step = action * self.config.timestep
        self.base_move(*tuple(move_step))

    def get_base_pose(self):
        return self.mj_model.body(self.config.robot_name).pos.copy(), self.mj_model.body(self.config.robot_name).quat.copy()

    def getObservation(self):
        rgb_cam_pose_lst = [self.getCameraPose(id) for id in self.config.obs_rgb_cam_id]
        depth_cam_pose_lst = [self.getCameraPose(id) for id in self.config.obs_depth_cam_id]
        if self.config.y_up:
            for pose_lst in [rgb_cam_pose_lst, depth_cam_pose_lst]:
                for i, (xyz, quat_wxyz) in enumerate(pose_lst):
                    Tmat = np.eye(4)
                    Tmat[:3, :3] = Rotation.from_quat(quat_wxyz[[1,2,3,0]]).as_matrix()
                    Tmat[:3, 3] = xyz[:]
                    new_Tmat = np.linalg.inv(self.Tmat_Zup2Yup) @ Tmat
                    pose_lst[i] = (new_Tmat[:3, 3], Rotation.from_matrix(new_Tmat[:3, :3]).as_quat()[[3,0,1,2]])
        self.obs = {
            "rgb_cam_posi"   : rgb_cam_pose_lst,
            "depth_cam_posi" : depth_cam_pose_lst,
            "rgb_img"        : self.img_rgb_obs_s,
            "depth_img"      : self.img_depth_obs_s,
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs    

    def checkTerminated(self):
        return False
    
    def getReward(self):
        return None

    def on_mouse_move(self, window, xpos, ypos):
        if self.cam_id == -1:
            super().on_mouse_move(window, xpos, ypos)
        else:
            if self.mouse_pressed['left']:
                self.camera_pose_changed = True
                height = self.config.render_set["height"]
                dx = float(xpos) - self.mouse_pos["x"]
                dy = float(ypos) - self.mouse_pos["y"]
                self.base_move(0.0, 0.0, -dx/height, 0.0)
                self.move_camera_pitch(dy/height)

            self.mouse_pos['x'] = xpos
            self.mouse_pos['y'] = ypos

    def move_camera_pitch(self, d_pitch):
        self.mj_data.qpos[self.pitch_joint_id] += d_pitch

    def base_move(self, dx_local, dy_local, angular_z, pitch_local):
        posi_, quat_wxyz = self.get_base_pose()
        yaw = Rotation.from_quat(quat_wxyz[[1,2,3,0]]).as_euler("zyx")[0]
        yaw += angular_z
        if yaw > np.pi:
            yaw -= 2. * np.pi
        elif yaw < -np.pi:
            yaw += 2. * np.pi
        self.mj_model.body(self.config.robot_name).quat[:] = Rotation.from_euler("zyx", [yaw, 0, 0]).as_quat()[[3,0,1,2]]

        base_posi = self.mj_model.body(self.config.robot_name).pos
        base_posi[0] += dx_local * np.cos(yaw) - dy_local * np.sin(yaw)
        base_posi[1] += dx_local * np.sin(yaw) + dy_local * np.cos(yaw)
        self.mj_data.qpos[0] += pitch_local
        self.mj_data.qpos[0] = np.clip(self.mj_data.qpos[0], -np.pi/2., np.pi/2.)

    def teleopProcess(self):
        pass

    def on_key(self, window, key, scancode, action, mods):
        super().on_key(window, key, scancode, action, mods)

        is_shift_pressed = (mods & glfw.MOD_SHIFT)
        move_step_ratio = 3.0 if is_shift_pressed else 1.0

        step = 1.0 / float(self.config.render_set["fps"]) * 5. * move_step_ratio
        dx = 0.0
        dy = 0.0
        dz = 0.0
        dpitch = 0.0

        if action == glfw.PRESS or action == glfw.REPEAT:
            # 同时监控多个按键
            if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
                dx = step
            elif glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
                dx = -step
            
            if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
                dy = step
            elif glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
                dy = -step

            if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
                dpitch = 0.05
            elif glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
                dpitch = -0.05

            if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
                self.mj_model.body(self.config.robot_name).pos[2] += 0.02
            elif glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
                self.mj_model.body(self.config.robot_name).pos[2] -= 0.02

        self.base_move(dx, dy, dz, dpitch)

    def printHelp(self):
        super().printHelp()
        print("-------------------------------------")
        print("dummy robot control:")
        print("w/s down    : move forward/backward")
        print("a/d right : move left/right")
        print("q/e : pitch up/down")
        print("arrow up/down : height up/down")
        print("left mouse drag : camera move yaw and pitch")
        print("press shift key to move faster")

if __name__ == "__main__":
    cfg = DummyRobotConfig()
    cfg.y_up = False
    
    camera_height = 1.0 #m

    dummy_robot_cam_id = 0
    cfg.obs_rgb_cam_id   = [dummy_robot_cam_id]
    cfg.obs_depth_cam_id = [dummy_robot_cam_id]

    cfg.render_set["fps"] = 60
    cfg.render_set["width"] = 1920
    cfg.render_set["height"] = 1080
    cfg.timestep = 1./cfg.render_set["fps"]
    cfg.decimation = 1
    cfg.mjcf_file_path = "mjcf/dummy_robot.xml"

    cfg.use_gaussian_renderer = True
    cfg.gs_model_dict["background"] = "scene/Air11F/air_11f_3.ply"
    #cfg.gs_model_dict["background"] = "scene/Air12F/air_12f.ply"
    #cfg.gs_model_dict["background"] = "scene/Air12F/air_12f.ply"
    robot = DummyRobot(cfg)
    robot.cam_id = dummy_robot_cam_id

    robot.mj_model.body("dummy_robot").pos[2] = camera_height

    action = np.zeros(4)
    # if z_up:
        # action[0] : lineal_velocity_x  local m    不论yup还是zup，始终为朝前为正方向
        # action[1] : lineal_velocity_y  local m    不论yup还是zup，始终为朝左为正方向
        # action[2] : angular_velocity_z rad        不论yup还是zup，始终为从上向下看逆时针旋转为正方向
        # action[3] : camera_pitch       rad        不论yup还是zup，始终为镜头俯仰
    # elif y_up:
        # action[0] : lineal_velocity_x   local m
        # action[1] : lineal_velocity_-z  local m
        # action[2] : angular_velocity_y  rad
        # action[3] : camera_pitch        rad 

    obs = robot.reset()
    rgb_cam_posi = obs["rgb_cam_posi"]
    depth_cam_posi = obs["depth_cam_posi"]
    rgb_img_0 = obs["rgb_img"][0]
    depth_img_0 = obs["depth_img"][0]

    print("rgb_cam_posi    = ", rgb_cam_posi)
    # [[posi_x, posi_y, posi_z], [quat_w, quat_x, quat_y, quat_z]]
    # [(array([0., 0., 1.]), array([ 0.49999816,  0.50000184, -0.5       , -0.5       ]))]

    print("depth_cam_posi  = ", depth_cam_posi)
    # [[posi_x, posi_y, posi_z], [quat_w, quat_x, quat_y, quat_z]]
    # [(array([0., 0., 1.]), array([ 0.49999816,  0.50000184, -0.5       , -0.5       ]))]

    print("rgb_img.shape   = ", rgb_img_0.shape  , "rgb_img.dtype    = ", rgb_img_0.dtype)
    # rgb_img.shape   =  (1080, 1920, 3) rgb_img.dtype    =  uint8

    print("depth_img.shape = ", depth_img_0.shape, "depth_img.dtype  = ", depth_img_0.dtype)
    # depth_img.shape =  (1080, 1920, 1) depth_img.dtype  =  float32

    robot.printHelp()

    img=robot.get_camera_image_direct("eye_side",changed_xyz=[0.5,0.5,0.947],lookat_position=[2,0,0])
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    # for i in range(300):
    #     img = robot.get_camera_image_direct(camera_name="eye_side",changed_xyz=pos_pairs[i][0],lookat_position=pos_pairs[i][1])
    #     print(f"Pair {i+1}: Camera Pos {pos_pairs[i][0]}, Lookat Pos {pos_pairs[i][1]}")
    #     img_filename = f"{output_dir}/camera_image_X{i:.1f}.png"
    #     plt.imsave(img_filename, img)
    #     print(f"Captured Image {i} from Position: {pos_pairs[i][0]} look at {pos_pairs[i][1]}" ) 

    while robot.running:
        obs, _, _, _, _ = robot.step(action)