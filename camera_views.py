import argparse
import time

import numpy as np
from PIL import Image

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import CameraMover


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="NutAssemblySquare")
    parser.add_argument("--robots", nargs="+", type=str, default="Sawyer", help="Which robot(s) to use in the env")
    parser.add_argument("--image_save_path", type=str, default="")
    args = parser.parse_args()

    CAMERA_NAME = "agentview"
    image_save_path = args.image_save_path
    assert image_save_path != "", "Please provide a valid save path."

        # make the environment
    env = robosuite.make(
        args.env,
        robots=args.robots,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=100,
    )
    env.reset()

    # Create the camera mover
    camera_mover = CameraMover(
        env=env,
        camera=CAMERA_NAME,
    )

    for i, theta in enumerate(np.arange(0., 2 * np.pi, 2 * np.pi / 100)):
    
        position=np.array([np.cos(theta) - 0.5, np.sin(theta), 1.35]) 
        rotation=T.mat2quat(T.euler2mat(np.array([0., np.pi / 4, (np.pi / 2) + theta])))
        print(f"position: {position}")
        print(f"rotation: {rotation}")

        # Make sure we're using the camera that we're modifying
        camera_id = env.sim.model.camera_name2id(CAMERA_NAME)
        #env.viewer.set_camera(camera_id=camera_id)

        camera_mover.set_camera_pose(pos=position, quat=rotation)

        # just spin to let user interact with glfw window
        spin_count = 0
        while True:
            action = np.zeros(env.action_dim)
            obs, reward, done, _ = env.step(action)
            #env.render()
            spin_count += 1
            if spin_count % 50 == 0:
                # convert from world coordinates to file coordinates (xml subtree)
                camera_pos, camera_quat = camera_mover.get_camera_pose()
                world_camera_pose = T.make_pose(camera_pos, T.quat2mat(camera_quat))

                print("pos", "{} {} {}".format(camera_pos[0], camera_pos[1], camera_pos[2]))
                print("quat", "{} {} {} {}".format(camera_quat[0], camera_quat[1], camera_quat[2], camera_quat[3]))
                print(f"state dict: {obs['agentview_image']}")
                im = Image.fromarray(obs['agentview_image'])
                print("Saving image...")
                im.save(f"{image_save_path}/view_{i}.jpg")
                break