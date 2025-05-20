import os
import time
import cv2
import numpy as np
import torch
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from models.ppo_mae import PPO_MAE
import envs

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def show_and_save_tactile(tactile, step, save_dir, size=(240, 240), max_shear=0.05, max_pressure=0.1, name='tactile'):
    """
    tactile: shape (4, 6, 32, 32)
    step: 当前步数
    save_dir: 保存目录
    输出格式：2行 × 4列（上：右手；下：左手）
    """
    tactile = np.asarray(tactile)
    tactile = tactile[0]  # shape: (4, 6, 32, 32)
    assert tactile.shape[1] == 6, "每帧有6个通道（右xyz + 左xyz）"

    frame_stack = tactile.shape[0]
    imgs_right = []
    imgs_left = []

    for f in range(frame_stack):
        t = tactile[f]  # shape: (6, 32, 32)
        right = t[:3]   # shape: (3, 32, 32)
        left = t[3:]    # shape: (3, 32, 32)

        def render_vector_field(s):  # s: (3, 32, 32)
            nx, ny = s.shape[2], s.shape[1]
            loc_x = np.linspace(0, size[1], nx)
            loc_y = np.linspace(size[0], 0, ny)
            img = np.zeros((size[0], size[1], 3))
            for i in range(len(loc_x)):
                for j in range(len(loc_y)):
                    dir_x = np.clip(s[0, j, i] / max_shear, -1, 1) * 20
                    dir_y = np.clip(s[1, j, i] / max_shear, -1, 1) * 20
                    color = np.clip(s[2, j, i] / max_pressure, 0, 1)
                    r = color
                    g = 1 - color
                    cv2.arrowedLine(img, (int(loc_x[i]), int(loc_y[j])),
                                    (int(loc_x[i] + dir_x), int(loc_y[j] - dir_y)),
                                    (0, g, r), 2, tipLength=0.5)
            return img

        img_r = render_vector_field(right)
        img_l = render_vector_field(left)
        imgs_right.append(img_r)
        imgs_left.append(img_l)

    # 横向拼每组，纵向拼两组
    top_row = np.concatenate(imgs_right, axis=1)
    bot_row = np.concatenate(imgs_left, axis=1)
    final_img = np.concatenate([top_row, bot_row], axis=0)

    # 显示 + 保存
    cv2.imshow(name, final_img)
    save_path = os.path.join(save_dir, f"tactile_{step:04d}.png")
    cv2.imwrite(save_path, (final_img * 255).astype(np.uint8))
    

    # 灰度图归一化 → uint8
    # ✅ 重新组织整个 (4, 6, 32, 32) 成 6行 × 4列 拼图
    grid_rows = []

    for ch in range(6):
        row_imgs = []
        for f in range(frame_stack):
            ch_img = tactile[f, ch]  # shape: (32, 32)
            ch_img = ((ch_img - ch_img.min()) / (ch_img.ptp() + 1e-6) * 255).astype(np.uint8)
            row_imgs.append(ch_img)
        row_concat = np.concatenate(row_imgs, axis=1)  # 一行：4帧并排
        grid_rows.append(row_concat)

    raw_img_grid = np.concatenate(grid_rows, axis=0)  # 多行堆叠：6个通道

    # 保存 raw 图像
    raw_path = os.path.join(save_dir, f"tactile_{step:04d}_raw.png")
    cv2.imwrite(raw_path, raw_img_grid)

    return tactile  # 返回原始张量

def save_tactile_and_info(tactile_tensor_list, id_list, time_list, pos_list, 
                         action_list, reward_list, save_dir):
    """保存触觉张量及相关信息为 npz 和 csv"""
    # 转换为numpy数组并验证维度
    tactile_tensor = np.stack(tactile_tensor_list, axis=0)  # (step, 1, 4, 6, 32, 32)
    assert tactile_tensor.ndim == 6, f"触觉张量维度错误: {tactile_tensor.shape}"
    
    id_array = np.array([x[0] if isinstance(x, np.ndarray) else x for x in id_list])
    time_array = np.array(time_list)
    pos_array = np.array(pos_list)      # 应该是 (step, 3)
    action_array = np.array(action_list)
    reward_array = np.array(reward_list)
    
    # 验证维度
    assert pos_array.shape[1] == 3, f"位置向量维度错误: {pos_array.shape}"
    
    # 验证长度一致性
    step_num = tactile_tensor.shape[0]
    arrays = [id_array, time_array, pos_array, action_array, reward_array]
    for arr in arrays:
        assert len(arr) == step_num, f"数组长度不一致: {len(arr)} != {step_num}"

    # 保存原始数据为npz
    np.savez(os.path.join(save_dir, "tactile_with_info.npz"),
             tactile_tensor=tactile_tensor,
             id=id_array,
             time=time_array,
             pos=pos_array,
             action=action_array,
             reward=reward_array)

    # 准备CSV数据 - 确保所有数据都是一维的
    csv_data = {
        "step": np.arange(step_num),
        "id": [x[0] if isinstance(x, (np.ndarray, list)) else x for x in id_array],
        "time": time_array.flatten(),  # 确保是一维
    }

    # pos展开为多列
    if pos_array.ndim == 2:
        for i in range(pos_array.shape[1]):
            csv_data[f"pos_{i}"] = pos_array[:, i].flatten()
    else:
        csv_data["pos"] = pos_array.flatten()


    # 计算触觉统计量
    right_tactile = tactile_tensor[..., :3, :, :]  # 右爪 xyz
    left_tactile = tactile_tensor[..., 3:, :, :]   # 左爪 xyz
    right_means = right_tactile.mean(axis=(1,2,4,5))  # (step, 3)
    left_means = left_tactile.mean(axis=(1,2,4,5))    # (step, 3)
    
    # 计算合力
    right_force = np.linalg.norm(right_means, axis=1)  # (step,)
    left_force = np.linalg.norm(left_means, axis=1)    # (step,)

    # 更新触觉相关的csv列
    for i in range(3):
        csv_data[f"right_tactile_{chr(120+i)}"] = right_means[:, i].flatten()
        csv_data[f"left_tactile_{chr(120+i)}"] = left_means[:, i].flatten()
    csv_data["right_force"] = right_force.flatten()
    csv_data["left_force"] = left_force.flatten()

    # 计算触觉统计量 - 只取最后一帧
    tactile_last = tactile_tensor[:, 0, -1]  # shape: (step, 6, 32, 32)
    
    # 分别计算6个方向的合力
    force_means = []
    force_names = ['right_x', 'right_y', 'right_z', 'left_x', 'left_y', 'left_z']
    
    for i in range(6):
        # 计算每个方向的合力 (每个通道的平均值)
        force = tactile_last[:, i].mean(axis=(1,2))  # shape: (step,)
        force_means.append(force)
        csv_data[f"force_{force_names[i]}"] = force.flatten()
    
    # 计算右爪和左爪的总合力
    right_total = np.sqrt(np.sum(np.square(force_means[:3]), axis=0))  # 右爪xyz合力
    left_total = np.sqrt(np.sum(np.square(force_means[3:]), axis=0))   # 左爪xyz合力
    
    csv_data["right_total_force"] = right_total.flatten()
    csv_data["left_total_force"] = left_total.flatten()
    csv_data["total_force"] = (right_total + left_total).flatten()

    # 添加action和reward
    if action_array.ndim == 3:  # (56, 1, 3)
        action_array = action_array.squeeze(1)
        for i in range(action_array.shape[1]):
            csv_data[f"action_{i}"] = action_array[:, i].flatten()
    else:
        csv_data["action"] = action_array.flatten()
    
    csv_data["reward"] = reward_array.flatten()
    csv_data["cumulative_reward"] = np.cumsum(reward_array).flatten()
    print(len(id_array), len(time_array), len(pos_array), len(action_array), len(reward_array), len(pos_array))

    # 保存为CSV
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(save_dir, "tactile_with_info.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV数据已保存到：{csv_path}")

def main():
    # 加载模型
    model_path = "./rl_model_2999520_steps.zip"
    model = PPO_MAE.load(model_path)

    # 创建环境
    env_config = {"use_latch": True}
    objects = ["square", "triangle", "horizontal", "vertical", "trapezoidal", "rhombus"]
    holders = ["holder1", "holder2", "holder3"]

    env_fns = [
        lambda: envs.make_env(
            "tactile_envs/Insertion-v0",
            0,
            seed=42,
            state_type="vision_and_touch",
            objects=["square"],
            holders=["holder1"],
            camera_idx=0,
            frame_stack=4,
            no_rotation=True,
            **env_config
        )()
    ]

    env = DummyVecEnv(env_fns)
    obs = env.reset()

    # 输出路径准备
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = f"inference_logs/{timestamp}"
    frames_dir = os.path.join(base_dir, "frames")
    tactile_dir = os.path.join(base_dir, "tactile")
    ensure_dir(frames_dir)
    ensure_dir(tactile_dir)

    # 初始化收集变量
    step_count = 0
    done = False
    
    pos_list = []
    tactile_tensor_list = []
    id_list = []
    time_list = []
    action_list = []
    reward_list = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        tactile = obs['tactile']
        if tactile is not None and np.any(tactile):
            tactile_tensor_list.append(tactile)
            id_list.append(info[0].get('id', -1))
            time_list.append(info[0].get('time', -1))
            pos_list.append(info[0].get('pos', -1))
            action_list.append(action)
            reward_list.append(reward)

        img = (env.get_attr('render')[0]()*255).astype(np.uint8)  # 获取当前渲染图像

        # 保存视觉图像
        cv2.imwrite(f"{frames_dir}/frame_{step_count:04d}.png", img)

        # 记录动作和奖励
        step_count += 1

    # 保存触觉张量及相关信息
    save_tactile_and_info(
        tactile_tensor_list,
        id_list,
        time_list,
        pos_list,
        action_list,
        reward_list,
        tactile_dir
    )

    print(f"推理完成，总步数：{step_count}")
    print(f"数据已保存到：{base_dir}")

    # 可选：合成视频
    try:
        os.system(f"ffmpeg -framerate 10 -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {base_dir}/inference_video.mp4")
        print("视频已生成")
    except Exception as e:
        print("ffmpeg 视频合成失败：", e)

if __name__ == "__main__":
    main()
