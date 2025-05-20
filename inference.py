import os
import time
import cv2
import numpy as np
import torch
from models.ppo_mae import PPO_MAE
import pandas as pd
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from models.ppo_mae import PPO_MAE
import envs
import traceback

# 显式设置 MuJoCo 渲染后端
os.environ["MUJOCO_GL"] = "egl"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
    # 正确提取最后一帧（frame stack 的最后一帧）
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

def save_parallel_tactile_info(env_data_lists, save_dir, env_idx, episode_idx):
    """
    保存单个环境的触觉数据和相关信息
    Args:
        env_data_lists: 包含单个环境数据的字典
        save_dir: 保存目录
        env_idx: 环境索引
        episode_idx: 轨迹索引
    """
    tactile_list = env_data_lists['tactile']
    if not tactile_list:  # 空列表直接返回
        return

    # 转换为numpy数组
    tactile_tensor = np.stack(tactile_list, axis=0)  # (step, 1, 4, 6, 32, 32)
    id_array = np.array([x[0] if isinstance(x, np.ndarray) else x for x in env_data_lists['id']])
    time_array = np.array(env_data_lists['time'])
    pos_array = np.array(env_data_lists['pos'])
    action_array = np.array(env_data_lists['action'])
    reward_array = np.array(env_data_lists['reward'])

    # 创建环境专属保存目录
    env_save_dir = os.path.join(save_dir, f"env_{env_idx}")
    ensure_dir(env_save_dir)
    
    # 保存npz
    np.savez(os.path.join(env_save_dir, f"episode_{episode_idx}.npz"),
             tactile=tactile_tensor,
             id=id_array,
             time=time_array,
             pos=pos_array,
             action=action_array,
             reward=reward_array)

    # 准备CSV数据
    step_num = len(tactile_list)
    csv_data = {
        "step": np.arange(step_num),
        "id": id_array,
        "time": time_array,
    }

    # 处理位置数据
    for i in range(pos_array.shape[1]):
        csv_data[f"pos_{i}"] = pos_array[:, i]

    # 只取最后一帧计算触觉统计量
    tactile_last = tactile_tensor[:, -1]  # (step, 6, 32, 32)
    force_names = ['right_x', 'right_y', 'right_z', 'left_x', 'left_y', 'left_z']
    
    # 计算6个方向的力
    force_means = []
    for i in range(6):
        force = tactile_last[:, i].mean(axis=(1,2))
        force_means.append(force)
        csv_data[f"force_{force_names[i]}"] = force

    # 计算合力
    right_total = np.sqrt(np.sum(np.square(force_means[:3]), axis=0))
    left_total = np.sqrt(np.sum(np.square(force_means[3:]), axis=0))
    csv_data["right_total_force"] = right_total
    csv_data["left_total_force"] = left_total
    csv_data["total_force"] = right_total + left_total

    # 处理动作和奖励
    if action_array.ndim > 1:
        for i in range(action_array.shape[1]):
            csv_data[f"action_{i}"] = action_array[:, i]
    else:
        csv_data["action"] = action_array
    
    csv_data["reward"] = reward_array
    csv_data["cumulative_reward"] = np.cumsum(reward_array)

    # 保存CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(env_save_dir, f"episode_{episode_idx}.csv"), index=False)

def main():
    try:
        # 显式设置 MuJoCo 渲染后端
        os.environ["MUJOCO_GL"] = "egl"

        # 加载模型
        print("加载模型中...")
        model_path = "./rl_model_2999520_steps.zip"
        model = PPO_MAE.load(model_path)
        print("模型加载完成")

        # 并行环境设置
        n_envs = 4
        episodes_per_env = 100
        
        zero_seed = 42
        print(f"设置随机种子: {zero_seed}")
        set_random_seed(zero_seed)

        # 创建环境
        print(f"创建 {n_envs} 个环境...")
        env_fns = [
            lambda i=i: envs.make_env(
                "tactile_envs/Insertion-v0",
                i,
                seed=zero_seed+i,
                state_type="vision_and_touch",
                objects=["square", "triangle", "horizontal", "vertical", "trapezoidal", "rhombus"],
                holders=["holder1", "holder2", "holder3"],
                camera_idx=0,
                frame_stack=4,
                no_rotation=True,
                use_latch=True,
                render_mode=None  # 禁用子进程中的渲染
            )()
            for i in range(n_envs)
        ]

        env = SubprocVecEnv(env_fns)
        print("环境创建完成")

        print("重置环境...")
        obs = env.reset()
        print("环境重置完成")

        # 准备输出目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = f"inference_logs/{timestamp}"
        frames_dir = os.path.join(base_dir, "frames")
        ensure_dir(base_dir)
        ensure_dir(frames_dir)
        print(f"输出目录创建完成: {base_dir}")

        # 初始化进度条
        pbar = tqdm(total=n_envs * episodes_per_env, desc="收集数据")
        env_episode_counts = [0] * n_envs
        total_episodes = 0

        # 初始化每个环境的数据存储
        env_data = [
            {
                'tactile': [],
                'id': [],
                'time': [],
                'pos': [],
                'action': [],
                'reward': []
            }
            for _ in range(n_envs)
        ]

        while min(env_episode_counts) < episodes_per_env:
            print(env_episode_counts)
            dones = [False] * n_envs
            step_counts = [0] * n_envs
            max_steps_per_episode = 100  # 设置每个环境的最大步数限制

            while not all(dones):
                # 将 obs 转换为每个环境的独立字典列表
                obs_list = [
                    {key: obs[key][env_idx] for key in obs.keys()}  # 提取每个环境的观测
                    for env_idx in range(n_envs)
                ]

                # 对每个环境分别进行动作预测
                actions = []
                for env_idx in range(n_envs):
                    if not dones[env_idx]:
                        action, _ = model.predict(obs_list[env_idx], deterministic=True)
                        actions.append(action)
                    else:
                        # 占位动作：用 zero 或上一次合法动作，似乎只是为了更新结束的最后一个多余步合法
                        action = np.zeros(model.action_space.shape, dtype=model.action_space.dtype)
                        actions.append(action)

                # 执行环境步进
                obs, rewards, dones_step, infos = env.step(actions)
                

                # 更新每个环境的状态
                for env_idx in range(n_envs):                    
                    if not dones[env_idx]:  # 仅更新未完成的环境
                        step_counts[env_idx] += 1
                        
                        # 收集数据
                        tactile = obs['tactile'][env_idx]
                        if tactile is not None and np.any(tactile):
                            env_data[env_idx]['tactile'].append(tactile)
                            env_data[env_idx]['id'].append(infos[env_idx].get('id', -1))
                            env_data[env_idx]['time'].append(infos[env_idx].get('time', -1))
                            env_data[env_idx]['pos'].append(infos[env_idx].get('pos', -1))
                            env_data[env_idx]['action'].append(actions[env_idx])
                            env_data[env_idx]['reward'].append(rewards[env_idx])

                        # 如果达到最大步数，强制标记为 done
                        if step_counts[env_idx] >= max_steps_per_episode:
                            dones[env_idx] = True

                        # 如果环境完成，更新计数器
                        if dones_step[env_idx]:
                            dones[env_idx] = True
                            env_episode_counts[env_idx] += 1

                            # 调用 save_parallel_tactile_info 保存数据
                            save_parallel_tactile_info(
                                env_data[env_idx],
                                base_dir,
                                env_idx,
                                env_episode_counts[env_idx] - 1  # 当前 episode 索引
                            )

                            # 清空当前环境的数据
                            env_data[env_idx] = {
                                'tactile': [],
                                'id': [],
                                'time': [],
                                'pos': [],
                                'action': [],
                                'reward': []
                            }


                # 渲染逻辑在主进程中执行
                for env_idx in range(n_envs):
                    if not dones[env_idx] and env_episode_counts[env_idx] < episodes_per_env:
                        try:
                            # 渲染当前环境
                            img = env.env_method("render", indices=env_idx)[0]
                            if img is not None:
                                img = (img * 255).astype(np.uint8)
                                env_frames_dir = os.path.join(frames_dir, f"env_{env_idx}")
                                ensure_dir(env_frames_dir)
                                img_path = os.path.join(
                                    env_frames_dir,
                                    f"ep_{env_episode_counts[env_idx]}_step_{step_counts[env_idx]:04d}.png"
                                )
                                cv2.imwrite(img_path, img)
                                if step_counts[env_idx] == 0:
                                    print(f"保存图片到: {img_path}")
                        except Exception as render_error:
                            print(f"环境 {env_idx} 渲染失败: {render_error}")
                            traceback.print_exc()
                            continue

        pbar.close()
        print(f"数据收集完成，共 {total_episodes} 条")
        print(f"数据保存在：{base_dir}")

    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        if 'pbar' in locals():
            pbar.close()

if __name__ == "__main__":
    main()