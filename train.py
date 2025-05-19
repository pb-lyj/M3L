import argparse
import torch

# 导入 Stable Baselines3 的 PPO 算法与工具
from stable_baselines3 import PPO
from models.ppo_mae import PPO_MAE

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList

# 注册环境（tactile_envs 是自定义的插孔环境）
import tactile_envs
import envs
from utils.callbacks import create_callbacks
from models.pretrain_models import VTT, VTMAE, MAEPolicy

# 工具函数：将字符串转为布尔值（用于 argparse）
def str2bool(v):
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    raise ValueError(f"boolean argument should be either True or False (got {v})")

# 主函数：训练流程主入口
def main():

    # ============ 【资源使用限制设置】 ============
    import os
    import torch

    max_threads = os.cpu_count()
    # 限制 PyTorch 使用的线程数，避免 CPU 资源占满
    torch.set_num_threads(max_threads // 2)

    print(f"[Info] Max CPU threads: {max_threads}, using: {max_threads // 2}")

    # 可选：限制可见 GPU，避免显存爆满（你可以取消注释使用）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # if torch.cuda.is_available():
    #     for i in range(torch.cuda.device_count()):
    #         torch.cuda.set_per_process_memory_fraction(0.5, i)
    # ============================================

    # ============ 【参数设置】 ============
    # program name
    parser = argparse.ArgumentParser("M3L")

    # 训练基础参数
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=int(1e5))
    parser.add_argument("--eval_every", type=int, default=int(2e5))
    parser.add_argument("--total_timesteps", type=int, default=int(3e6))

    # 日志与 WandB 设置
    parser.add_argument("--wandb_dir", type=str, default="./wandb/")
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    # 环境配置
    # Training M3L: `MUJOCO_GL='egl' python train.py --env tactile_envs/Insertion-v0`
    parser.add_argument(
        "--env",
        type=str,
        default="tactile_envs/Insertion-v0",
        choices=[
            "tactile_envs/Insertion-v0",
            "Door",
            "HandManipulateBlockRotateZFixed-v1",
            "HandManipulateEggRotateFixed-v1",
            "HandManipulatePenRotateFixed-v1",
        ],
    )
    parser.add_argument("--n_envs", type=int, default=64)
    parser.add_argument(
        "--state_type",
        type=str,
        default="vision_and_touch",
        choices=["vision", "touch", "vision_and_touch"],
    )
    parser.add_argument("--norm_reward", type=str2bool, default=True)
    parser.add_argument("--use_latch", type=str2bool, default=True)

    # 相机与堆叠帧设置
    parser.add_argument("--camera_idx", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--no_rotation", type=str2bool, default=True)

    # MAE 表征网络设置
    parser.add_argument("--representation", type=str2bool, default=True)
    parser.add_argument("--early_conv_masking", type=str2bool, default=True)
    parser.add_argument("--dim_embedding", type=int, default=256)
    parser.add_argument("--use_sincosmod_encodings", type=str2bool, default=True)
    parser.add_argument("--masking_ratio", type=float, default=0.95)
    parser.add_argument("--mae_batch_size", type=int, default=256)
    parser.add_argument("--train_mae_every", type=int, default=1)

    # PPO 强化学习配置
    parser.add_argument("--rollout_length", type=int, default=32768)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--lr_ppo", type=float, default=1e-4)
    parser.add_argument("--vision_only_control", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=256)

    # PPO + MAE 联合优化设置
    parser.add_argument("--separate_optimizer", type=str2bool, default=False)

    # 自动读取命令行参数，打包为对象传递给 config
    config = parser.parse_args()

    # 设置随机种子
    set_random_seed(config.seed)

    # ============ 【环境构建】 ============
    # 根据状态类型配置触觉传感器数量
    num_tactiles = 0
    if config.state_type == "vision_and_touch" or config.state_type == "touch":
        num_tactiles = 2
        if config.env in [
            "HandManipulateBlockRotateZFixed-v1",
            "HandManipulateEggRotateFixed-v1",
            "HandManipulatePenRotateFixed-v1",
        ]:
            num_tactiles = 1

    # 环境通用配置
    env_config = {
        "use_latch": config.use_latch,
    }

    # 插孔对象和 holder 多样性（为插孔任务预设形状组合）
    objects = [
        "square", "triangle", "horizontal", "vertical", "trapezoidal", "rhombus",
    ]
    holders = ["holder1", "holder2", "holder3"]

    # 创建多个并行环境（用于向量化训练）
    # 每个环境互不通信，也不会交换参数，都是在自己的 episode 里采样
    env_list = [
        envs.make_env(
            config.env,
            i,
            config.seed,
            config.state_type,
            objects=objects,
            holders=holders,
            camera_idx=config.camera_idx,
            frame_stack=config.frame_stack,
            no_rotation=config.no_rotation,
            **env_config,
        )
        for i in range(config.n_envs)
    ]

    # 使用 Subproc 或 Dummy 启动向量化环境
    # SubprocVecEnv: 多个进程 同时运行 多个独立的环境; DummyVecEnv: 多个同步环境 在主线程中串行运行（仅用于调试或小规模）
    if config.n_envs < 100:
        env = SubprocVecEnv(env_list)
    else:
        env = DummyVecEnv(env_list)

    # 启用 reward 归一化
    env = VecNormalize(env, norm_obs=False, norm_reward=config.norm_reward)

    # ============ 【MAE 网络构建】 ============
    # VTT 是视觉 + 触觉 transformer 编码器
    v = VTT(
        image_size=(64, 64),
        tactile_size=(32, 32),
        image_patch_size=8,
        tactile_patch_size=4,
        dim=config.dim_embedding,
        depth=4,
        heads=4,
        mlp_dim=config.dim_embedding * 2,
        num_tactiles=num_tactiles,
        image_channels=3 * config.frame_stack,
        tactile_channels=3 * config.frame_stack,
        frame_stack=config.frame_stack,
    )

    # VTMAE 是自监督重建模块
    mae = VTMAE(
        encoder=v,
        masking_ratio=config.masking_ratio,
        decoder_dim=config.dim_embedding,
        decoder_depth=3,
        decoder_heads=4,
        num_tactiles=num_tactiles,
        early_conv_masking=config.early_conv_masking,
        use_sincosmod_encodings=config.use_sincosmod_encodings,
        frame_stack=config.frame_stack,
    )
    if torch.cuda.is_available():
        mae.cuda()
    mae.eval()  # 初始为 eval 模式

    # ============ 【联合训练 PPO + MAE】 ============
    if config.representation:
        # 初始化 MAE 训练器（用于自监督学习）
        mae.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})

        # 使用 MAE 特征提取器构建策略网络
        policy = MAEPolicy
        policy_kwargs = {
            "mae_model": mae,
            "dim_embeddings": config.dim_embedding,
            "vision_only_control": config.vision_only_control,
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "frame_stack": config.frame_stack,
        }

        # 初始化 PPO + MAE 模型
        # model对象 是PPO_MAE类（继承自stable_baseline3） 有 .learn 的训练循环方法
        model = PPO_MAE(
            policy,
            env,
            verbose=1,
            learning_rate=config.lr_ppo,
            tensorboard_log=config.wandb_dir + "tensorboard/",
            batch_size=config.batch_size,
            n_steps=config.rollout_length // config.n_envs,
            n_epochs=config.ppo_epochs,
            mae_batch_size=config.mae_batch_size,
            separate_optimizer=config.separate_optimizer,
            policy_kwargs=policy_kwargs,
            mae=mae,
        )

        # 构建 callback 用于记录和评估
        callbacks = create_callbacks(config, model, num_tactiles, objects, holders)

        # 开始训练
        model.learn(
            total_timesteps=config.total_timesteps, callback=CallbackList(callbacks)
        )

    else:
        # 不使用表征时，退化为常规 PPO
        model = PPO(
            MAEPolicy,
            env,
            verbose=1,
            learning_rate=config.lr_ppo,
            tensorboard_log=config.wandb_dir + "ppo_privileged_tensorboard/",
            batch_size=config.batch_size,
            n_steps=config.rollout_length // config.n_envs,
            n_epochs=config.ppo_epochs,
            policy_kwargs={
                "mae_model": mae,
                "dim_embeddings": config.dim_embedding,
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
                "frame_stack": config.frame_stack,
            },
        )
        callbacks = create_callbacks(config, model, num_tactiles, objects, holders)
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks)

# 运行入口
if __name__ == "__main__":
    main()
