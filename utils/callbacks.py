import wandb
import numpy as np

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.vec_env import DummyVecEnv

import envs
from utils.wandb_logger import WandbLogger
from utils.pretrain_utils import log_videos


class TensorboardCallback(BaseCallback):
    """
    自定义 Tensorboard 回调，用于记录额外的统计量（如平均成功率）。
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 每个 step 都记录平均成功率到 TensorBoard
        self.logger.record("rollout/avg_success", np.mean(self.model.ep_success_buffer))
        return True


class EvalCallback(BaseCallback):
    """
    自定义评估回调，每隔一定间隔进行一次评估 roll-out。
    并记录评估 return、采集图像和视频（用于可视化）。
    """

    def __init__(
        self,
        env,
        state_type,
        no_tactile=False,
        representation=True,
        eval_every=1,
        verbose=0,
        config=None,
        objects=["square"],
        holders=["holder2"],
        camera_idx=0,
        frame_stack=1,
    ):
        super(EvalCallback, self).__init__(verbose)
        self.n_samples = 4
        self.eval_seed = 100
        self.no_tactile = no_tactile
        self.representation = representation

        # 构造评估环境（只构建一个 env）
        env_config = {"use_latch": config.use_latch}

        self.test_env = DummyVecEnv(
            [
                envs.make_env(
                    env,
                    0,
                    self.eval_seed,
                    state_type=state_type,
                    objects=objects,
                    holders=holders,
                    camera_idx=camera_idx,
                    frame_stack=frame_stack,
                    no_rotation=config.no_rotation,
                    **env_config
                )
            ]
        )
        self.count = 0  # 当前 rollout 计数
        self.eval_every = eval_every  # 评估间隔（以 rollout 次数为单位）

    def _on_step(self) -> bool:
        # 每个训练 step 会调用一次（此处不做操作）
        return True

    def _on_rollout_start(self) -> None:
        """
        每次 rollout 开始前触发。
        到达评估间隔时执行评估过程。
        """
        self.count += 1
        if self.count >= self.eval_every:
            # 执行评估并获取数据
            ret, obses, rewards_per_step = self.eval_model()
            frame_stack = obses[0]["image"].shape[1]

            # 记录评估 reward（return）
            self.logger.record("eval/return", ret)

            # 将视频上传给 wandb 记录
            log_videos(
                obses,
                rewards_per_step,
                self.logger,
                self.model.num_timesteps,
                frame_stack=frame_stack,
            )
            self.count = 0  # 重置计数器

    def eval_model(self):
        """
        执行一次完整的评估 rollout，返回总回报 + 每步数据。
        """
        print("Collect eval rollout")
        obs = self.test_env.reset()
        dones = [False]
        reward = 0
        obses = []
        rewards_per_step = []

        while not dones[0]:
            action, _ = self.model.predict(obs, deterministic=False)
            obs, rewards, dones, info = self.test_env.step(action)
            reward += rewards[0]
            rewards_per_step.append(rewards[0])
            obses.append(obs)

        return reward, obses, rewards_per_step


def create_callbacks(config, model, num_tactiles, objects, holders):
    """
    构造 callback 列表，包含：
    - 评估回调（EvalCallback）
    - checkpoint 保存回调（CheckpointCallback）
    - TensorBoard 记录回调
    - 配置 WandB 日志记录器

    Args:
        config: 命令行参数解析结果
        model: 当前训练的 RL 模型
        num_tactiles: 使用的触觉通道数
        objects: 插孔对象形状集合
        holders: 插孔 holder 配置集合

    Returns:
        callbacks: 回调列表
    """
    no_tactile = num_tactiles == 0

    project_name = "MultimodalLearning"
    if config.env in ["Door"]:
        project_name += "_robosuite"

    callbacks = []

    # 添加评估回调
    eval_callback = EvalCallback(
        config.env,
        config.state_type,
        no_tactile=no_tactile,
        representation=config.representation,
        eval_every=config.eval_every // config.rollout_length,  # 以 rollout 数控制频率
        config=config,
        objects=objects,
        holders=holders,
        camera_idx=config.camera_idx,
        frame_stack=config.frame_stack,
    )
    callbacks.append(eval_callback)

    # 添加模型保存回调
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.save_freq // config.n_envs, 1),  # 按照环境数对齐
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,  # 保存环境归一化状态
    )
    callbacks.append(checkpoint_callback)

    # 添加自定义 TensorBoard 记录回调
    callbacks.append(TensorboardCallback())

    # ========== 配置 WandB 日志系统 ==========
    default_logger = configure_logger(
        verbose=1, tensorboard_log=model.tensorboard_log, tb_log_name="PPO"
    )

    wandb.init(
        project=project_name,
        config=config,
        save_code=True,
        name=default_logger.dir.split("/")[-1],
        dir=config.wandb_dir,
        id=config.wandb_id,
        entity=config.wandb_entity,
    )

    # 替换默认 logger 为 WandBLogger
    logger = WandbLogger(
        default_logger.dir, default_logger.output_formats, log_interval=1000
    )
    model.set_logger(logger)

    # 同步 checkpoint 保存路径到 WandB 目录
    checkpoint_callback.save_path = wandb.run.dir

    return callbacks
