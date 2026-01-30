import logging
import os
import time
import threading

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    print("-------------------------------")
    print("Cfgs:",OmegaConf.to_yaml(cfg, resolve=True))
    print("-------------------------------")

    run = init_wandb(cfg)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv
    print("Available environments:", IsaacEnv.REGISTRY.keys())
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    algo_name = cfg.algo.name.lower()
    if algo_name in {"mappo_graph", "mappo_graph_attention"}:
        if cfg.task.get("ravel_obs", False) or cfg.task.get("ravel_obs_central", False):
            print("[INFO] Disabling observation raveling for graph-attention policy.")
        cfg.task.ravel_obs = False
        cfg.task.ravel_obs_central = False
    base_env = env_class(cfg, headless=cfg.headless)
    # propagate env.num_envs into algo.num_envs so algorithms know how many envs to expect
    cfg.algo.num_envs = getattr(base_env, "num_envs", 1)
    cfg.algo.mpc_dt = base_env.dt

    transforms = [InitTracker()]

    # a CompositeSpec is by default processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
        checkpoint_path = cfg.get("checkpoint_path", None)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=base_env.device)
            policy.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully.")
        else:
            print("No checkpoint found, starting training from scratch.")
    except Exception as e:
        raise NotImplementedError(f"Algorithm: {cfg.algo.name} has error ({e})")

    # run.watch(policy, log="all",log_freq=128)

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        exploration_type: ExplorationType=ExplorationType.MODE
    ):

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        # log video
        # Calculate fps with safety check to avoid None or zero values
        calculated_fps = 0.5 / (cfg.sim.dt * cfg.sim.substeps) if (cfg.sim.dt and cfg.sim.substeps) else 30.0
        # Ensure fps is a valid positive integer
        video_fps = int(max(1.0, float(calculated_fps)) if calculated_fps else 30.0)
        
        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"),
            fps=video_fps,
            format="mp4"
        )

        # log distributions
        # df = pd.DataFrame(traj_stats)
        # table = wandb.Table(dataframe=df)
        # info["eval/return"] = wandb.plot.histogram(table, "return")
        # info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")

        return info

    pbar = tqdm(collector, total=total_frames//frames_per_batch)
    # env.train()
    env.eval()
    
    # Track curriculum stage for curriculum learning
    previous_curriculum_stage = base_env.get_curriculum_stage()
    
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
        
        # Update curriculum stage in environment based on total frames
        if hasattr(base_env, 'curriculum_enable') and base_env.curriculum_enable:
            stage_changed = base_env.update_curriculum_stage(collector._frames, total_frames)
            current_stage = base_env.get_curriculum_stage()
            
            # Update policy's curriculum stage
            if hasattr(policy, 'update_curriculum_stage'):
                policy_stage_changed = policy.update_curriculum_stage(current_stage)
                
                if policy_stage_changed:
                    print(f"[Training] Curriculum stage transition at {collector._frames} frames")
                    print(f"[Training] New stage: {current_stage}")
                    
                    # ===== 重要：阶段切换时重置环境 =====
                    print("[Training] Resetting all environments for new stage...")
                    # 重置所有环境
                    all_env_ids = torch.arange(base_env.num_envs, device=base_env.device)
                    base_env._reset_idx(all_env_ids)
                    
                    # 清空数据收集器的缓存（如果有的话）
                    # 注意：SyncDataCollector 使用 return_same_td=True，所以不需要特殊清理
                    # 但我们需要确保下一次收集的数据是新阶段的数据
                    
                    print("[Training] Environment reset complete")
                    
                    # Log stage change to wandb
                    info["curriculum/stage"] = current_stage
                    info["curriculum/stage_change"] = 1.0
            
            # Log current stage periodically
            info["curriculum/current_stage"] = current_stage
        
        env.train()
        info.update(policy.train_op(data.to_tensordict()))
        env.eval()

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())
            # env.train()
            # base_env.train()

        if save_interval > 0 and i % save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            logging.info(f"Step {i}: Saved checkpoint to {str(ckpt_path)}")

        run.log(info)
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break

    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate())
    run.log(info)

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            f"{cfg.task.name}-{cfg.algo.name.lower()}",
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        wandb.save(ckpt_path)
        run.log_artifact(model_artifact)

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
