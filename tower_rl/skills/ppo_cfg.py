"""rsl_rl runner config built from a skill YAML's ``ppo`` section (asymmetric actor-critic)."""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


def runner_cfg(skill: dict, experiment: str, max_iterations: int | None = None, seed: int = 42,
               device: str = "cuda:0") -> RslRlOnPolicyRunnerCfg:
    p = skill["ppo"]
    return RslRlOnPolicyRunnerCfg(
        seed=seed, device=device,
        num_steps_per_env=p["num_steps_per_env"],
        max_iterations=max_iterations or p["max_iterations"],
        save_interval=p["save_interval"],
        experiment_name=experiment,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=p["init_noise_std"], actor_obs_normalization=True, critic_obs_normalization=True,
            actor_hidden_dims=p["actor_hidden_dims"], critic_hidden_dims=p["critic_hidden_dims"], activation=p["activation"],
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=p["num_learning_epochs"], num_mini_batches=p["num_mini_batches"],
            learning_rate=p["learning_rate"], schedule=p["schedule"], gamma=p["gamma"], lam=p["lam"],
            entropy_coef=p["entropy_coef"], desired_kl=p["desired_kl"], max_grad_norm=p["max_grad_norm"],
            value_loss_coef=p["value_loss_coef"], use_clipped_value_loss=True, clip_param=p["clip_param"],
        ),
    )
