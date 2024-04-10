import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ppo_altAct.core as core
from spinup.utils.logx import EpochLogger
import shutil
import os


grad_norm_clip = 0.5


class WorldModel:  # :)
    def __init__(self, env):
        self.env = env

    def step(self, a):
        qpos0, qvel0 = (
            self.env.sim.data.qpos.flat.copy(),
            self.env.sim.data.qvel.flat.copy(),
        )
        s, r, d, _ = self.env.step(a)
        self.env.set_state(qpos0, qvel0)
        return s, r, d, _

class PPOBuffer:
    def __init__(self, size, gamma=0.99, lam=0.97):

        self.gamma, self.lam = gamma, lam
        self.max_size = size

        self.reset()

        self.gamma, self.lam = gamma, lam
        self.max_size = size

    @property
    def rollout_size(self):
        return len(self.obs_buf)

    @property
    def total_size(self):
        return len(self.obs_out)

    def store(
        self, obs, act, rew, val, logp, alt_obss, alt_acts, alt_rews, alt_vals
    ):

        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.logp_buf.append(logp)

        self.obs_buf_alt.append(alt_obss)
        self.act_buf_alt.append(alt_acts)
        self.rew_buf_alt.append(alt_rews)
        self.val_buf_alt.append(alt_vals)

    def finish_path(self, last_val=0):

        assert (
            len(self.act_buf)
            == len(self.obs_buf)
            == len(self.rew_buf)
            == len(self.val_buf)
            == len(self.logp_buf)
        )

        ### For actions taken:
        self.obs_out.extend(self.obs_buf)
        self.act_out.extend(self.act_buf)
        self.logp_out.extend(self.logp_buf)

        self.rew_buf.append(last_val)
        self.val_buf.append(last_val)
        rews = np.array(self.rew_buf)
        vals = np.array(self.val_buf)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_out.extend(core.discount_cumsum(deltas, self.gamma * self.lam))
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_out.extend(core.discount_cumsum(rews, self.gamma)[:-1])

        ### For imagined actions:
        assert (
            len(self.act_buf_alt)
            == len(self.obs_buf_alt)
            == len(self.rew_buf_alt)
            == len(self.val_buf_alt)
        )
        M = len(self.act_buf_alt)
        if M == 0:
            print("Warning: no alternative actions")
            
        for i in range(M):
            mean_r_alt, std_r_alt = np.mean(self.rew_buf_alt[i]), np.std(
                self.rew_buf_alt[i]
            )
            # print(len(self.obs_buf_alt[i]), len(self.act_buf_alt[i]), len(self.rew_buf_alt[i]), len(self.val_buf_alt[i]))

            assert (
                len(self.act_buf_alt[i])
                == len(self.obs_buf_alt[i])
                == len(self.rew_buf_alt[i])
                == len(self.val_buf_alt[i])
            )
            for obs_alt, act_alt, rew_alt in zip(
                self.obs_buf_alt[i],
                self.act_buf_alt[i],
                self.rew_buf_alt[i],
            ):
                self.obs_out_alt.append(obs_alt)
                self.act_out_alt.append(act_alt)
                self.adv_out_alt.append((rew_alt - mean_r_alt) / std_r_alt)

        self.reset_rollout()

    def get(self):

        if (
            not self.total_size == self.max_size
        ):  # buffer has to be full before you can get
            print(
                f"Warning: buffer has to be full before you can get, current size: {self.total_size}"
            )
            print("get: obs_buf: ", len(self.obs_buf))

        # Actions taken
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_out), np.std(self.adv_out)
        adv_out = (self.adv_out - adv_mean) / adv_std

        assert (
            len(self.obs_out)
            == len(self.act_out)
            == len(self.ret_out)
            == len(self.logp_out)
            == len(self.adv_out)
        )
        obs_out = np.stack(self.obs_out)
        obs_out = torch.tensor(obs_out, dtype=torch.float32)
        act_out = np.stack(self.act_out)  # (M, act_dim)
        act_out = torch.tensor(act_out, dtype=torch.float32)

        # Imagined actions
        assert len(self.obs_out_alt) == len(self.act_out_alt) == len(self.adv_out_alt)

        self.logp_out = np.array(self.logp_out)
        data = dict(
            obs=obs_out,
            act=act_out,
            ret=torch.tensor(self.ret_out, dtype=torch.float32),
            adv=torch.tensor(adv_out, dtype=torch.float32),
            logp=torch.tensor(self.logp_out, dtype=torch.float32),
            obs_alt=torch.tensor(self.obs_out_alt, dtype=torch.float32),
            act_alt=torch.tensor(self.act_out_alt, dtype=torch.float32),
            adv_alt=torch.tensor(self.adv_out_alt, dtype=torch.float32),
        )
        self.reset()
        return data

    def reset_rollout(self):
        # Taken actions
        self.obs_buf, self.act_buf = [], []
        self.rew_buf, self.val_buf, self.logp_buf = [], [], []
        # Imagined actions
        self.obs_buf_alt, self.act_buf_alt = [], []
        self.rew_buf_alt, self.val_buf_alt = [], []

    def reset(self):
        self.reset_rollout()
        # Actions taken
        self.obs_out, self.act_out = [], []
        self.ret_out, self.logp_out, self.adv_out = [], [], []
        # Imagined actions
        self.obs_out_alt, self.act_out_alt = [], []
        self.adv_out_alt = []

def ppo_altAct(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
    logger_kwargs=dict(),
    save_freq=10,
    n_alternative_actions = 3,
    train_alt_pi_iters = 5
):
    print(
        f"env_fn: {env_fn}\n",
        f"actor_critic: {actor_critic}\n",
        f"ac_kwargs: {ac_kwargs}\n",
        f"seed: {seed}\n",
        f"steps_per_epoch: {steps_per_epoch}\n",
        f"epochs: {epochs}\n",
        f"gamma: {gamma}\n",
        f"clip_ratio: {clip_ratio}\n",
        f"pi_lr: {pi_lr}\n",
        f"vf_lr: {vf_lr}\n",
        f"train_pi_iters: {train_pi_iters}\n",
        f"train_v_iters: {train_v_iters}\n",
        f"lam: {lam}\n",
        f"max_ep_len: {max_ep_len}\n",
        f"target_kl: {target_kl}\n",
        f"logger_kwargs: {logger_kwargs}\n",
        f"save_freq: {save_freq}\n",
        sep="",
    )
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()

    # World Model
    world_model = WorldModel(env)

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / 1)
    buf = PPOBuffer(size=local_steps_per_epoch)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):

        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        pi, logp = ac.pi(obs, act)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()
        buf.reset()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info["kl"]
            if kl > 1.5 * target_kl:
                logger.log("Early stopping at step %d due to reaching max kl." % i)
                break
            loss_pi.backward()
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Alternative actions stuff
        obs_alt, act_alt, adv_alt = data["obs_alt"], data["act_alt"], data["adv_alt"]
        if len(obs_alt)>0:
            for i in range(train_alt_pi_iters):
                pi_optimizer.zero_grad()
                pi, logp = ac.pi(obs_alt, act_alt)
                loss = (-(logp * adv_alt) / 10).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.pi.parameters(), grad_norm_clip / 10.0)
                pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV=(loss_v.item() - v_l_old),
        )

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            # a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            # alternative actions stuff
            alt_obss = []
            alt_acts = []
            alt_rews = []
            alt_vals = []
            for _ in range(n_alternative_actions):
                a_alt, v_alt, logp_alt = ac.step(
                    torch.as_tensor(o, dtype=torch.float32)
                )
                alt_obss.append(o)
                alt_acts.append(a_alt)
                alt_vals.append(v_alt)
                s_alt, r_alt, d_alt, _ = world_model.step(a_alt)
                alt_rews.append(r_alt)

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(
                obs=o,
                act=a,
                rew=r,
                val=v,
                logp=logp,
                alt_obss=alt_obss,
                alt_acts=alt_acts,
                alt_rews=alt_rews,
                alt_vals=alt_vals,
            )
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print(
                        "Warning: trajectory cut off by epoch at %d steps." % ep_len,
                        flush=True,
                    )
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    v = ac.v(torch.as_tensor(o, dtype=torch.float32)).detach().numpy()
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("ClipFrac", average_only=True)
        logger.log_tabular("StopIter", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Walker2d-v3")
    parser.add_argument("--hid", type=str, default="[64,32]")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=750)
    parser.add_argument("--exp_name", type=str, default="ppo")
    parser.add_argument("--n_alternative_actions", type=int, default=3)
    args = parser.parse_args()

    hidden_dims = eval(args.hid)

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    os.makedirs(logger_kwargs["output_dir"], exist_ok=True)
    shutil.copyfile(__file__, os.path.join(logger_kwargs["output_dir"], "ppo_altAct.py"))
    shutil.copyfile(
        os.path.dirname(__file__) + "/core.py",
        os.path.join(logger_kwargs["output_dir"], "core.py"),
    )
    # save the command line arguments
    with open(os.path.join(logger_kwargs["output_dir"], "args.txt"), "w") as f:
        f.write(str(args))

    ppo_altAct(
        lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=hidden_dims),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        n_alternative_actions=args.n_alternative_actions,
        logger_kwargs=logger_kwargs,
    )