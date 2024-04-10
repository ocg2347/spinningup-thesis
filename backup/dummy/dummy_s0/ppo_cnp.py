import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ppo_cnp_altAct_pwm.core as core
from spinup.utils.logx import EpochLogger
import shutil
import os

context_window_len = 10
n_context_max = 5
use_time_input = True
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
    def __init__(self, cont_dim, size, gamma=0.99, lam=0.95):

        self.n_context_max = max(1, n_context_max)
        self.cont_dim = cont_dim
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
        self,
        conts,
        obs,
        act,
        rew,
        val,
        logp,
        alt_conts,
        alt_obss,
        alt_acts,
        alt_rews,
    ):

        self.cont_buf.append(conts)
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.logp_buf.append(logp)

        self.cont_buf_alt.append(alt_conts)
        self.obs_buf_alt.append(alt_obss)
        self.act_buf_alt.append(alt_acts)
        self.rew_buf_alt.append(alt_rews)

    def finish_path(self, last_val=0):

        assert (
            len(self.act_buf)
            == len(self.obs_buf)
            == len(self.rew_buf)
            == len(self.val_buf)
            == len(self.logp_buf)
            == len(self.cont_buf)
        )

        ### For actions taken:
        self.obs_out.extend(self.obs_buf)
        self.act_out.extend(self.act_buf)
        self.cont_out.extend(self.cont_buf)
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
            == len(self.cont_buf_alt)
        )
        M = len(self.act_buf_alt)
        for i in range(M):
            mean_r_alt, std_r_alt = np.mean(self.rew_buf_alt[i]), np.std(
                self.rew_buf_alt[i]
            )
            assert (
                len(self.act_buf_alt[i])
                == len(self.obs_buf_alt[i])
                == len(self.rew_buf_alt[i])
                == len(self.cont_buf_alt[i])
            )
            for cont_alt, obs_alt, act_alt, rew_alt in zip(
                self.cont_buf_alt[i],
                self.obs_buf_alt[i],
                self.act_buf_alt[i],
                self.rew_buf_alt[i],
            ):
                self.obs_out_alt.append(obs_alt)
                self.act_out_alt.append(act_alt)
                self.cont_out_alt.append(cont_alt)
                self.adv_out_alt.append((rew_alt - mean_r_alt) / (std_r_alt + 1e-8))

        self.reset_rollout()

    def prepare_context_and_mask_batch(self, cont_list):

        M = len(cont_list)
        context_batch = torch.zeros(
            M, self.n_context_max, self.cont_dim, dtype=torch.float32
        )
        cont_mask_batch = torch.zeros(M, self.n_context_max, dtype=torch.float32)
        for i in range(M):
            n_context_i = cont_list[i].shape[1]
            context_batch[i, :n_context_i, :] = torch.tensor(
                cont_list[i], dtype=torch.float32
            )
            cont_mask_batch[i, :n_context_i] = torch.ones(
                n_context_i, dtype=torch.float32
            )
        return context_batch, cont_mask_batch

    def get(self):

        if (
            not self.total_size == self.max_size
        ):  # buffer has to be full before you can get
            raise ValueError("Buffer is not full yet!")

        # Actions taken
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_out), np.std(self.adv_out)
        adv_out = (self.adv_out - adv_mean) / adv_std

        # context_batch: M x n_context_max x (obs_dim + act_dim)
        # cont_mask_batch: M x n_context_max
        assert (
            len(self.cont_out)
            == len(self.obs_out)
            == len(self.act_out)
            == len(self.ret_out)
            == len(self.logp_out)
            == len(self.adv_out)
        )
        context_batch_out, cont_mask_batch_out = self.prepare_context_and_mask_batch(
            self.cont_out
        )
        assert context_batch_out.shape == (
            len(self.cont_out),
            self.n_context_max,
            self.cont_dim,
        )
        assert cont_mask_batch_out.shape == (len(self.cont_out), self.n_context_max)
        obs_out = np.stack(self.obs_out)
        obs_out = torch.tensor(obs_out, dtype=torch.float32)
        obs_out = obs_out.unsqueeze(1)  # (M, obs_dim)->(M, 1, obs_dim)
        act_out = np.stack(self.act_out)  # (M, act_dim)
        act_out = torch.tensor(act_out, dtype=torch.float32)
        act_out = act_out.unsqueeze(1)  # (M, act_dim)->(M, 1, act_dim)

        # Imagined actions
        assert (
            len(self.cont_out_alt)
            == len(self.obs_out_alt)
            == len(self.act_out_alt)
            == len(self.adv_out_alt)
        )
        print("cont_out_alt: ", len(self.cont_out_alt))
        context_batch_out_alt, cont_mask_batch_out_alt = (
            self.prepare_context_and_mask_batch(self.cont_out_alt)
        )
        obs_out_alt = np.stack(self.obs_out_alt)
        obs_out_alt = torch.tensor(obs_out_alt, dtype=torch.float32)
        obs_out_alt = obs_out_alt.unsqueeze(1)  # (M, obs_dim)->(M, 1, obs_dim)
        act_out_alt = np.stack(self.act_out_alt)  # (M, act_dim)
        act_out_alt = torch.tensor(act_out_alt, dtype=torch.float32)
        act_out_alt = act_out_alt.unsqueeze(1)  # (M, act_dim)->(M, 1, act_dim)

        data = dict(
            cont=context_batch_out,
            cont_mask=cont_mask_batch_out,
            obs=obs_out,
            act=act_out,
            ret=torch.tensor(self.ret_out, dtype=torch.float32).reshape(-1, 1, 1),
            adv=torch.tensor(adv_out, dtype=torch.float32).reshape(-1, 1, 1),
            logp=torch.tensor(self.logp_out, dtype=torch.float32).reshape(-1, 1, 1),
            cont_alt=context_batch_out_alt,
            cont_mask_alt=cont_mask_batch_out_alt,
            obs_alt=obs_out_alt,
            act_alt=act_out_alt,
            adv_alt=torch.tensor(self.adv_out_alt, dtype=torch.float32).reshape(
                -1, 1, 1
            ),
        )
        self.reset()
        return data

    def reset_rollout(self):
        # Taken actions
        self.cont_buf, self.obs_buf, self.act_buf = [], [], []
        self.rew_buf, self.val_buf, self.logp_buf = [], [], []
        # Imagined actions
        self.cont_buf_alt, self.obs_buf_alt, self.act_buf_alt = [], [], []
        self.rew_buf_alt = []

    def reset(self):
        self.reset_rollout()
        # Actions taken
        self.cont_out, self.obs_out, self.act_out = [], [], []
        self.ret_out, self.logp_out, self.adv_out = [], [], []
        # Imagined actions
        self.cont_out_alt, self.obs_out_alt, self.act_out_alt = [], [], []
        self.adv_out_alt = []


def ppo_cnp(
    env_id,
    alt_act_adv,
    actor_critic=core.CNPActorMLPCritic,
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
    train_alt_pi_iters = 5,
):
    assert alt_act_adv in ['rew', 'val']

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = gym.make(env_id)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    obs_dim = obs_dim + 1 if use_time_input else obs_dim
    context_dim = obs_dim + act_dim

    # World Model
    world_model = WorldModel(env)

    # Create actor-critic module
    ac = actor_critic(
        env.observation_space,
        env.action_space,
        use_time_input=use_time_input,
        encoder_output_size=8,
        **ac_kwargs,
    )

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / 1)
    buf = PPOBuffer(
        cont_dim=context_dim,
        size=local_steps_per_epoch,
        gamma=gamma,
        lam=lam,
    )

    # sample action using the last s,a as context and current s as target
    @torch.no_grad()
    def sample_action(o, deterministic=False):
        if buf.rollout_size == 0 or context_window_len == 0 or n_context_max == 0:
            context_points = torch.zeros(1, 1, context_dim, dtype=torch.float32)
        else:
            context_window_idxes = np.arange(
                2, min(context_window_len, buf.rollout_size) + 1
            )  # exclude last step because we will include it anyways
            n_context = np.random.randint(0, n_context_max)
            n_context = min(context_window_idxes.size, n_context)
            context_idxes = np.random.choice(
                context_window_idxes, n_context, replace=False
            )
            context_idxes = np.append(context_idxes, 1)

            context_points = []
            for i in context_idxes:
                context = np.concatenate([buf.obs_buf[-i], buf.act_buf[-i]])
                if use_time_input:
                    t_wrt_st = (
                        -i / context_window_len
                    )  # time with respect to current o, (s_t)
                    context[0] = t_wrt_st  # set time to relative time wrt s_t!!!!!!!
                context_points.append(context)
            context_points = np.array(context_points)  # (n_context, dim_context)
            context_points = torch.tensor(
                context_points, dtype=torch.float32
            ).unsqueeze(
                0
            )  # (1 x n_context, dim_context)
        a_mean, a_std = ac.pi.cnp(
            observation=context_points,
            target=torch.as_tensor(o, dtype=torch.float32).reshape(1, 1, obs_dim),
        )
        a_mean = a_mean[0, 0, :]
        if deterministic:
            a = a_mean
        else:
            a = torch.normal(a_mean, a_std)
        # compute log prob of the action using a_man, a_std
        a_dist = torch.distributions.Normal(a_mean, a_std)
        log_p = a_dist.log_prob(a).sum()  # (1, 1, act_dim) -> (1, 1)

        return (
            a.numpy(),
            log_p.item(),
            context_points.numpy(),
        )

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):

        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]
        cont, cont_mask = data["cont"], data["cont_mask"]
        assert cont.shape[0] == obs.shape[0] == act.shape[0] == adv.shape[0] == logp_old.shape[0] == cont_mask.shape[0]

        pi, logp = ac.pi(
            observation=cont, target=obs, target_truth=act, observation_mask=cont_mask
        )  # (M, 1, 1)

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

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        if use_time_input:
            obs = obs[:, 0, 1:]
        ret = ret.reshape(-1, 1)
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
        cont_alt, cont_mask = data["cont_alt"], data["cont_mask_alt"]
        obs_alt, act_alt, adv_alt = data["obs_alt"], data["act_alt"], data["adv_alt"]
        
        if cont_alt.shape[0] > 0:
            print("cont_alt: ", cont_alt.shape)
            for i in range(train_alt_pi_iters):
                print("train_alt_pi_iters: ", i)
                pi_optimizer.zero_grad()
                _, logp = ac.pi(
                    observation=cont_alt,
                    target=obs_alt,
                    target_truth=act_alt,
                    observation_mask=cont_mask,
                )  # (M, 1, 1)
                loss = (-(logp * adv_alt) / 10).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.pi.parameters(), grad_norm_clip)
                pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        # ac.pi.cnp.log_std.exp() to list
        log_std_policy = ac.pi.cnp.log_std.exp()
        log_std_policy = log_std_policy.tolist()
        log_std_policy = [round(x, 5) for x in log_std_policy]
        logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV=(loss_v.item() - v_l_old),
            cnp_std = log_std_policy
        )

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            # a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            v = ac.v(torch.as_tensor(o, dtype=torch.float32)).detach().numpy()
            if use_time_input:
                o = np.insert(o, 0, 0.0)  # insert 0 in the beginning!!!
            a, logp, context_points = sample_action(o)

            # alternative actions stuff
            alt_conts = []
            alt_obss = []
            alt_acts = []
            alt_rews = []
            # alt_vals = []
            for _ in range(n_alternative_actions):
                a_alt, logp_alt, context_alt = sample_action(o)
                alt_conts.append(context_alt)
                alt_obss.append(o)
                alt_acts.append(a_alt)
                s_alt, r_alt, d_alt, _ = world_model.step(a_alt)
                if alt_act_adv == 'rew':
                    alt_rews.append(r_alt)
                elif alt_act_adv == 'val':
                    if d_alt:
                        alt_rews.append(r_alt)
                    else:
                        v_alt_next = ac.v(torch.as_tensor(s_alt, dtype=torch.float32)).detach().numpy()
                        alt_rews.append(r_alt + gamma * v_alt_next)


            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(
                conts=context_points,
                obs=o,
                act=a,
                rew=r,
                val=v,
                logp=logp,
                alt_conts=alt_conts,
                alt_obss=alt_obss,
                alt_acts=alt_acts,
                alt_rews=alt_rews,
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
    parser.add_argument("--alt_act_adv", type=str)
    args = parser.parse_args()

    hidden_dims = eval(args.hid)

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    os.makedirs(logger_kwargs["output_dir"], exist_ok=True)
    shutil.copyfile(__file__, os.path.join(logger_kwargs["output_dir"], "ppo_cnp.py"))
    shutil.copyfile(
        os.path.dirname(__file__) + "/core.py",
        os.path.join(logger_kwargs["output_dir"], "core.py"),
    )
    shutil.copyfile(
        os.path.dirname(__file__) + "/models.py",
        os.path.join(logger_kwargs["output_dir"], "models.py"),
    )

    ppo_cnp(
        env_id=args.env,
        alt_act_adv=args.alt_act_adv,
        actor_critic=core.CNPActorMLPCritic,
        ac_kwargs=dict(hidden_sizes=hidden_dims),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
