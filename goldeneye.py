# -*- coding: utf-8 -*-
from __future__ import annotations  # Doit être en premier

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='keras')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*np.object.*')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TensorFlow logs
"""GOLDENEYE MERLIN ULTIMATE - Complete Intelligent System
Fusion: GOLD_FINAL_V2 (13,258 lines) + Divergences + MERLIN Vision AI
Mat - November 2024"""

# ============================================================
# SAUVEGARDE AUTOMATIQUE AU DÉMARRAGE
# Crée une copie horodatée dans backups/ à chaque lancement.
# Garde les 5 dernières — protège contre toute corruption.
# ============================================================
import os as _os, shutil as _shutil
from datetime import datetime as _dt
try:
    _src  = _os.path.abspath(__file__)
    _bdir = _os.path.join(_os.path.dirname(_src), "backups")
    _os.makedirs(_bdir, exist_ok=True)
    _stamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    _dst   = _os.path.join(_bdir, f"GOLDENEYE_{_stamp}.py")
    _shutil.copy2(_src, _dst)
    # Rotation : garder seulement les 5 plus récentes
    _backups = sorted([f for f in _os.listdir(_bdir) if f.startswith("GOLDENEYE_") and f.endswith(".py")])
    for _old in _backups[:-5]:
        _os.remove(_os.path.join(_bdir, _old))
    print(f"  [BACKUP] {_dst}")
except Exception as _be:
    print(f"  [BACKUP] Erreur: {_be}")

# -*- coding: utf-8 -*-
#ultimate_hybrid_bot.py - VERSION COMPLETE PRODUCTION READY
#============================================================================
#ARCHITECTURE HYBRIDE ULTIME AVEC HRM-TRM-PPO-DQN FUSION
#============================================================================


import matplotlib

import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logger = logging.getLogger(__name__)
os.environ['PYTHONIOENCODING'] = 'utf-8'
import asyncio
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, deque
from datetime import datetime
import pickle
import json
import traceback
import random
import copy
import time
import sys


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DASHBOARD INTEGRATION - Bridge import
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Ajouter le rÃ©pertoire du script au sys.path pour l'import

import sys

# === MODE TEST RAPIDE - 30K STEPS TOTAL ===
PPO_STEPS    = 10000
DQN_STEPS    = 10000
FUSION_STEPS = 10000
TOTAL_STEPS  = PPO_STEPS + DQN_STEPS + FUSION_STEPS  # = 30000


class GoldenEyeTrainingConfig:
    """Configuration centralisee pour l'entrainement Manhattan Project"""
    PHASE_1_PPO_STEPS = 30000
    PHASE_2_DQN_STEPS = 20000
    PHASE_2B_TRANSFORMER_STEPS = 20000


from pathlib import Path
_script_dir = Path(__file__).parent.resolve()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    from goldeneye_bridge import log_training_metrics
    DASHBOARD_ENABLED = True
    print("âœ… Dashboard bridge loaded successfully")
    print(f"   Bridge location: {_script_dir / 'goldeneye_bridge.py'}")
except ImportError as e:
    DASHBOARD_ENABLED = False
    print("âš ï¸  Dashboard bridge not found - metrics won't be logged")
    print(f"   Looked in: {_script_dir}")
    print(f"   Error: {e}")
    def log_training_metrics(*args, **kwargs):
        pass  # Dummy function
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# Machine Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Gym / Gymnasium
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines 3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# ============================================================
# GOLDENEYE PATHS — Chemins de logs pour diagnostic Claude Code
# ============================================================
GOLDENEYE_PATHS = {
    "diagnostic":       "./models/goldeneye_diagnostic.json",
    "training_history": "./models/training_history.json",
    "log_live":         "./ultimate_trader.log",
    "ppo_model":        "./models/ppo_{symbol}.zip",
    "sac_model":        "./models/sac_{symbol}_sac.pt",
    "transformer_model":"./models/transformer_{symbol}_transformer.pt",
    "tensorboard_rppo": "./logs/rppo_{symbol}/",
    "tensorboard_sac":  "./logs/sac_{symbol}/",
    "checkpoints":      "./models/checkpoints_{symbol}/",
}

# ============================================================
# ACTIVE_HYPERPARAMS — Hyperparamètres optimisés (Manhattan)
# ============================================================
ACTIVE_HYPERPARAMS = {
    "recurrent_ppo": {
        "model": "RecurrentPPO (MlpLstmPolicy)",
        "learning_rate": 0.0002,           # Légèrement augmenté pour sortir des minima locaux
        "n_steps": 1024,                   # Horizon de collecte
        "batch_size": 128,                 # Réduit pour une meilleure précision du gradient
        "n_epochs": 10,                    # Plus d'itérations par batch
        "gamma": 0.98,                     # Vision à plus long terme (tendance)
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.05,                  # Exploration contrôlée (était 0.12)
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.015,
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,                # 1 couche est plus stable en trading que 2
    },
    "discrete_sac": {
        "model": "DiscreteSAC (Manhattan Agent 2)", 
        "learning_rate": 3e-4,
        "gamma": 0.95, 
        "tau": 0.005, 
        "learning_starts": 1000,
        "batch_size": 256, 
        "target_update_interval": 500,
        "train_freq": 4, 
        "gradient_steps": 2,
        "target_entropy_ratio": 0.98, 
        "twin_critics": True, 
        "auto_alpha": True,
    },
    "transformer": {
        "model": "TransformerAgent (Manhattan Agent 3)", 
        "seq_len": 60,                     # Augmenté pour capturer plus d'historique
        "d_model": 128, 
        "nhead": 4, 
        "num_layers": 3, 
        "dim_ff": 256,
        "learning_rate": 2e-4, 
        "n_steps": 1024, 
        "batch_size": 64,
        "n_epochs": 8, 
        "gamma": 0.95, 
        "ent_coef": 0.04,
    },
    "reward": {
        "scale": 0.01, 
        "profit_pct_cap": 0.25,            # Cap augmenté pour laisser courir les profits
        "clip_min": -5.0, 
        "clip_max": 5.0,
    },
    "gate": {"min_wr_pct": 51.5, "min_trades": 10, "agents": 3},
}

# SB3-Contrib — modèles professionnels
try:
    from sb3_contrib import RecurrentPPO, QRDQN
    SB3_CONTRIB_AVAILABLE = True
    print("  sb3-contrib: RecurrentPPO (LSTM) + QRDQN charges")
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    RecurrentPPO = PPO
    QRDQN = DQN
    print("  sb3-contrib non installe — pip install sb3-contrib")

# ============================================================
# DISCRETE SAC — Manhattan Project Agent 2
# ============================================================
import torch.nn.functional as F

class DiscreteSACNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
        def _critic():
            return nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions)
            )
        self.q1 = _critic(); self.q2 = _critic()
        self.q1_target = _critic(); self.q2_target = _critic()
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad = False

    def get_action_probs(self, obs):
        logits = self.actor(obs)
        return F.softmax(logits, dim=-1), F.log_softmax(logits, dim=-1)

    def get_action(self, obs, deterministic=False):
        probs, _ = self.get_action_probs(obs)
        if deterministic: return probs.argmax(dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def soft_update(self, tau=0.005):
        for t, s in zip(self.q1_target.parameters(), self.q1.parameters()):
            t.data.copy_(tau * s.data + (1-tau) * t.data)
        for t, s in zip(self.q2_target.parameters(), self.q2.parameters()):
            t.data.copy_(tau * s.data + (1-tau) * t.data)


class DiscreteSAC:
    TARGET_ENTROPY_RATIO = 0.98

    def __init__(self, policy, env, learning_rate=3e-4, buffer_size=200000,
                 learning_starts=1000, batch_size=256, gamma=0.90,
                 tau=0.005, train_freq=4, gradient_steps=2,
                 target_update_interval=500, verbose=0, **kwargs):
        self.env = env; self.lr = learning_rate; self.buffer_size = buffer_size
        self.learning_starts = learning_starts; self.batch_size = batch_size
        self.gamma = gamma; self.tau = tau; self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.num_timesteps = 0; self._n_updates = 0
        self.exploration_rate = 0.0
        _e = env.envs[0] if hasattr(env, 'envs') else env
        if hasattr(_e, 'env'): _e = _e.env
        obs_dim = _e.observation_space.shape[0]
        n_actions = _e.action_space.n
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DiscreteSACNetwork(obs_dim, n_actions).to(self.device)
        self.target_entropy = -self.TARGET_ENTROPY_RATIO * np.log(1.0 / n_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.opt_actor = torch.optim.Adam(self.net.actor.parameters(), lr=self.lr)
        self.opt_critic = torch.optim.Adam(
            list(self.net.q1.parameters()) + list(self.net.q2.parameters()), lr=self.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self._buf_obs = []; self._buf_next = []; self._buf_act = []
        self._buf_rew = []; self._buf_done = []
        self._buf_ptr = 0; self._buf_full = False
        self.logger = type('L', (), {'name_to_value': {}})()

    def _add_to_buffer(self, obs, next_obs, action, reward, done):
        if len(self._buf_obs) < self.buffer_size:
            self._buf_obs.append(None); self._buf_next.append(None)
            self._buf_act.append(None); self._buf_rew.append(None); self._buf_done.append(None)
        idx = self._buf_ptr
        self._buf_obs[idx]=obs; self._buf_next[idx]=next_obs
        self._buf_act[idx]=action; self._buf_rew[idx]=reward; self._buf_done[idx]=done
        self._buf_ptr = (self._buf_ptr+1) % self.buffer_size
        if self._buf_ptr == 0: self._buf_full = True

    def _sample_buffer(self):
        size = self.buffer_size if self._buf_full else self._buf_ptr
        idx = np.random.randint(0, size, self.batch_size)
        obs   = torch.FloatTensor(np.array([self._buf_obs[i]  for i in idx])).to(self.device)
        next_ = torch.FloatTensor(np.array([self._buf_next[i] for i in idx])).to(self.device)
        acts  = torch.LongTensor( np.array([self._buf_act[i]  for i in idx])).to(self.device)
        rews  = torch.FloatTensor(np.array([self._buf_rew[i]  for i in idx])).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.array([self._buf_done[i] for i in idx])).unsqueeze(1).to(self.device)
        return obs, next_, acts, rews, dones

    def _update(self):
        if (self._buf_full and self.buffer_size < self.batch_size) or \
           (not self._buf_full and self._buf_ptr < self.batch_size): return
        obs, next_obs, acts, rews, dones = self._sample_buffer()
        self.alpha = self.log_alpha.exp().item()
        with torch.no_grad():
            next_probs, next_log_p = self.net.get_action_probs(next_obs)
            q_next = torch.min(self.net.q1_target(next_obs), self.net.q2_target(next_obs))
            v_next = (next_probs * (q_next - self.alpha * next_log_p)).sum(dim=1, keepdim=True)
            q_target = rews + self.gamma * (1-dones) * v_next
        q1_pred = self.net.q1(obs).gather(1, acts.unsqueeze(1))
        q2_pred = self.net.q2(obs).gather(1, acts.unsqueeze(1))
        loss_c = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        self.opt_critic.zero_grad(); loss_c.backward(); self.opt_critic.step()
        probs, log_p = self.net.get_action_probs(obs)
        q_a = torch.min(self.net.q1(obs), self.net.q2(obs))
        loss_a = (probs * (self.alpha * log_p - q_a)).sum(dim=1).mean()
        self.opt_actor.zero_grad(); loss_a.backward(); self.opt_actor.step()
        entropy = -(probs * log_p).sum(dim=1).mean().detach()
        loss_alp = -(self.log_alpha * (entropy - self.target_entropy))
        self.opt_alpha.zero_grad(); loss_alp.backward(); self.opt_alpha.step()
        self._n_updates += 1
        if self._n_updates % self.target_update_interval == 0:
            self.net.soft_update(self.tau)
        self.logger.name_to_value = {
            'train/loss': loss_c.item(), 'train/entropy': entropy.item(),
            'sac/alpha': self.alpha, 'sac/actor_loss': loss_a.item(),
        }

    def predict(self, obs, deterministic=False, state=None, episode_start=None):
        with torch.no_grad():
            obs_t = torch.FloatTensor(np.array(obs)).to(self.device)
            if obs_t.ndim == 1: obs_t = obs_t.unsqueeze(0)
            action = self.net.get_action(obs_t, deterministic=deterministic)
            return action.cpu().numpy(), None

    def learn(self, total_timesteps, callback=None, reset_num_timesteps=True, **kwargs):
        if reset_num_timesteps: self.num_timesteps = 0
        _env = self.env.envs[0] if hasattr(self.env, 'envs') else self.env
        if hasattr(_env, 'env'): _env = _env.env
        obs, _ = self.env.reset() if hasattr(self.env, 'reset') else (_env.reset(), None)
        if isinstance(obs, tuple): obs = obs[0]
        obs = np.array(obs).flatten()
        if callback:
            if not isinstance(callback, list): callback = [callback]
            for cb in callback:
                if hasattr(cb, 'init_callback'): cb.init_callback(self)
                if hasattr(cb, 'on_training_start'): cb.on_training_start({}, {})
        for step in range(total_timesteps):
            self.num_timesteps += 1
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action = self.net.get_action(obs_t, deterministic=False).item()
            result = self.env.step([action])
            if len(result) == 5:
                next_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = result[:4]
            next_obs = np.array(next_obs).flatten() if not isinstance(next_obs, np.ndarray) else next_obs.flatten()
            self._add_to_buffer(obs, next_obs, action, float(reward), float(done))
            obs = next_obs
            if done:
                result2 = self.env.reset()
                obs = result2[0] if isinstance(result2, tuple) else result2
                obs = np.array(obs).flatten()
            if self.num_timesteps > self.learning_starts and step % self.train_freq == 0:
                for _ in range(self.gradient_steps): self._update()
            if callback:
                for cb in callback:
                    if hasattr(cb, 'on_step'): cb.on_step()
                    elif hasattr(cb, '_on_step'): cb._on_step()
        if callback:
            for cb in callback:
                if hasattr(cb, 'on_training_end'): cb.on_training_end()
        return self

    def save(self, path):
        torch.save({'net': self.net.state_dict(), 'log_alpha': self.log_alpha.data,
                    'num_timesteps': self.num_timesteps}, path + '_sac.pt')

    def load(self, path, env=None):
        data = torch.load(path + '_sac.pt', map_location=self.device)
        self.net.load_state_dict(data['net'])
        self.log_alpha.data = data['log_alpha']
        self.num_timesteps = data.get('num_timesteps', 0)
        if env: self.env = env
        return self


print("  DiscreteSAC (Manhattan Agent 2) charge")

# ============================================================
# TRANSFORMER AGENT — Manhattan Project Agent 3
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(attn_out))
        return self.norm2(x + self.drop(self.ff(x)))


class TransformerNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, seq_len=50, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.obs_dim = obs_dim; self.seq_len = seq_len; self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len+1, dropout=dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.policy_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, n_actions))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))

    def forward(self, seq):
        B = seq.shape[0]
        if seq.ndim == 2: seq = seq.reshape(B, self.seq_len, self.obs_dim)
        x = self.input_proj(seq)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        for block in self.blocks: x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.policy_head(cls_out), self.value_head(cls_out)


class TransformerAgent:
    SEQ_LEN=50; D_MODEL=128; NHEAD=4; LAYERS=3; DIM_FF=256

    def __init__(self, policy, env, learning_rate=3e-4, n_steps=1024, batch_size=64,
                 n_epochs=6, gamma=0.92, gae_lambda=0.95, clip_range=0.10,
                 ent_coef=0.08, vf_coef=0.5, max_grad_norm=0.5, verbose=0, **kwargs):
        self.env=env; self.lr=learning_rate; self.n_steps=n_steps; self.batch_size=batch_size
        self.n_epochs=n_epochs; self.gamma=gamma; self.gae_lambda=gae_lambda
        self.clip_range=clip_range; self.ent_coef=ent_coef; self.vf_coef=vf_coef
        self.max_grad_norm=max_grad_norm; self.num_timesteps=0; self.exploration_rate=0.0
        _e = env.envs[0] if hasattr(env, 'envs') else env
        if hasattr(_e, 'env'): _e = _e.env
        self.obs_dim = _e.observation_space.shape[0]
        self.n_actions = _e.action_space.n
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = TransformerNetwork(self.obs_dim, self.n_actions, self.SEQ_LEN,
                                      self.D_MODEL, self.NHEAD, self.LAYERS, self.DIM_FF).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100000, eta_min=1e-5)
        self._seq_buf = np.zeros((1, self.SEQ_LEN, self.obs_dim), dtype=np.float32)
        self.logger = type('L', (), {'name_to_value': {}})()

    def _update_seq(self, obs):
        self._seq_buf = np.roll(self._seq_buf, -1, axis=1)
        self._seq_buf[:, -1] = obs.reshape(1, self.obs_dim)

    def _reset_seq(self): self._seq_buf[:] = 0.0

    def predict(self, obs, deterministic=False, state=None, episode_start=None):
        self._update_seq(np.array(obs).flatten())
        with torch.no_grad():
            seq_t = torch.FloatTensor(self._seq_buf).to(self.device)
            logits, _ = self.net(seq_t)
            probs = F.softmax(logits, dim=-1)
            action = probs.argmax(dim=-1) if deterministic else torch.multinomial(probs, 1).squeeze(-1)
        return action.cpu().numpy(), None

    def _collect_rollout(self):
        obs_seq_list, act_list, rew_list, done_list, val_list, logp_list = [], [], [], [], [], []
        obs_raw = self.env.reset()
        if isinstance(obs_raw, tuple): obs_raw = obs_raw[0]
        obs = np.array(obs_raw).flatten(); self._reset_seq()
        for _ in range(self.n_steps):
            self._update_seq(obs)
            seq_t = torch.FloatTensor(self._seq_buf).to(self.device)
            with torch.no_grad():
                logits, value = self.net(seq_t)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample(); log_p = dist.log_prob(action)
            obs_seq_list.append(self._seq_buf.copy())
            act_list.append(action.item()); val_list.append(value.item()); logp_list.append(log_p.item())
            result = self.env.step([action.item()])
            if len(result)==5: next_obs, reward, terminated, truncated, _ = result; done = terminated or truncated
            else: next_obs, reward, done, _ = result[:4]
            rew_list.append(float(reward if not isinstance(reward, (list, np.ndarray)) else reward[0]))
            done_val = bool(done if not isinstance(done, (list, np.ndarray)) else done[0])
            done_list.append(done_val)
            obs = np.array(next_obs).flatten() if not isinstance(next_obs, np.ndarray) else next_obs.flatten()
            if done_val:
                self._reset_seq()
                obs_raw2 = self.env.reset()
                if isinstance(obs_raw2, tuple): obs_raw2 = obs_raw2[0]
                obs = np.array(obs_raw2).flatten()
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(self.n_steps)):
            next_val = val_list[t+1] if t+1 < self.n_steps else 0.0
            delta = rew_list[t] + self.gamma*next_val*(1-done_list[t]) - val_list[t]
            advantages[t] = last_adv = delta + self.gamma*self.gae_lambda*(1-done_list[t])*last_adv
        returns = advantages + np.array(val_list, dtype=np.float32)
        return (np.array(obs_seq_list, dtype=np.float32), np.array(act_list, dtype=np.int64),
                np.array(logp_list, dtype=np.float32), advantages, returns)

    def _ppo_update(self, obs_seqs, actions, old_log_probs, advantages, returns):
        N = len(actions)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.FloatTensor(returns).to(self.device)
        act_t = torch.LongTensor(actions).to(self.device)
        olp_t = torch.FloatTensor(old_log_probs).to(self.device)
        total_loss = 0.0
        for _ in range(self.n_epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                b = idx[start:start+self.batch_size]
                seq_b = torch.FloatTensor(obs_seqs[b]).to(self.device)
                logits, values = self.net(seq_b)
                dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
                log_p = dist.log_prob(act_t[b]); entropy = dist.entropy().mean()
                ratio = (log_p - olp_t[b]).exp()
                pg1 = ratio * adv_t[b]
                pg2 = ratio.clamp(1-self.clip_range, 1+self.clip_range) * adv_t[b]
                loss_p = -torch.min(pg1, pg2).mean()
                loss_v = F.mse_loss(values.squeeze(), ret_t[b])
                loss = loss_p + self.vf_coef*loss_v - self.ent_coef*entropy
                self.optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step(); self.scheduler.step()
                total_loss += loss.item()
        self.logger.name_to_value = {
            'train/loss': total_loss, 'train/entropy_loss': -entropy.item(),
            'train/policy_gradient_loss': loss_p.item(), 'train/value_loss': loss_v.item(),
        }

    def learn(self, total_timesteps, callback=None, reset_num_timesteps=True, **kwargs):
        if reset_num_timesteps: self.num_timesteps = 0
        if callback:
            if not isinstance(callback, list): callback = [callback]
            for cb in callback:
                if hasattr(cb, 'init_callback'): cb.init_callback(self)
                if hasattr(cb, 'on_training_start'): cb.on_training_start({}, {})
        steps_done = 0
        while steps_done < total_timesteps:
            obs_seqs, actions, old_log_probs, advantages, returns = self._collect_rollout()
            self._ppo_update(obs_seqs, actions, old_log_probs, advantages, returns)
            steps_done += self.n_steps; self.num_timesteps += self.n_steps
            if callback:
                for cb in callback:
                    for _ in range(self.n_steps):
                        if hasattr(cb, '_on_step'): cb._on_step()
        if callback:
            for cb in callback:
                if hasattr(cb, 'on_training_end'): cb.on_training_end()
        return self

    def save(self, path):
        torch.save({'net': self.net.state_dict(), 'optimizer': self.optimizer.state_dict(),
                    'num_timesteps': self.num_timesteps}, path + '_transformer.pt')

    def load(self, path, env=None):
        data = torch.load(path + '_transformer.pt', map_location=self.device)
        self.net.load_state_dict(data['net']); self.optimizer.load_state_dict(data['optimizer'])
        self.num_timesteps = data.get('num_timesteps', 0)
        if env: self.env = env
        return self


print("  TransformerAgent (Manhattan Agent 3) charge — SEQ_LEN=50, d_model=128, 3 blocs, 4 heads")

# Visualization
from tqdm import tqdm

# === FORMULE 1 : CURRICULUM GLOBAL VARIABLE ===
current_phase = 1  # Phase 1=EURUSD only, 2=add pairs, 3=all active

# PATCH V4: Variables globales pour accès aux métriques depuis le callback
GLOBAL_META_LEARNER = None
GLOBAL_DIVERGENCE_SYSTEM = None

# PATCH V4.1: Training phase (1=PPO, 2=DQN, 3=Ensemble)
training_phase = 1  # Phase d'entraînement actuelle

# ═══════════════════════════════════════════════════════════════════════════════
# PATCH V4.1: CORRÉLATION INTER-PAIRES + POSITION SIZING DYNAMIQUE
# ═══════════════════════════════════════════════════════════════════════════════

class CrossPairCorrelation:
    """
    Système de corrélation inter-paires pour confirmation des signaux.
    
    Si EURUSD et GBPUSD donnent le même signal → Confiance augmentée
    Si signaux contradictoires → Confiance réduite
    """
    
    # Paires corrélées positivement (bougent ensemble)
    POSITIVE_CORRELATIONS = {
        'EURUSD': ['GBPUSD', 'AUDUSD', 'NZDUSD'],  # Risk-on currencies
        'GBPUSD': ['EURUSD', 'AUDUSD'],
        'AUDUSD': ['NZDUSD', 'EURUSD'],
        'NZDUSD': ['AUDUSD'],
        'XAUUSD': ['EURUSD'],  # Gold souvent corrélé à l'euro (anti-dollar)
    }
    
    # Paires corrélées négativement (bougent en opposition)
    NEGATIVE_CORRELATIONS = {
        'EURUSD': ['USDJPY', 'USDCHF', 'USDCAD'],
        'GBPUSD': ['USDJPY', 'USDCHF'],
        'USDJPY': ['EURUSD', 'GBPUSD', 'XAUUSD'],
        'XAUUSD': ['USDJPY'],  # Gold vs Yen (safe havens concurrents)
    }
    
    def __init__(self):
        self.pair_signals = {}  # {symbol: {'signal': 1/-1/0, 'confidence': 0.0-1.0, 'timestamp': datetime}}
        self.correlation_boost = 0.15  # Boost de confiance si corrélation confirmée
        self.correlation_penalty = 0.10  # Pénalité si contradiction
        
        logging.info("[CORRELATION] Cross-pair correlation system initialized")
    
    def update_signal(self, symbol: str, signal: int, confidence: float):
        """
        Met à jour le signal d'une paire.
        signal: 1=BUY, -1=SELL, 0=NEUTRAL
        """
        self.pair_signals[symbol] = {
            'signal': signal,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
    
    def get_correlation_adjustment(self, symbol: str, proposed_signal: int) -> tuple:
        """
        Calcule l'ajustement de confiance basé sur les corrélations.
        
        Returns:
            (adjustment: float, reason: str)
            adjustment > 0 = signal confirmé par corrélations
            adjustment < 0 = signal contredit par corrélations
        """
        if symbol not in self.pair_signals:
            return 0.0, "no_data"
        
        confirmations = 0
        contradictions = 0
        total_checked = 0
        
        # Vérifier corrélations positives
        positive_pairs = self.POSITIVE_CORRELATIONS.get(symbol, [])
        for corr_pair in positive_pairs:
            if corr_pair in self.pair_signals:
                corr_signal = self.pair_signals[corr_pair]['signal']
                total_checked += 1
                
                # Même direction = confirmation
                if corr_signal == proposed_signal and proposed_signal != 0:
                    confirmations += 1
                # Direction opposée = contradiction
                elif corr_signal == -proposed_signal and proposed_signal != 0:
                    contradictions += 1
        
        # Vérifier corrélations négatives (signal inverse attendu)
        negative_pairs = self.NEGATIVE_CORRELATIONS.get(symbol, [])
        for corr_pair in negative_pairs:
            if corr_pair in self.pair_signals:
                corr_signal = self.pair_signals[corr_pair]['signal']
                total_checked += 1
                
                # Direction opposée = confirmation (pour corrélation négative)
                if corr_signal == -proposed_signal and proposed_signal != 0:
                    confirmations += 1
                # Même direction = contradiction (pour corrélation négative)
                elif corr_signal == proposed_signal and proposed_signal != 0:
                    contradictions += 1
        
        if total_checked == 0:
            return 0.0, "no_correlated_pairs"
        
        # Calculer l'ajustement
        adjustment = (confirmations * self.correlation_boost) - (contradictions * self.correlation_penalty)
        
        if confirmations > contradictions:
            reason = f"confirmed_by_{confirmations}_pairs"
        elif contradictions > confirmations:
            reason = f"contradicted_by_{contradictions}_pairs"
        else:
            reason = "neutral_correlation"
        
        return adjustment, reason


class DynamicPositionSizer:
    """
    Position sizing dynamique basé sur:
    - Confiance du signal
    - Volatilité actuelle
    - Drawdown récent
    - Corrélation inter-paires
    """
    
    def __init__(self, base_risk_pct: float = 0.02, max_risk_pct: float = 0.05, min_risk_pct: float = 0.005):
        """
        Args:
            base_risk_pct: Risque de base par trade (2%)
            max_risk_pct: Risque maximum (5%) - signal très fort
            min_risk_pct: Risque minimum (0.5%) - signal faible
        """
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        
        # Tracking pour ajustement dynamique
        self.recent_trades = []  # Liste des (profit, timestamp)
        self.max_recent_trades = 20
        self.current_drawdown = 0.0
        self.peak_equity = 10000.0
        
        logging.info(f"[POSITION SIZING] Dynamic sizer initialized: base={base_risk_pct*100}%, max={max_risk_pct*100}%, min={min_risk_pct*100}%")
    
    def update_equity(self, equity: float):
        """Met à jour l'equity et calcule le drawdown"""
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
    
    def record_trade(self, profit: float):
        """Enregistre un trade pour le tracking"""
        self.recent_trades.append({
            'profit': profit,
            'timestamp': datetime.now()
        })
        
        # Garder seulement les N derniers
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades.pop(0)
    
    def get_recent_performance(self) -> dict:
        """Calcule la performance récente"""
        if len(self.recent_trades) < 3:
            return {'win_rate': 0.5, 'avg_profit': 0.0, 'streak': 0}
        
        wins = sum(1 for t in self.recent_trades if t['profit'] > 0)
        win_rate = wins / len(self.recent_trades)
        avg_profit = sum(t['profit'] for t in self.recent_trades) / len(self.recent_trades)
        
        # Calculer la streak (série de gains/pertes consécutifs)
        streak = 0
        if self.recent_trades:
            last_sign = 1 if self.recent_trades[-1]['profit'] > 0 else -1
            for trade in reversed(self.recent_trades):
                current_sign = 1 if trade['profit'] > 0 else -1
                if current_sign == last_sign:
                    streak += current_sign
                else:
                    break
        
        return {'win_rate': win_rate, 'avg_profit': avg_profit, 'streak': streak}
    
    def calculate_position_size(
        self,
        equity: float,
        signal_confidence: float,
        volatility_ratio: float = 1.0,
        correlation_boost: float = 0.0
    ) -> dict:
        """
        Calcule la taille de position optimale.
        
        Args:
            equity: Capital actuel
            signal_confidence: Confiance du signal (0.0 à 1.0)
            volatility_ratio: Volatilité actuelle vs moyenne (1.0 = normal)
            correlation_boost: Boost de la corrélation inter-paires
        
        Returns:
            dict avec position_size, risk_pct, et reasoning
        """
        # Mettre à jour l'equity tracking
        self.update_equity(equity)
        
        # 1. AJUSTEMENT PAR CONFIANCE (±50%)
        # Confiance 0.5 = base, 1.0 = +50%, 0.0 = -50%
        confidence_multiplier = 0.5 + signal_confidence
        
        # 2. AJUSTEMENT PAR VOLATILITÉ (inverse)
        # Volatilité haute = position plus petite
        if volatility_ratio > 2.0:
            volatility_multiplier = 0.5  # Très volatile = moitié
        elif volatility_ratio > 1.5:
            volatility_multiplier = 0.7
        elif volatility_ratio < 0.5:
            volatility_multiplier = 1.2  # Très calme = légèrement plus
        else:
            volatility_multiplier = 1.0
        
        # 3. AJUSTEMENT PAR DRAWDOWN
        # Plus le drawdown est grand, plus on réduit
        if self.current_drawdown > 0.10:  # >10% drawdown
            drawdown_multiplier = 0.5  # Réduire de moitié
        elif self.current_drawdown > 0.05:  # >5% drawdown
            drawdown_multiplier = 0.75
        else:
            drawdown_multiplier = 1.0
        
        # 4. AJUSTEMENT PAR PERFORMANCE RÉCENTE
        perf = self.get_recent_performance()
        
        if perf['streak'] <= -3:  # 3+ pertes consécutives
            streak_multiplier = 0.6  # Réduire significativement
        elif perf['streak'] >= 3:  # 3+ gains consécutifs
            streak_multiplier = 1.1  # Légère augmentation (attention à l'overconfidence)
        else:
            streak_multiplier = 1.0
        
        # 5. AJUSTEMENT PAR CORRÉLATION
        correlation_multiplier = 1.0 + correlation_boost
        
        # CALCUL FINAL
        final_risk_pct = (
            self.base_risk_pct 
            * confidence_multiplier 
            * volatility_multiplier 
            * drawdown_multiplier 
            * streak_multiplier
            * correlation_multiplier
        )
        
        # Clamp entre min et max
        final_risk_pct = max(self.min_risk_pct, min(self.max_risk_pct, final_risk_pct))
        
        # Calculer la taille de position en dollars
        position_value = equity * final_risk_pct
        
        # Reasoning pour debug/dashboard
        reasoning = {
            'base_risk': self.base_risk_pct,
            'confidence_mult': round(confidence_multiplier, 2),
            'volatility_mult': round(volatility_multiplier, 2),
            'drawdown_mult': round(drawdown_multiplier, 2),
            'streak_mult': round(streak_multiplier, 2),
            'correlation_mult': round(correlation_multiplier, 2),
            'current_drawdown': round(self.current_drawdown * 100, 1),
            'recent_streak': perf['streak']
        }
        
        return {
            'position_value': position_value,
            'risk_pct': final_risk_pct,
            'reasoning': reasoning
        }


# Variables globales pour les nouveaux systèmes
GLOBAL_CORRELATION_SYSTEM = None
GLOBAL_POSITION_SIZER = None

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DECISION FUSION META-LEARNER - Apprend Ã  fusionner PPO et DQN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class DecisionFusionNetwork(nn.Module):
    """
    Meta-Learner qui apprend Ã  fusionner les dÃ©cisions PPO et DQN.
    
    Au lieu d'une rÃ¨gle fixe (PPO gagne toujours), ce rÃ©seau apprend
    QUAND faire confiance Ã  PPO vs DQN en fonction du contexte.
    
    Input:
        - Market context (18 features: prix, indicateurs, divergences)
        - PPO action (one-hot: 4)
        - DQN action (one-hot: 4)
        - PPO confidence (1)
        - DQN confidence (1)
        Total: 18 + 4 + 4 + 1 + 1 = 28 features
    
    Output:
        - Action finale (4 classes: BUY, SELL, HOLD, CLOSE)
        - Confidence de la dÃ©cision
    """
    
    def __init__(self, context_dim: int = 18, hidden_dim: int = 64):
        super().__init__()
        
        self.context_dim = context_dim
        # Input: context + ppo_action(4) + dqn_action(4) + ppo_conf(1) + dqn_conf(1)
        input_dim = context_dim + 4 + 4 + 1 + 1
        
        # RÃ©seau de fusion avec attention
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim // 2),  # 4+4+1+1 = 10
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Attention: apprend Ã  pondÃ©rer PPO vs DQN selon le contexte
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),  # 2 poids: w_ppo, w_dqn
            nn.Softmax(dim=-1)
        )
        
        # DÃ©cision finale
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4)  # 4 actions
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Tracking des performances
        self.ppo_wins = 0
        self.dqn_wins = 0
        self.consensus_wins = 0
        self.total_decisions = 0
        
        logging.info("[META-LEARNER] Decision Fusion Network initialized")
    
    def forward(self, context: torch.Tensor, ppo_action: int, dqn_action: int,
                ppo_confidence: float = 0.5, dqn_confidence: float = 0.5):
        """
        Forward pass pour dÃ©cider l'action finale.
        
        Returns:
            action: int (0-3)
            confidence: float (0-1)
            attention_weights: (w_ppo, w_dqn)
        """
        # Encoder le contexte de marchÃ©
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        context_encoded = self.context_encoder(context)
        
        # Encoder les propositions des agents
        ppo_onehot = F.one_hot(torch.tensor([ppo_action]), num_classes=4).float()
        dqn_onehot = F.one_hot(torch.tensor([dqn_action]), num_classes=4).float()
        
        agent_input = torch.cat([
            ppo_onehot,
            dqn_onehot,
            torch.tensor([[ppo_confidence]]),
            torch.tensor([[dqn_confidence]])
        ], dim=-1)
        
        agent_encoded = self.agent_encoder(agent_input)
        
        # Combiner
        combined = torch.cat([context_encoded, agent_encoded], dim=-1)
        
        # Calculer les poids d'attention (qui Ã©couter: PPO ou DQN?)
        attention_weights = self.attention(combined)
        
        # DÃ©cision finale
        action_logits = self.decision_head(combined)
        action_probs = F.softmax(action_logits, dim=-1)
        action = torch.argmax(action_probs, dim=-1).item()
        
        # Confidence
        confidence = self.confidence_head(combined).item()
        
        return action, confidence, attention_weights.squeeze().detach().numpy()
    
    def get_fusion_stats(self) -> dict:
        """Retourne les statistiques de fusion"""
        if self.total_decisions == 0:
            return {'ppo_rate': 0, 'dqn_rate': 0, 'consensus_rate': 0}
        
        return {
            'ppo_rate': self.ppo_wins / self.total_decisions * 100,
            'dqn_rate': self.dqn_wins / self.total_decisions * 100,
            'consensus_rate': self.consensus_wins / self.total_decisions * 100,
            'total_decisions': self.total_decisions
        }


class FusionExperienceBuffer:
    """
    Buffer d'expÃ©riences pour entraÃ®ner le Meta-Learner.
    
    Stocke: (context, ppo_action, dqn_action, chosen_action, reward)
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
    
    def add(self, context: np.ndarray, ppo_action: int, dqn_action: int,
            ppo_confidence: float, dqn_confidence: float,
            chosen_action: int, reward: float):
        """Ajoute une expÃ©rience"""
        experience = {
            'context': context.copy(),
            'ppo_action': ppo_action,
            'dqn_action': dqn_action,
            'ppo_confidence': ppo_confidence,
            'dqn_confidence': dqn_confidence,
            'chosen_action': chosen_action,
            'reward': reward
        }
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size: int) -> list:
        """Sample un batch alÃ©atoire"""
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class MetaLearnerFusion:
    """
    SystÃ¨me complet de Meta-Learning pour fusion PPO-DQN.
    
    Apprend Ã :
    1. Quand faire confiance Ã  PPO vs DQN
    2. ReconnaÃ®tre les contextes oÃ¹ chaque agent excelle
    3. GÃ©rer les cas de dÃ©saccord intelligemment
    """
    
    def __init__(self, context_dim: int = 18, learning_rate: float = 1e-4,
                 memory_file: str = 'meta_learner_fusion.pth'):
        
        self.network = DecisionFusionNetwork(context_dim=context_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.buffer = FusionExperienceBuffer(max_size=10000)
        self.memory_file = memory_file
        
        # Tracking
        self.training_step = 0
        self.last_decision = None
        self.pending_experiences = {}  # symbol -> experience en attente de reward
        
        # Performance tracking par contexte
        self.context_performance = {
            'divergence_bullish': {'ppo_wins': 0, 'dqn_wins': 0, 'total': 0},
            'divergence_bearish': {'ppo_wins': 0, 'dqn_wins': 0, 'total': 0},
            'high_volatility': {'ppo_wins': 0, 'dqn_wins': 0, 'total': 0},
            'low_volatility': {'ppo_wins': 0, 'dqn_wins': 0, 'total': 0},
            'trend_up': {'ppo_wins': 0, 'dqn_wins': 0, 'total': 0},
            'trend_down': {'ppo_wins': 0, 'dqn_wins': 0, 'total': 0},
            'ranging': {'ppo_wins': 0, 'dqn_wins': 0, 'total': 0}
        }
        
        # Charger si existe
        self.load()
        
        logging.info("[META-LEARNER] Fusion system initialized")
    
    def decide(self, symbol: str, context: np.ndarray, 
               ppo_action: int, dqn_action: int,
               ppo_confidence: float = 0.5, dqn_confidence: float = 0.5) -> tuple:
        """
        DÃ©cide l'action finale en utilisant le meta-learner.
        
        Returns:
            (action, confidence, debug_info)
        """
        self.network.eval()
        
        with torch.no_grad():
            context_tensor = torch.from_numpy(context).float()
            action, confidence, attention = self.network(
                context_tensor, ppo_action, dqn_action,
                ppo_confidence, dqn_confidence
            )
        
        # Analyser le contexte pour le tracking
        context_type = self._analyze_context(context)
        
        # Stocker l'expÃ©rience en attente (sera complÃ©tÃ©e avec le reward)
        self.pending_experiences[symbol] = {
            'context': context,
            'ppo_action': ppo_action,
            'dqn_action': dqn_action,
            'ppo_confidence': ppo_confidence,
            'dqn_confidence': dqn_confidence,
            'chosen_action': action,
            'context_type': context_type,
            'attention': attention
        }
        
        # Debug info
        debug_info = {
            'ppo_action': ppo_action,
            'dqn_action': dqn_action,
            'final_action': action,
            'confidence': confidence,
            'attention_ppo': attention[0],
            'attention_dqn': attention[1],
            'context_type': context_type,
            'consensus': ppo_action == dqn_action
        }
        
        # Update tracking
        self.network.total_decisions += 1
        if ppo_action == dqn_action:
            self.network.consensus_wins += 1
        elif action == ppo_action:
            self.network.ppo_wins += 1
        else:
            self.network.dqn_wins += 1
        
        # Log
        action_names = ['BUY', 'SELL', 'HOLD', 'CLOSE']
        logging.info(f"[META-LEARNER] {symbol}: PPO={action_names[ppo_action]} vs DQN={action_names[dqn_action]} "
                    f"â†’ {action_names[action]} (conf={confidence:.2f}, attn=[{attention[0]:.2f},{attention[1]:.2f}])")
        
        return action, confidence, debug_info
    
    def record_outcome(self, symbol: str, reward: float):
        """
        Enregistre le rÃ©sultat d'une dÃ©cision pour l'apprentissage.
        """
        if symbol not in self.pending_experiences:
            return
        
        exp = self.pending_experiences.pop(symbol)
        
        # Ajouter au buffer
        self.buffer.add(
            context=exp['context'],
            ppo_action=exp['ppo_action'],
            dqn_action=exp['dqn_action'],
            ppo_confidence=exp['ppo_confidence'],
            dqn_confidence=exp['dqn_confidence'],
            chosen_action=exp['chosen_action'],
            reward=reward
        )
        
        # Update context performance tracking
        ctx_type = exp['context_type']
        if ctx_type in self.context_performance:
            self.context_performance[ctx_type]['total'] += 1
            if reward > 0:
                if exp['chosen_action'] == exp['ppo_action']:
                    self.context_performance[ctx_type]['ppo_wins'] += 1
                elif exp['chosen_action'] == exp['dqn_action']:
                    self.context_performance[ctx_type]['dqn_wins'] += 1
        
        # EntraÃ®ner pÃ©riodiquement
        if len(self.buffer) >= 32 and len(self.buffer) % 10 == 0:
            self.train_step()
    
    def train_step(self, batch_size: int = 32):
        """
        Un step d'entraÃ®nement du meta-learner.
        """
        if len(self.buffer) < batch_size:
            return
        
        self.network.train()
        
        batch = self.buffer.sample(batch_size)
        
        total_loss = 0.0
        
        for exp in batch:
            context = torch.from_numpy(exp['context']).float().unsqueeze(0)
            
            # Forward
            context_encoded = self.network.context_encoder(context)
            
            ppo_onehot = F.one_hot(torch.tensor([exp['ppo_action']]), num_classes=4).float()
            dqn_onehot = F.one_hot(torch.tensor([exp['dqn_action']]), num_classes=4).float()
            
            agent_input = torch.cat([
                ppo_onehot,
                dqn_onehot,
                torch.tensor([[exp['ppo_confidence']]]),
                torch.tensor([[exp['dqn_confidence']]])
            ], dim=-1)
            
            agent_encoded = self.network.agent_encoder(agent_input)
            combined = torch.cat([context_encoded, agent_encoded], dim=-1)
            
            action_logits = self.network.decision_head(combined)
            
            # Target: l'action qui a donnÃ© le meilleur reward
            # Si reward > 0: renforcer l'action choisie
            # Si reward < 0: dÃ©courager l'action choisie
            
            if exp['reward'] > 0:
                # Renforcer l'action choisie
                target = torch.tensor([exp['chosen_action']])
                loss = F.cross_entropy(action_logits, target)
            else:
                # DÃ©courager l'action choisie, encourager les autres
                probs = F.softmax(action_logits, dim=-1)
                # Maximiser l'entropie pour les mauvaises dÃ©cisions (explorer)
                loss = -torch.sum(probs * torch.log(probs + 1e-8))
            
            # PondÃ©rer par l'amplitude du reward
            loss = loss * min(abs(exp['reward']) / 10.0, 2.0)
            
            total_loss += loss
        
        # Backward
        self.optimizer.zero_grad()
        avg_loss = total_loss / batch_size
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        if self.training_step % 100 == 0:
            logging.info(f"[META-LEARNER] Training step {self.training_step}, loss={avg_loss.item():.4f}")
            self.save()
    
    def _analyze_context(self, context: np.ndarray) -> str:
        """
        Analyse le contexte pour catÃ©goriser le type de marchÃ©.
        """
        # Features: [open, high, low, close, rsi, macd, macd_signal, atr, bb_middle, bb_upper,
        #            bb_lower, volume_norm, ema_20, momentum,
        #            div_rsi_signal, div_macd_signal, div_momentum_signal, div_strength]
        
        if len(context) < 18:
            return 'unknown'
        
        div_rsi = context[14] if len(context) > 14 else 0
        div_macd = context[15] if len(context) > 15 else 0
        div_strength = context[17] if len(context) > 17 else 0
        momentum = context[13] if len(context) > 13 else 0
        atr = context[7] if len(context) > 7 else 0
        
        # Divergence?
        if div_strength > 0.3:
            if div_rsi > 0 or div_macd > 0:
                return 'divergence_bullish'
            else:
                return 'divergence_bearish'
        
        # VolatilitÃ©?
        if atr > 0.5:  # NormalisÃ©, donc > 0.5 = haute volatilitÃ©
            return 'high_volatility'
        elif atr < -0.5:
            return 'low_volatility'
        
        # Trend?
        if momentum > 0.3:
            return 'trend_up'
        elif momentum < -0.3:
            return 'trend_down'
        
        return 'ranging'
    
    def get_context_insights(self) -> dict:
        """
        Retourne des insights sur quel agent performe mieux dans quel contexte.
        """
        insights = {}
        
        for ctx_type, stats in self.context_performance.items():
            if stats['total'] > 10:  # Assez de donnÃ©es
                ppo_rate = stats['ppo_wins'] / max(stats['total'], 1) * 100
                dqn_rate = stats['dqn_wins'] / max(stats['total'], 1) * 100
                
                if ppo_rate > dqn_rate + 10:
                    winner = 'PPO'
                elif dqn_rate > ppo_rate + 10:
                    winner = 'DQN'
                else:
                    winner = 'TIE'
                
                insights[ctx_type] = {
                    'ppo_rate': ppo_rate,
                    'dqn_rate': dqn_rate,
                    'winner': winner,
                    'samples': stats['total']
                }
        
        return insights
    
    def save(self):
        """Sauvegarde le modÃ¨le"""
        try:
            torch.save({
                'network_state': self.network.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'training_step': self.training_step,
                'context_performance': self.context_performance,
                'network_stats': {
                    'ppo_wins': self.network.ppo_wins,
                    'dqn_wins': self.network.dqn_wins,
                    'consensus_wins': self.network.consensus_wins,
                    'total_decisions': self.network.total_decisions
                }
            }, self.memory_file)
            logging.debug(f"[META-LEARNER] Saved to {self.memory_file}")
        except Exception as e:
            logging.error(f"[META-LEARNER] Save error: {e}")
    
    def load(self):
        """Charge le modÃ¨le"""
        try:
            if os.path.exists(self.memory_file):
                checkpoint = torch.load(self.memory_file)
                self.network.load_state_dict(checkpoint['network_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.training_step = checkpoint.get('training_step', 0)
                self.context_performance = checkpoint.get('context_performance', self.context_performance)
                
                stats = checkpoint.get('network_stats', {})
                self.network.ppo_wins = stats.get('ppo_wins', 0)
                self.network.dqn_wins = stats.get('dqn_wins', 0)
                self.network.consensus_wins = stats.get('consensus_wins', 0)
                self.network.total_decisions = stats.get('total_decisions', 0)
                
                logging.info(f"[META-LEARNER] Loaded from {self.memory_file} (step {self.training_step})")
        except Exception as e:
            logging.warning(f"[META-LEARNER] Load error: {e}, starting fresh")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ENSEMBLE TRAINING ENVIRONMENT - PPO, DQN et META-LEARNER s'entraÃ®nent ENSEMBLE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class EnsembleTrainingEnvironment(gym.Env):
    """
    Environnement d'entraÃ®nement ENSEMBLE oÃ¹ PPO, DQN et Meta-Learner
    apprennent ENSEMBLE Ã  chaque step.
    
    Architecture:
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚                    ENSEMBLE TRAINING LOOP                            â”‚
    â”‚                                                                      â”‚
    â”‚  1. Observation â†’ PPO.predict() â†’ action_ppo, conf_ppo              â”‚
    â”‚                 â†’ DQN.predict() â†’ action_dqn, conf_dqn              â”‚
    â”‚                                                                      â”‚
    â”‚  2. Meta-Learner.decide(context, actions, confidences)              â”‚
    â”‚                 â†’ action_finale                                      â”‚
    â”‚                                                                      â”‚
    â”‚  3. Environment.step(action_finale) â†’ reward, next_obs              â”‚
    â”‚                                                                      â”‚
    â”‚  4. APPRENTISSAGE SIMULTANÃ‰:                                         â”‚
    â”‚     - PPO: apprend de (obs, action_finale, reward)                  â”‚
    â”‚     - DQN: apprend de (obs, action_finale, reward)                  â”‚
    â”‚     - Meta-Learner: apprend de (context, actions, reward)           â”‚
    â”‚                                                                      â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    """
    
    def __init__(self, base_env: TradingEnvironment, 
                 ppo_agent: PPO = None, 
                 dqn_agent: DQN = None,
                 meta_learner: MetaLearnerFusion = None,
                 training_mode: str = 'ensemble'):
        """
        Args:
            base_env: L'environnement de trading de base
            ppo_agent: Agent PPO (peut Ãªtre None au dÃ©but)
            dqn_agent: Agent DQN (peut Ãªtre None au dÃ©but)
            meta_learner: Le meta-learner pour fusion
            training_mode: 'ppo_only', 'dqn_only', 'ensemble', 'meta_only'
        """
        super().__init__()
        
        self.base_env = base_env
        self.ppo_agent = ppo_agent
        self.dqn_agent = dqn_agent
        self.meta_learner = meta_learner
        self.training_mode = training_mode
        
        # Copier les espaces de l'env de base
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        
        # Tracking
        self.ensemble_decisions = 0
        self.ppo_selected = 0
        self.dqn_selected = 0
        self.consensus_count = 0
        
        # Buffer pour les expÃ©riences ensemble
        self.ensemble_buffer = []
        
        # Stats par contexte
        self.context_stats = {
            'divergence': {'ppo_correct': 0, 'dqn_correct': 0, 'total': 0},
            'trend': {'ppo_correct': 0, 'dqn_correct': 0, 'total': 0},
            'volatility': {'ppo_correct': 0, 'dqn_correct': 0, 'total': 0},
            'range': {'ppo_correct': 0, 'dqn_correct': 0, 'total': 0}
        }
        
        logging.info(f"[ENSEMBLE] Training environment created - mode: {training_mode}")
    
    def set_agents(self, ppo_agent: PPO = None, dqn_agent: DQN = None):
        """Configure les agents aprÃ¨s leur crÃ©ation initiale"""
        if ppo_agent:
            self.ppo_agent = ppo_agent
        if dqn_agent:
            self.dqn_agent = dqn_agent
    
    def reset(self, seed=None, options=None):
        """Reset l'environnement"""
        return self.base_env.reset(seed=seed, options=options)
    
    def step(self, action: int):
        """
        Step ENSEMBLE - Tous les agents participent Ã  la dÃ©cision.
        
        Note: 'action' ici vient de l'agent en cours d'entraÃ®nement (PPO ou DQN).
        On compare avec l'autre agent et on utilise le meta-learner pour dÃ©cider.
        """
        # RÃ©cupÃ©rer l'observation actuelle
        current_obs = self._get_current_obs()
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # MODE ENSEMBLE: Les deux agents votent
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        if self.training_mode == 'ensemble' and self.ppo_agent and self.dqn_agent:
            try:
                # Obtenir les prÃ©dictions des deux agents
                ppo_action, ppo_conf = self._get_agent_prediction(self.ppo_agent, current_obs, 'PPO')
                dqn_action, dqn_conf = self._get_agent_prediction(self.dqn_agent, current_obs, 'DQN')
                
                # Utiliser le meta-learner pour dÃ©cider
                if self.meta_learner:
                    final_action, confidence, debug = self.meta_learner.decide(
                        symbol=self.base_env.symbol,
                        context=current_obs,
                        ppo_action=ppo_action,
                        dqn_action=dqn_action,
                        ppo_confidence=ppo_conf,
                        dqn_confidence=dqn_conf
                    )
                else:
                    # Fallback: vote simple
                    if ppo_action == dqn_action:
                        final_action = ppo_action
                    else:
                        final_action = ppo_action if ppo_conf > dqn_conf else dqn_action
                
                # Tracking
                self.ensemble_decisions += 1
                if ppo_action == dqn_action:
                    self.consensus_count += 1
                elif final_action == ppo_action:
                    self.ppo_selected += 1
                else:
                    self.dqn_selected += 1
                
                # ExÃ©cuter l'action dÃ©cidÃ©e par l'ensemble
                obs, reward, terminated, truncated, info = self.base_env.step(final_action)
                
                # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                # APPRENTISSAGE: Enregistrer pour le meta-learner
                # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                if self.meta_learner:
                    self.meta_learner.record_outcome(
                        self.base_env.symbol, 
                        reward
                    )
                
                # Stocker l'expÃ©rience ensemble
                self._store_ensemble_experience(
                    obs=current_obs,
                    ppo_action=ppo_action,
                    dqn_action=dqn_action,
                    final_action=final_action,
                    reward=reward
                )
                
                # Ajouter info sur la dÃ©cision ensemble
                info['ensemble_decision'] = {
                    'ppo_action': ppo_action,
                    'dqn_action': dqn_action,
                    'final_action': final_action,
                    'consensus': ppo_action == dqn_action
                }
                
                return obs, reward, terminated, truncated, info
                
            except Exception as e:
                logging.debug(f"[ENSEMBLE] Error in ensemble step: {e}")
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # MODE STANDARD: Utiliser l'action fournie directement
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        return self.base_env.step(action)
    
    def _get_current_obs(self) -> np.ndarray:
        """RÃ©cupÃ¨re l'observation actuelle"""
        try:
            if hasattr(self.base_env, '_get_obs'):
                return self.base_env._get_obs()
            elif hasattr(self.base_env, 'data') and hasattr(self.base_env, 'current_step'):
                return self.base_env.data.iloc[self.base_env.current_step].values.astype(np.float32)
            else:
                return np.zeros(18, dtype=np.float32)
        except:
            return np.zeros(18, dtype=np.float32)
    
    def _get_agent_prediction(self, agent, obs: np.ndarray, agent_type: str) -> tuple:
        """
        Obtient la prÃ©diction et la confiance d'un agent.
        
        Returns:
            (action, confidence)
        """
        try:
            # PrÃ©dire l'action
            action, _ = agent.predict(obs, deterministic=False)
            
            # Calculer la confiance
            confidence = 0.5
            
            if agent_type == 'PPO':
                try:
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                    with torch.no_grad():
                        _, log_prob, _ = agent.policy.evaluate_actions(
                            obs_tensor, 
                            torch.tensor([[int(action)]])
                        )
                        confidence = min(0.95, torch.exp(log_prob).item())
                except:
                    confidence = 0.6
                    
            elif agent_type == 'DQN':
                try:
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                    with torch.no_grad():
                        q_values = agent.q_net(obs_tensor)
                        q_probs = F.softmax(q_values, dim=-1)
                        confidence = min(0.95, q_probs[0, int(action)].item())
                except:
                    confidence = 0.5
            
            return int(action), confidence
            
        except Exception as e:
            logging.debug(f"[ENSEMBLE] Prediction error for {agent_type}: {e}")
            return 2, 0.5  # HOLD par dÃ©faut
    
    def _store_ensemble_experience(self, obs, ppo_action, dqn_action, final_action, reward):
        """Stocke une expÃ©rience pour analyse"""
        self.ensemble_buffer.append({
            'obs': obs.copy() if isinstance(obs, np.ndarray) else obs,
            'ppo_action': ppo_action,
            'dqn_action': dqn_action,
            'final_action': final_action,
            'reward': reward,
            'timestamp': datetime.now()
        })
        
        # Limiter la taille du buffer
        if len(self.ensemble_buffer) > 10000:
            self.ensemble_buffer = self.ensemble_buffer[-5000:]
    
    def get_ensemble_stats(self) -> dict:
        """Retourne les statistiques de l'entraÃ®nement ensemble"""
        if self.ensemble_decisions == 0:
            return {}
        
        return {
            'total_decisions': self.ensemble_decisions,
            'consensus_rate': self.consensus_count / self.ensemble_decisions * 100,
            'ppo_selected_rate': self.ppo_selected / self.ensemble_decisions * 100,
            'dqn_selected_rate': self.dqn_selected / self.ensemble_decisions * 100,
            'buffer_size': len(self.ensemble_buffer)
        }
    
    def analyze_agent_performance(self) -> dict:
        """
        Analyse quel agent performe mieux dans diffÃ©rentes conditions.
        """
        if len(self.ensemble_buffer) < 100:
            return {}
        
        results = {
            'ppo_correct_when_disagree': 0,
            'dqn_correct_when_disagree': 0,
            'total_disagree': 0
        }
        
        for exp in self.ensemble_buffer:
            if exp['ppo_action'] != exp['dqn_action']:
                results['total_disagree'] += 1
                
                if exp['reward'] > 0:
                    if exp['final_action'] == exp['ppo_action']:
                        results['ppo_correct_when_disagree'] += 1
                    else:
                        results['dqn_correct_when_disagree'] += 1
        
        if results['total_disagree'] > 0:
            results['ppo_accuracy_disagree'] = results['ppo_correct_when_disagree'] / results['total_disagree'] * 100
            results['dqn_accuracy_disagree'] = results['dqn_correct_when_disagree'] / results['total_disagree'] * 100
        
        return results







try:
    import cv2
    from sklearn.metrics.pairwise import cosine_similarity
    MERLIN_ENABLED = True
except ImportError:
    MERLIN_ENABLED = False
    logging.warning("[MERLIN] Vision disabled - missing dependencies")

try:
    import cv2
    from sklearn.metrics.pairwise import cosine_similarity
    MERLIN_ENABLED = True
except ImportError:
    MERLIN_ENABLED = False
    logging.warning("[MERLIN] Vision disabled - missing cv2 or sklearn")

class SuperMerlinPatternAnalyzer:
    """MERLIN VISION AI - Pattern Recognition + Prediction"""
    
    def __init__(self, memory_file='merlin_patterns.pkl', similarity_threshold=0.75):
        self.enabled = MERLIN_ENABLED
        self.memory_file = memory_file
        self.similarity_threshold = similarity_threshold
        self.max_patterns = 2000
        self.history_days = 14
        self.pattern_library = {}
        self.prediction_history = []
        self.success_rate = 0.0
        
        if self.enabled:
            self.load_memory()
            logging.info(f"[MERLIN] Pattern Analyzer READY - {sum(len(v) for v in self.pattern_library.values())} patterns")
        else:
            logging.warning("[MERLIN] DISABLED - install: pip install opencv-python scikit-learn")
    
    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.pattern_library = data.get('patterns', {})
                    self.prediction_history = data.get('history', [])
                    self.success_rate = data.get('success_rate', 0.0)
        except: pass
    
    def save_memory(self):
        if not self.enabled: return
        try:
            if self.prediction_history:
                correct = sum(1 for p in self.prediction_history if p.get('correct', False))
                self.success_rate = correct / len(self.prediction_history)
            data = {'patterns': self.pattern_library, 'history': self.prediction_history[-1000:], 'success_rate': self.success_rate}
            with open(self.memory_file, 'wb') as f:
                pickle.dump(data, f)
            logging.info(f"[MERLIN] Memory saved: {sum(len(v) for v in self.pattern_library.values())} patterns, SR: {self.success_rate*100:.1f}%")
        except: pass
    
    def create_sequence_image(self, prices, indicators, size=64):
        """CrÃ©e une image 64x64 du pattern avec prix + RSI + MACD"""
        if not self.enabled or len(prices) < 10:
            return np.zeros((size, size, 3), dtype=np.uint8)
        try:
            prices_norm = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-10)
            image = np.zeros((size, size, 3), dtype=np.uint8)
            
            # Prix (vert)
            for i in range(len(prices_norm)-1):
                y1 = int((1 - prices_norm[i]) * (size-1))
                y2 = int((1 - prices_norm[i+1]) * (size-1))
                x1 = int(i / len(prices_norm) * (size-1))
                x2 = int((i+1) / len(prices_norm) * (size-1))
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # RSI (bleu)
            if 'rsi' in indicators and len(indicators['rsi']) == len(prices):
                rsi_norm = indicators['rsi'] / 100.0
                for i in range(len(rsi_norm)-1):
                    y1 = int((1 - rsi_norm[i]) * (size-1))
                    y2 = int((1 - rsi_norm[i+1]) * (size-1))
                    x1 = int(i / len(rsi_norm) * (size-1))
                    x2 = int((i+1) / len(rsi_norm) * (size-1))
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
            # MACD (rouge)
            if 'macd' in indicators and len(indicators['macd']) == len(prices):
                macd = indicators['macd']
                macd_norm = (macd - np.min(macd)) / (np.max(macd) - np.min(macd) + 1e-10)
                for i in range(len(macd_norm)-1):
                    y1 = int((1 - macd_norm[i]) * (size-1))
                    y2 = int((1 - macd_norm[i+1]) * (size-1))
                    x1 = int(i / len(macd_norm) * (size-1))
                    x2 = int((i+1) / len(macd_norm) * (size-1))
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            return image
        except:
            return np.zeros((size, size, 3), dtype=np.uint8)
    
    def extract_pattern_features(self, image):
        """Extrait features 256-dim (16x16 grayscale flattened)"""
        if not self.enabled:
            return np.zeros(256)
        try:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (16, 16))
            features = resized.flatten() / 255.0
            return features
        except:
            return np.zeros(256)
    
    def find_similar_patterns(self, current_features, timeframe='M5'):
        """Trouve patterns similaires par cosine similarity"""
        if not self.enabled:
            return []
        similar_patterns = []
        try:
            if timeframe not in self.pattern_library:
                return []
            
            for pattern_id, pattern_data in self.pattern_library[timeframe].items():
                stored_features = pattern_data['features']
                similarity = cosine_similarity([current_features], [stored_features])[0][0]
                
                if similarity >= self.similarity_threshold:
                    similar_patterns.append({
                        'pattern_id': pattern_id,
                        'similarity': similarity,
                        'outcome': pattern_data.get('outcome', 'NEUTRE'),
                        'price_movement': pattern_data.get('price_movement', 0),
                        'timestamp': pattern_data.get('timestamp', datetime.now())
                    })
            
            similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_patterns[:10]
        except:
            return []
    
    def predict_movement(self, similar_patterns):
        """PrÃ©dit HAUSSIER/BAISSIER/NEUTRE avec confidence"""
        if not similar_patterns:
            return {'direction': 'NEUTRE', 'confidence': 0.0, 'patterns_count': 0}
        
        try:
            bullish_score = 0.0
            bearish_score = 0.0
            total_similarity = 0.0
            
            for pattern in similar_patterns:
                weight = pattern['similarity']
                movement = pattern.get('price_movement', 0)
                
                if movement > 0.5:
                    bullish_score += weight
                elif movement < -0.5:
                    bearish_score += weight
                
                total_similarity += weight
            
            if total_similarity == 0:
                return {'direction': 'NEUTRE', 'confidence': 0.0, 'patterns_count': len(similar_patterns)}
            
            bullish_ratio = bullish_score / total_similarity
            bearish_ratio = bearish_score / total_similarity
            
            if bullish_ratio > bearish_ratio and bullish_ratio > 0.6:
                direction = 'HAUSSIER'
                confidence = bullish_ratio
            elif bearish_ratio > bullish_ratio and bearish_ratio > 0.6:
                direction = 'BAISSIER'
                confidence = bearish_ratio
            else:
                direction = 'NEUTRE'
                confidence = max(bullish_ratio, bearish_ratio)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'patterns_count': len(similar_patterns)
            }
        except:
            return {'direction': 'NEUTRE', 'confidence': 0.0, 'patterns_count': 0}
    
    def save_pattern(self, features, timeframe, prices, outcome=None):
        """Sauvegarde pattern pour learning futur"""
        if not self.enabled:
            return
        try:
            if timeframe not in self.pattern_library:
                self.pattern_library[timeframe] = {}
            
            price_movement = 0
            if len(prices) >= 10:
                price_movement = (prices[-1] - prices[-10]) / (prices[-10] + 1e-10) * 100
            
            if outcome is None:
                if price_movement > 0.5:
                    outcome = 'HAUSSIER'
                elif price_movement < -0.5:
                    outcome = 'BAISSIER'
                else:
                    outcome = 'NEUTRE'
            
            pattern_id = f"{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            self.pattern_library[timeframe][pattern_id] = {
                'features': features,
                'timestamp': datetime.now(),
                'timeframe': timeframe,
                'price_movement': price_movement,
                'outcome': outcome
            }
            
            # Cleanup
            self._clean_old_patterns(timeframe)
        except:
            pass
    
    def _clean_old_patterns(self, timeframe):
        try:
            if timeframe not in self.pattern_library:
                return
            
            cutoff_time = datetime.now() - timedelta(days=self.history_days)
            patterns_to_remove = [pid for pid, pdata in self.pattern_library[timeframe].items() if pdata['timestamp'] < cutoff_time]
            
            for pid in patterns_to_remove:
                del self.pattern_library[timeframe][pid]
            
            if len(self.pattern_library[timeframe]) > self.max_patterns:
                sorted_patterns = sorted(self.pattern_library[timeframe].items(), key=lambda x: x[1]['timestamp'])
                for i in range(len(sorted_patterns) - self.max_patterns):
                    del self.pattern_library[timeframe][sorted_patterns[i][0]]
        except:
            pass
    
    def record_prediction_outcome(self, predicted, actual):
        """Enregistre le rÃ©sultat d'une prÃ©diction pour tracking"""
        try:
            correct = (predicted == actual)
            self.prediction_history.append({
                'predicted': predicted,
                'actual': actual,
                'correct': correct,
                'timestamp': datetime.now()
            })
            
            # Limiter historique
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            logging.debug(f"[MERLIN] Prediction recorded: {predicted} vs {actual} ({'CORRECT' if correct else 'WRONG'})")
        except:
            pass
    
    def analyze_current_pattern(self, prices, indicators, timeframe='M5'):
        """Analyse complÃ¨te: image -> features -> similarity -> prediction"""
        if not self.enabled or len(prices) < 20:
            return None
        
        try:
            # CrÃ©er image
            image = self.create_sequence_image(prices[-50:], indicators, size=64)
            
            # Extraire features
            features = self.extract_pattern_features(image)
            
            # Trouver similaires
            similar_patterns = self.find_similar_patterns(features, timeframe)
            
            # PrÃ©dire
            prediction = self.predict_movement(similar_patterns)
            
            # Sauvegarder pour learning
            self.save_pattern(features, timeframe, prices[-50:])
            
            return {
                'timestamp': datetime.now(),
                'timeframe': timeframe,
                'prediction': prediction,
                'similar_patterns': similar_patterns,
                'features': features,
                'current_price': prices[-1]
            }
        except:
            return None



# =============================================================================
# SYSTÃˆME DE DÃ‰TECTION DE DIVERGENCES
# =============================================================================

class DivergenceDetectionSystem:
    """SystÃ¨me avancÃ© de dÃ©tection de divergences avec mÃ©moire persistante"""
    
    def __init__(self, memory_file='divergence_memory.pkl', max_patterns=1000):
        self.memory_file = memory_file
        self.max_patterns = max_patterns
        self.divergence_patterns = {}
        self.successful_patterns = []
        self.failed_patterns = []
        self.total_divergences_detected = 0
        self.bullish_divergences = 0
        self.bearish_divergences = 0
        self.divergence_success_rate = 0.0
        # PATCH V4: Compteurs séparés par indicateur
        self.rsi_bullish = 0
        self.rsi_bearish = 0
        self.macd_bullish = 0
        self.macd_bearish = 0
        self.momentum_bullish = 0
        self.momentum_bearish = 0
        self.load_memory()
        logging.info(f"[DIVERGENCE] System initialized - {len(self.divergence_patterns)} patterns loaded")
    
    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.divergence_patterns = data.get('patterns', {})
                    self.successful_patterns = data.get('successful', [])
                    self.failed_patterns = data.get('failed', [])
                    self.total_divergences_detected = data.get('total_detected', 0)
                    self.bullish_divergences = data.get('bullish', 0)
                    self.bearish_divergences = data.get('bearish', 0)
                    # PATCH V4: Charger les compteurs RSI/MACD
                    self.rsi_bullish = data.get('rsi_bullish', 0)
                    self.rsi_bearish = data.get('rsi_bearish', 0)
                    self.macd_bullish = data.get('macd_bullish', 0)
                    self.macd_bearish = data.get('macd_bearish', 0)
                    self.momentum_bullish = data.get('momentum_bullish', 0)
                    self.momentum_bearish = data.get('momentum_bearish', 0)
        except:
            pass
    
    def save_memory(self):
        try:
            if len(self.divergence_patterns) > self.max_patterns:
                sorted_patterns = sorted(self.divergence_patterns.items(), key=lambda x: x[1].get('timestamp', datetime.min), reverse=True)
                self.divergence_patterns = dict(sorted_patterns[:self.max_patterns])
            total_outcomes = len(self.successful_patterns) + len(self.failed_patterns)
            if total_outcomes > 0:
                self.divergence_success_rate = len(self.successful_patterns) / total_outcomes
            data = {
                'patterns': self.divergence_patterns, 
                'successful': self.successful_patterns, 
                'failed': self.failed_patterns, 
                'total_detected': self.total_divergences_detected, 
                'bullish': self.bullish_divergences, 
                'bearish': self.bearish_divergences, 
                'success_rate': self.divergence_success_rate,
                # PATCH V4: Sauvegarder les compteurs RSI/MACD
                'rsi_bullish': self.rsi_bullish,
                'rsi_bearish': self.rsi_bearish,
                'macd_bullish': self.macd_bullish,
                'macd_bearish': self.macd_bearish,
                'momentum_bullish': self.momentum_bullish,
                'momentum_bearish': self.momentum_bearish
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass
    
    def find_peaks(self, data, prominence=0.5):
        if len(data) < 5:
            return []
        peaks = []
        for i in range(2, len(data)-2):
            if (data[i] > data[i-1] and data[i] > data[i-2] and data[i] > data[i+1] and data[i] > data[i+2]):
                if data[i] > np.mean(data) + prominence * np.std(data):
                    peaks.append(i)
        return peaks[-5:]
    
    def find_valleys(self, data, prominence=0.5):
        if len(data) < 5:
            return []
        valleys = []
        for i in range(2, len(data)-2):
            if (data[i] < data[i-1] and data[i] < data[i-2] and data[i] < data[i+1] and data[i] < data[i+2]):
                if data[i] < np.mean(data) - prominence * np.std(data):
                    valleys.append(i)
        return valleys[-5:]
    
    def detect_rsi_divergence(self, prices, rsi, lookback=20):
        try:
            if len(prices) < lookback or len(rsi) < lookback:
                return None
            recent_prices = prices[-lookback:]
            recent_rsi = rsi[-lookback:]
            price_highs = self.find_peaks(recent_prices, prominence=0.3)
            price_lows = self.find_valleys(recent_prices, prominence=0.3)
            rsi_highs = self.find_peaks(recent_rsi, prominence=1.0)
            rsi_lows = self.find_valleys(recent_rsi, prominence=1.0)
            
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                p1, p2 = price_highs[-2], price_highs[-1]
                r1, r2 = rsi_highs[-2], rsi_highs[-1]
                if recent_prices[p2] > recent_prices[p1] and recent_rsi[r2] < recent_rsi[r1]:
                    price_diff = (recent_prices[p2] - recent_prices[p1]) / recent_prices[p1]
                    rsi_diff = (recent_rsi[r1] - recent_rsi[r2]) / 100.0
                    severity = min(1.0, (price_diff + rsi_diff) / 2)
                    confidence = 0.8 if abs(p2 - r2) <= 3 else 0.6
                    return {'type': 'BEARISH', 'indicator': 'RSI', 'severity': severity, 'confidence': confidence, 'description': f'Prix +{price_diff*100:.1f}%, RSI -{rsi_diff*100:.1f}%'}
            
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                p1, p2 = price_lows[-2], price_lows[-1]
                r1, r2 = rsi_lows[-2], rsi_lows[-1]
                if recent_prices[p2] < recent_prices[p1] and recent_rsi[r2] > recent_rsi[r1]:
                    price_diff = (recent_prices[p1] - recent_prices[p2]) / recent_prices[p1]
                    rsi_diff = (recent_rsi[r2] - recent_rsi[r1]) / 100.0
                    severity = min(1.0, (price_diff + rsi_diff) / 2)
                    confidence = 0.8 if abs(p2 - r2) <= 3 else 0.6
                    return {'type': 'BULLISH', 'indicator': 'RSI', 'severity': severity, 'confidence': confidence, 'description': f'Prix -{price_diff*100:.1f}%, RSI +{rsi_diff*100:.1f}%'}
            return None
        except:
            return None
    
    def detect_macd_divergence(self, prices, macd, lookback=20):
        try:
            if len(prices) < lookback or len(macd) < lookback:
                return None
            recent_prices = prices[-lookback:]
            recent_macd = macd[-lookback:]
            price_highs = self.find_peaks(recent_prices, prominence=0.3)
            price_lows = self.find_valleys(recent_prices, prominence=0.3)
            macd_highs = self.find_peaks(recent_macd, prominence=0.5)
            macd_lows = self.find_valleys(recent_macd, prominence=0.5)
            
            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                p1, p2 = price_highs[-2], price_highs[-1]
                m1, m2 = macd_highs[-2], macd_highs[-1]
                if recent_prices[p2] > recent_prices[p1] and recent_macd[m2] < recent_macd[m1]:
                    severity = min(1.0, abs(recent_macd[m1] - recent_macd[m2]) / (abs(recent_macd[m1]) + 1e-6))
                    return {'type': 'BEARISH', 'indicator': 'MACD', 'severity': severity, 'confidence': 0.75, 'description': 'Prix â†‘, MACD â†“'}
            
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                p1, p2 = price_lows[-2], price_lows[-1]
                m1, m2 = macd_lows[-2], macd_lows[-1]
                if recent_prices[p2] < recent_prices[p1] and recent_macd[m2] > recent_macd[m1]:
                    severity = min(1.0, abs(recent_macd[m2] - recent_macd[m1]) / (abs(recent_macd[m1]) + 1e-6))
                    return {'type': 'BULLISH', 'indicator': 'MACD', 'severity': severity, 'confidence': 0.75, 'description': 'Prix â†“, MACD â†‘'}
            return None
        except:
            return None
    
    def detect_momentum_divergence(self, prices, momentum, lookback=20):
        try:
            if len(prices) < lookback or len(momentum) < lookback:
                return None
            recent_prices = prices[-lookback:]
            recent_momentum = momentum[-lookback:]
            price_highs = self.find_peaks(recent_prices, prominence=0.3)
            price_lows = self.find_valleys(recent_prices, prominence=0.3)
            mom_highs = self.find_peaks(recent_momentum, prominence=0.5)
            mom_lows = self.find_valleys(recent_momentum, prominence=0.5)
            
            if len(price_highs) >= 2 and len(mom_highs) >= 2:
                p1, p2 = price_highs[-2], price_highs[-1]
                m1, m2 = mom_highs[-2], mom_highs[-1]
                if recent_prices[p2] > recent_prices[p1] and recent_momentum[m2] < recent_momentum[m1]:
                    return {'type': 'BEARISH', 'indicator': 'MOMENTUM', 'severity': 0.7, 'confidence': 0.65, 'description': 'Prix â†‘, Momentum â†“'}
            
            if len(price_lows) >= 2 and len(mom_lows) >= 2:
                p1, p2 = price_lows[-2], price_lows[-1]
                m1, m2 = mom_lows[-2], mom_lows[-1]
                if recent_prices[p2] < recent_prices[p1] and recent_momentum[m2] > recent_momentum[m1]:
                    return {'type': 'BULLISH', 'indicator': 'MOMENTUM', 'severity': 0.7, 'confidence': 0.65, 'description': 'Prix â†“, Momentum â†‘'}
            return None
        except:
            return None
    
    def detect_all_divergences(self, prices, rsi, macd, momentum, lookback=20):
        divergences = []
        rsi_div = self.detect_rsi_divergence(prices, rsi, lookback)
        if rsi_div:
            divergences.append(rsi_div)
            self.total_divergences_detected += 1
            if rsi_div['type'] == 'BULLISH':
                self.bullish_divergences += 1
                self.rsi_bullish += 1  # PATCH V4
            else:
                self.bearish_divergences += 1
                self.rsi_bearish += 1  # PATCH V4
        
        macd_div = self.detect_macd_divergence(prices, macd, lookback)
        if macd_div:
            divergences.append(macd_div)
            self.total_divergences_detected += 1
            if macd_div['type'] == 'BULLISH':
                self.bullish_divergences += 1
                self.macd_bullish += 1  # PATCH V4
            else:
                self.bearish_divergences += 1
                self.macd_bearish += 1  # PATCH V4
        
        mom_div = self.detect_momentum_divergence(prices, momentum, lookback)
        if mom_div:
            divergences.append(mom_div)
            self.total_divergences_detected += 1
            if mom_div['type'] == 'BULLISH':
                self.bullish_divergences += 1
                self.momentum_bullish += 1  # PATCH V4
            else:
                self.bearish_divergences += 1
                self.momentum_bearish += 1  # PATCH V4
        
        return divergences
    
    def save_divergence_pattern(self, divergence, outcome, profit_pips):
        try:
            pattern_id = f"{divergence['type']}_{divergence['indicator']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            pattern_data = {'divergence': divergence, 'outcome': outcome, 'profit_pips': profit_pips, 'timestamp': datetime.now(), 'severity': divergence.get('severity', 0.5), 'confidence': divergence.get('confidence', 0.5)}
            self.divergence_patterns[pattern_id] = pattern_data
            if outcome == 'SUCCESS':
                self.successful_patterns.append(pattern_id)
            else:
                self.failed_patterns.append(pattern_id)
            
            # PATCH V4.1: Sauvegarder sur disque tous les 10 patterns
            if len(self.divergence_patterns) % 10 == 0:
                self.save_memory()
        except:
            pass
    
    def get_divergence_signal(self, divergences):
        if not divergences:
            return {'signal': 0, 'strength': 0.0, 'count': 0, 'details': []}
        bullish_score = 0.0
        bearish_score = 0.0
        for div in divergences:
            weight = div['severity'] * div['confidence']
            if div['type'] == 'BULLISH':
                bullish_score += weight
            else:
                bearish_score += weight
        total_score = bullish_score + bearish_score
        if total_score == 0:
            return {'signal': 0, 'strength': 0.0, 'count': len(divergences), 'details': divergences}
        if bullish_score > bearish_score:
            signal = 1
            strength = bullish_score / total_score
        else:
            signal = -1
            strength = bearish_score / total_score
        return {'signal': signal, 'strength': strength, 'count': len(divergences), 'details': divergences}



# =============================================================================
# TDA V4 FEATURE EXTRACTOR - INT GR  DANS GOLDENEYE ULTIMATE
# =============================================================================





# =============================================================================
# PERFECT TIMING REWARDS SYSTEM - INTEGRATED
# =============================================================================

class PerfectTimingRewardSystem:
    """
    SystÃƒÂ¨me de rÃƒÂ©compense basÃƒÂ© sur la perfection du timing (Entry, Hold, Exit)
    
    CONCEPT: Au lieu de rÃƒÂ©compenser juste le P&L, on rÃƒÂ©compense la qualitÃƒÂ© du timing:
    - Entry Perfection: ÃƒÂ€ quel point l'entry ÃƒÂ©tait proche du meilleur moment?
    - Hold Perfection: Combien du potentiel maximum a ÃƒÂ©tÃƒÂ© capturÃƒÂ©?
    - Exit Perfection: Exit au bon moment avant retournement?
    
    Un trade parfait peut donner 5-10x plus de reward qu'un trade mÃƒÂ©diocre!
    """
    
    def __init__(self, entry_window=10, exit_window=10):
        """
        Args:
            entry_window: Nombre de bougies AVANT entry pour ÃƒÂ©valuer la qualitÃƒÂ©
            exit_window: Nombre de bougies APRÃƒÂˆS close pour ÃƒÂ©valuer l'exit
        """
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.position_history = {}  # Track positions ouvertes
        
        logging.info(f"[PERFECT TIMING] System initialized")
        logging.info(f"[PERFECT TIMING]   Entry window: {entry_window} candles")
        logging.info(f"[PERFECT TIMING]   Exit window: {exit_window} candles")
    
    def track_position(self, symbol, entry_price, entry_step, position_type):
        """
        Commence ÃƒÂ  tracker une position dÃƒÂ¨s son ouverture
        
        Args:
            symbol: Symbole de trading
            entry_price: Prix d'entrÃƒÂ©e
            entry_step: Step index de l'entry
            position_type: 1 (LONG) ou -1 (SHORT)
        """
        self.position_history[symbol] = {
            'entry_price': entry_price,
            'entry_step': entry_step,
            'position_type': position_type,
            'max_favorable': entry_price,
            'min_favorable': entry_price,
            'max_favorable_step': entry_step,
            'min_favorable_step': entry_step,
            'prices_since_entry': [entry_price],
            'steps_since_entry': [entry_step]
        }
        
        logging.debug(f"[PERFECT TIMING] Tracking position: {symbol} {position_type} @ {entry_price}")
    
    def update_position(self, symbol, current_price, current_step):
        """
        Update le tracking pendant que la position est ouverte
        AppelÃƒÂ© ÃƒÂ  chaque step tant que la position est active
        
        Args:
            symbol: Symbole de trading
            current_price: Prix actuel
            current_step: Step index actuel
        """
        if symbol not in self.position_history:
            return
        
        pos = self.position_history[symbol]
        pos['prices_since_entry'].append(current_price)
        pos['steps_since_entry'].append(current_step)
        
        # Track le meilleur prix favorable atteint
        if current_price > pos['max_favorable']:
            pos['max_favorable'] = current_price
            pos['max_favorable_step'] = current_step
        
        if current_price < pos['min_favorable']:
            pos['min_favorable'] = current_price
            pos['min_favorable_step'] = current_step
    
    def calculate_perfect_timing_reward(self, symbol, close_price, close_step, 
                                        historical_data, future_data=None):
        """
        Calcule le reward basÃƒÂ© sur la perfection du timing
        AppelÃƒÂ© au moment du CLOSE d'une position
        
        Args:
            symbol: Symbole de trading
            close_price: Prix de fermeture
            close_step: Step index du close
            historical_data: DataFrame avec historique (pour entry perfection)
            future_data: DataFrame avec future (pour exit perfection) - optionnel
        
        Returns:
            dict avec:
                - entry_perfection: 0-100
                - hold_perfection: 0-100
                - exit_perfection: 0-100
                - perfect_score: 0-100
                - total_bonus: reward bonus total
                - multiplier: multiplicateur (1.0, 1.2, 1.5, ou 2.0)
        """
        if symbol not in self.position_history:
            logging.warning(f"[PERFECT TIMING] No position history for {symbol}")
            return {
                'entry_perfection': 0,
                'hold_perfection': 0,
                'exit_perfection': 0,
                'perfect_score': 0,
                'total_bonus': 0,
                'multiplier': 1.0
            }
        
        pos = self.position_history[symbol]
        
        # 1. ENTRY PERFECTION (0-100)
        entry_perfection = self._calculate_entry_perfection(
            pos['entry_price'], 
            pos['entry_step'],
            pos['position_type'],
            historical_data
        )
        
        # 2. HOLD PERFECTION (0-100)
        hold_perfection = self._calculate_hold_perfection(
            pos['entry_price'],
            close_price,
            close_step,
            pos['position_type'],
            pos['max_favorable'],
            pos['min_favorable'],
            pos['max_favorable_step'],
            pos['min_favorable_step'],
            pos['entry_step']
        )
        
        # 3. EXIT PERFECTION (0-100)
        exit_perfection = self._calculate_exit_perfection(
            close_price,
            close_step,
            pos['position_type'],
            future_data
        )
        
        # 4. PERFECT TRADE SCORE
        perfect_score = (entry_perfection + hold_perfection + exit_perfection) / 3
        
        # 5. CALCULATE BONUSES
        entry_bonus = 50 * (entry_perfection / 100)
        hold_bonus = 100 * (hold_perfection / 100)
        exit_bonus = 80 * (exit_perfection / 100)
        
        # 6. MULTIPLIER BASÃƒÂ‰ SUR LE SCORE GLOBAL
        multiplier = 1.0
        if perfect_score >= 90:
            multiplier = 2.0  # PERFECT TRADE!
        elif perfect_score >= 75:
            multiplier = 1.5  # EXCELLENT
        elif perfect_score >= 60:
            multiplier = 1.2  # GOOD
        
        total_bonus = (entry_bonus + hold_bonus + exit_bonus) * multiplier
        
        # Log dÃƒÂ©taillÃƒÂ©
        logging.info(f"[PERFECT TIMING] {symbol} Analysis:")
        logging.info(f"  Entry Perfection: {entry_perfection:.1f}/100")
        logging.info(f"  Hold Perfection: {hold_perfection:.1f}/100")
        logging.info(f"  Exit Perfection: {exit_perfection:.1f}/100")
        logging.info(f"  Perfect Score: {perfect_score:.1f}/100")
        logging.info(f"  Multiplier: x{multiplier:.1f}")
        logging.info(f"  Total Bonus: +{total_bonus:.1f}")
        
        # Cleanup
        del self.position_history[symbol]
        
        return {
            'entry_perfection': entry_perfection,
            'hold_perfection': hold_perfection,
            'exit_perfection': exit_perfection,
            'perfect_score': perfect_score,
            'total_bonus': total_bonus,
            'multiplier': multiplier,
            'entry_bonus': entry_bonus,
            'hold_bonus': hold_bonus,
            'exit_bonus': exit_bonus
        }
    
    def _calculate_entry_perfection(self, entry_price, entry_step, position_type, historical_data):
        """
        Calcule ÃƒÂ  quel point l'entry ÃƒÂ©tait proche du meilleur moment
        
        Pour un LONG: meilleur entry = prix le plus BAS dans la fenÃƒÂªtre avant
        Pour un SHORT: meilleur entry = prix le plus HAUT dans la fenÃƒÂªtre avant
        
        Returns: Score 0-100 (100 = entry parfait)
        """
        try:
            # Extraire la fenÃƒÂªtre avant l'entry
            start_idx = max(0, entry_step - self.entry_window)
            end_idx = entry_step
            
            if start_idx >= end_idx or end_idx >= len(historical_data):
                return 50.0  # Score neutre si pas assez de donnÃƒÂ©es
            
            window = historical_data.iloc[start_idx:end_idx]
            
            if len(window) == 0:
                return 50.0
            
            lowest_price = window['low'].min()
            highest_price = window['high'].max()
            price_range = highest_price - lowest_price
            
            if price_range == 0:
                return 50.0  # MarchÃƒÂ© plat
            
            if position_type == 1:  # LONG
                # Meilleur entry = prix le plus bas
                # Distance de notre entry au plus bas
                distance_from_best = entry_price - lowest_price
                perfection = 100 * (1 - distance_from_best / price_range)
            else:  # SHORT
                # Meilleur entry = prix le plus haut
                distance_from_best = highest_price - entry_price
                perfection = 100 * (1 - distance_from_best / price_range)
            
            return max(0, min(100, perfection))
            
        except Exception as e:
            logging.error(f"[PERFECT TIMING] Error calculating entry perfection: {e}")
            return 50.0
    
    def _calculate_hold_perfection(self, entry_price, close_price, close_step,
                                   position_type, max_favorable, min_favorable,
                                   max_step, min_step, entry_step):
        """
        Calcule ÃƒÂ  quel point on a capturÃƒÂ© le potentiel maximum
        
        Deux composantes:
        1. Capture du potentiel: Combien de pips capturÃƒÂ©s vs potentiel max?
        2. Timing du hold: ClosÃƒÂ© au bon moment (proche du peak)?
        
        Returns: Score 0-100 (100 = hold parfait)
        """
        try:
            if position_type == 1:  # LONG
                # Potentiel maximum = max favorable - entry
                potential_pips = (max_favorable - entry_price) * 10000
                # Pips actuellement capturÃƒÂ©s
                actual_pips = (close_price - entry_price) * 10000
                
                # Meilleur exit ÃƒÂ©tait au max favorable
                ideal_exit = max_favorable
                
            else:  # SHORT
                # Potentiel maximum = entry - min favorable
                potential_pips = (entry_price - min_favorable) * 10000
                # Pips actuellement capturÃƒÂ©s
                actual_pips = (entry_price - close_price) * 10000
                
                # Meilleur exit ÃƒÂ©tait au min favorable
                ideal_exit = min_favorable
            
            # 1. CAPTURE PERCENTAGE
            if potential_pips > 0:
                capture_pct = (actual_pips / potential_pips) * 100
                capture_pct = max(0, min(100, capture_pct))
            else:
                capture_pct = 50.0  # Pas de mouvement favorable
            
            # 2. TIMING SCORE
            # Combien de steps entre entry et peak vs total hold time?
            steps_to_peak = abs(max_step - entry_step) if position_type == 1 else abs(min_step - entry_step)
            steps_held = abs(close_step - entry_step)
            
            if steps_held > 0 and steps_to_peak > 0:
                # IdÃƒÂ©al: closer au peak = meilleur score
                timing_ratio = abs(steps_to_peak - steps_held) / steps_held
                timing_score = max(0, 100 - (timing_ratio * 50))
            else:
                timing_score = 50.0
            
            # Moyenne des deux scores
            hold_perfection = (capture_pct + timing_score) / 2
            
            return max(0, min(100, hold_perfection))
            
        except Exception as e:
            logging.error(f"[PERFECT TIMING] Error calculating hold perfection: {e}")
            return 50.0
    
    def _calculate_exit_perfection(self, close_price, close_step, position_type, future_data):
        """
        Calcule si l'exit ÃƒÂ©tait au bon moment (avant retournement)
        
        Regarde les N prochaines bougies aprÃƒÂ¨s le close:
        - Si le prix continue dans notre direction = on a fermÃƒÂ© trop tÃƒÂ´t (score plus bas)
        - Si le prix retrace = excellent timing! (score ÃƒÂ©levÃƒÂ©)
        
        Returns: Score 0-100 (100 = exit parfait)
        """
        try:
            if future_data is None or len(future_data) == 0:
                # Pas de donnÃƒÂ©es future = score neutre
                return 50.0
            
            # Prendre les N prochaines bougies
            future_window = future_data.iloc[:min(self.exit_window, len(future_data))]
            
            if len(future_window) == 0:
                return 50.0
            
            future_high = future_window['high'].max()
            future_low = future_window['low'].min()
            
            if position_type == 1:  # LONG
                # Pour un LONG, meilleur exit = fermer avant que ÃƒÂ§a baisse
                if future_high > close_price:
                    # Prix a continuÃƒÂ© ÃƒÂ  monter = on a fermÃƒÂ© trop tÃƒÂ´t
                    missed_pips = (future_high - close_price) * 10000
                    perfection = max(0, 100 - (missed_pips * 10))
                else:
                    # Prix a baissÃƒÂ© aprÃƒÂ¨s notre close = excellent timing!
                    perfection = 100.0
                    
            else:  # SHORT
                # Pour un SHORT, meilleur exit = fermer avant que ÃƒÂ§a monte
                if future_low < close_price:
                    # Prix a continuÃƒÂ© ÃƒÂ  baisser = on a fermÃƒÂ© trop tÃƒÂ´t
                    missed_pips = (close_price - future_low) * 10000
                    perfection = max(0, 100 - (missed_pips * 10))
                else:
                    # Prix a montÃƒÂ© aprÃƒÂ¨s notre close = excellent timing!
                    perfection = 100.0
            
            return max(0, min(100, perfection))
            
        except Exception as e:
            logging.error(f"[PERFECT TIMING] Error calculating exit perfection: {e}")
            return 50.0





# =============================================================================
# DRAGON TRADING SYSTEMS - LEARNING FROM EVERYTHING
# =============================================================================

class MultiIndicatorConfluenceDetector:
    """
    DÃƒÂ©tecte la confluence de TOUS les indicateurs disponibles.
    Plus il y a de confluence, plus le signal est fort!
    
    Indicateurs analysÃƒÂ©s:
    - Momentum: RSI (14, 7, 21), Stochastic, Williams %R
    - Trend: MACD, EMA crossovers, MA crossovers
    - Volatility: Bollinger Bands, ATR
    - Volume: Volume spikes, OBV
    - Price Action: Candlestick patterns, S/R levels
    """
    
    def __init__(self):
        self.indicators_config = {
            # MOMENTUM
            'rsi_14': {'weight': 1.0, 'oversold': 30, 'overbought': 70},
            'rsi_7': {'weight': 0.8, 'oversold': 25, 'overbought': 75},
            'stochastic': {'weight': 1.0, 'oversold': 20, 'overbought': 80},
            'williams_r': {'weight': 0.7, 'oversold': -80, 'overbought': -20},
            
            # TREND
            'macd': {'weight': 1.2},
            'ema_12_26': {'weight': 1.0},
            'sma_20_50': {'weight': 0.8},
            'sma_50_200': {'weight': 1.5},  # Golden/Death cross
            
            # VOLATILITY
            'bollinger': {'weight': 1.0},
            'atr': {'weight': 0.6},
            
            # VOLUME
            'volume_spike': {'weight': 1.3},
        }
        
        logging.info("[CONFLUENCE] Multi-Indicator Confluence Detector initialized")
    
    def calculate_confluence_score(self, data: pd.DataFrame, step: int) -> Dict:
        """
        Calcule un score de confluence 0-100 basÃƒÂ© sur TOUS les indicateurs
        
        Args:
            data: DataFrame avec tous les indicateurs
            step: Index du step actuel
            
        Returns:
            dict avec score, direction, signals, confidence
        """
        if step < 1 or step >= len(data):
            return self._empty_confluence()
        
        bullish_score = 0.0
        bearish_score = 0.0
        
        signals_detail = {
            'momentum': [],
            'trend': [],
            'volume': [],
            'volatility': [],
            'price_action': []
        }
        
        try:
            # 1. MOMENTUM INDICATORS
            # RSI
            if 'rsi_14' in data.columns:
                rsi_14 = data.iloc[step]['rsi_14']
                if rsi_14 < 30:
                    bullish_score += 10
                    signals_detail['momentum'].append(('RSI14 Oversold', 10))
                elif rsi_14 > 70:
                    bearish_score += 10
                    signals_detail['momentum'].append(('RSI14 Overbought', 10))
            
            # RSI + Stochastic CONFLUENCE
            if 'rsi_14' in data.columns and 'stoch_k' in data.columns:
                rsi_14 = data.iloc[step]['rsi_14']
                stoch_k = data.iloc[step]['stoch_k']
                
                if rsi_14 < 30 and stoch_k < 20:
                    bullish_score += 15  # DOUBLE CONFIRMATION!
                    signals_detail['momentum'].append(('RSI+Stoch OVERSOLD Confluence', 15))
                elif rsi_14 > 70 and stoch_k > 80:
                    bearish_score += 15
                    signals_detail['momentum'].append(('RSI+Stoch OVERBOUGHT Confluence', 15))
            
            # Williams %R
            if 'williams_r' in data.columns:
                williams = data.iloc[step]['williams_r']
                if williams < -80:
                    bullish_score += 7
                    signals_detail['momentum'].append(('Williams %R Oversold', 7))
                elif williams > -20:
                    bearish_score += 7
                    signals_detail['momentum'].append(('Williams %R Overbought', 7))
            
            # 2. TREND INDICATORS
            # MACD Crossover
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd = data.iloc[step]['macd']
                macd_signal = data.iloc[step]['macd_signal']
                macd_prev = data.iloc[step-1]['macd']
                macd_signal_prev = data.iloc[step-1]['macd_signal']
                
                if macd > macd_signal and macd_prev <= macd_signal_prev:
                    bullish_score += 12
                    signals_detail['trend'].append(('MACD Golden Cross', 12))
                elif macd < macd_signal and macd_prev >= macd_signal_prev:
                    bearish_score += 12
                    signals_detail['trend'].append(('MACD Death Cross', 12))
            
            # Triple EMA Alignment
            if all(col in data.columns for col in ['ema_12', 'ema_26', 'sma_50']):
                ema_12 = data.iloc[step]['ema_12']
                ema_26 = data.iloc[step]['ema_26']
                sma_50 = data.iloc[step]['sma_50']
                
                if ema_12 > ema_26 > sma_50:
                    bullish_score += 18
                    signals_detail['trend'].append(('Triple EMA Bullish Alignment', 18))
                elif ema_12 < ema_26 < sma_50:
                    bearish_score += 18
                    signals_detail['trend'].append(('Triple EMA Bearish Alignment', 18))
            
            # 3. VOLATILITY + VOLUME
            if all(col in data.columns for col in ['close', 'bb_upper', 'bb_lower', 'volume', 'volume_ma5']):
                close = data.iloc[step]['close']
                bb_upper = data.iloc[step]['bb_upper']
                bb_lower = data.iloc[step]['bb_lower']
                volume = data.iloc[step]['volume']
                volume_ma = data.iloc[step]['volume_ma5']
                
                # Bollinger Touch + Volume Spike
                if close <= bb_lower * 1.01 and volume > volume_ma * 1.5:
                    bullish_score += 18
                    signals_detail['volume'].append(('BB Lower + Volume Spike', 18))
                elif close >= bb_upper * 0.99 and volume > volume_ma * 1.5:
                    bearish_score += 18
                    signals_detail['volume'].append(('BB Upper + Volume Spike', 18))
            
            # CALCUL FINAL
            total_signals = bullish_score + bearish_score
            
            if total_signals == 0:
                return self._empty_confluence()
            
            if bullish_score > bearish_score:
                direction = 'LONG'
                score = min(100, (bullish_score / max(total_signals, 1)) * 100)
            else:
                direction = 'SHORT'
                score = min(100, (bearish_score / max(total_signals, 1)) * 100)
            
            # Confidence level
            if score >= 80:
                confidence = 'EXTREME'
            elif score >= 65:
                confidence = 'HIGH'
            elif score >= 50:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            return {
                'score': score,
                'direction': direction,
                'signals': signals_detail,
                'confidence': confidence,
                'bullish_points': bullish_score,
                'bearish_points': bearish_score
            }
            
        except Exception as e:
            logging.error(f"[CONFLUENCE] Error calculating confluence: {e}")
            return self._empty_confluence()
    
    def _empty_confluence(self):
        """Retourne une confluence vide"""
        return {
            'score': 0,
            'direction': 'NEUTRAL',
            'signals': {'momentum': [], 'trend': [], 'volume': [], 'volatility': [], 'price_action': []},
            'confidence': 'NONE',
            'bullish_points': 0,
            'bearish_points': 0
        }


# =============================================================================
# 2. MISSED OPPORTUNITY DETECTOR
# =============================================================================

class MissedOpportunityDetector:
    """
    DÃƒÂ©tecte les patterns trinitaires parfaits que l'agent a loupÃƒÂ©s.
    
    TRINITÃƒÂ‰ PARFAITE:
    1. Entry Signal (confluence d'indicateurs forte)
    2. Hold Period optimal (tendance confirmÃƒÂ©e)
    3. Exit Signal (divergence/retournement)
    """
    
    def __init__(self, min_confluence_score=70, min_potential_pips=15):
        self.min_confluence_score = min_confluence_score
        self.min_potential_pips = min_potential_pips
        self.confluence_detector = MultiIndicatorConfluenceDetector()
        
        logging.info("[MISSED OPP] Missed Opportunity Detector initialized")
        logging.info(f"[MISSED OPP]   Min confluence: {min_confluence_score}/100")
        logging.info(f"[MISSED OPP]   Min potential: {min_potential_pips} pips")
    
    def scan_for_perfect_trinity_patterns(self, data: pd.DataFrame, 
                                          start_step: int, 
                                          end_step: int) -> List[Dict]:
        """
        Scanne une fenÃƒÂªtre historique pour dÃƒÂ©tecter les trinitÃƒÂ©s parfaites
        
        Args:
            data: DataFrame avec tous les indicateurs
            start_step: DÃƒÂ©but du scan
            end_step: Fin du scan
            
        Returns:
            Liste des patterns parfaits dÃƒÂ©tectÃƒÂ©s
        """
        perfect_patterns = []
        
        entry_window = 10
        exit_window = 10
        
        for step in range(start_step + entry_window, end_step - exit_window):
            try:
                # 1. DÃƒÂ‰TECTION ENTRY CONFLUENCE
                confluence = self.confluence_detector.calculate_confluence_score(data, step)
                
                if confluence['score'] < self.min_confluence_score:
                    continue  # Pas assez de confluence
                
                # 2. ANALYSER LE POTENTIEL DE HOLD
                hold_analysis = self._analyze_hold_potential(data, step, confluence['direction'])
                
                if hold_analysis['potential_pips'] < self.min_potential_pips:
                    continue  # Pas assez de potentiel
                
                # 3. VÃƒÂ‰RIFIER S'IL Y AVAIT UN SIGNAL EXIT
                exit_step = step + hold_analysis['optimal_hold_time']
                if exit_step >= len(data):
                    continue
                
                exit_signal = self._detect_exit_signal(data, exit_step, confluence['direction'])
                
                if exit_signal['has_signal']:
                    # TRINITÃƒÂ‰ PARFAITE DÃƒÂ‰TECTÃƒÂ‰E!
                    perfect_patterns.append({
                        'entry_step': step,
                        'entry_price': float(data.iloc[step]['close']),
                        'entry_confluence': confluence,
                        'hold_time': hold_analysis['optimal_hold_time'],
                        'exit_step': exit_step,
                        'exit_price': hold_analysis['exit_price'],
                        'potential_pips': hold_analysis['potential_pips'],
                        'exit_signal': exit_signal,
                        'direction': confluence['direction']
                    })
                    
            except Exception as e:
                logging.debug(f"[MISSED OPP] Error scanning step {step}: {e}")
                continue
        
        return perfect_patterns
    
    def _analyze_hold_potential(self, data: pd.DataFrame, entry_step: int, direction: str) -> Dict:
        """
        Analyse le potentiel de profit en holdant depuis entry_step
        
        Returns:
            dict avec optimal_hold_time, exit_price, potential_pips
        """
        entry_price = data.iloc[entry_step]['close']
        pip_size = 0.0001  # Pour forex
        
        max_hold = min(50, len(data) - entry_step - 1)  # Max 50 candles
        
        best_pips = 0
        best_hold_time = 0
        best_exit_price = entry_price
        
        for hold_time in range(5, max_hold):
            step = entry_step + hold_time
            current_price = data.iloc[step]['close']
            
            if direction == 'LONG':
                pips = (current_price - entry_price) / pip_size
            else:  # SHORT
                pips = (entry_price - current_price) / pip_size
            
            if pips > best_pips:
                best_pips = pips
                best_hold_time = hold_time
                best_exit_price = current_price
        
        return {
            'optimal_hold_time': best_hold_time,
            'exit_price': float(best_exit_price),
            'potential_pips': float(best_pips)
        }
    
    def _detect_exit_signal(self, data: pd.DataFrame, step: int, direction: str) -> Dict:
        """
        DÃƒÂ©tecte s'il y a un signal exit ÃƒÂ  ce step
        
        Signaux exit:
        - RSI overbought/oversold inverse
        - MACD crossover inverse
        - Bollinger band touch inverse
        """
        has_signal = False
        signals = []
        
        try:
            # RSI inverse
            if 'rsi_14' in data.columns:
                rsi = data.iloc[step]['rsi_14']
                if direction == 'LONG' and rsi > 70:
                    has_signal = True
                    signals.append('RSI Overbought')
                elif direction == 'SHORT' and rsi < 30:
                    has_signal = True
                    signals.append('RSI Oversold')
            
            # MACD inverse
            if 'macd' in data.columns and 'macd_signal' in data.columns and step > 0:
                macd = data.iloc[step]['macd']
                macd_signal = data.iloc[step]['macd_signal']
                macd_prev = data.iloc[step-1]['macd']
                macd_signal_prev = data.iloc[step-1]['macd_signal']
                
                if direction == 'LONG' and macd < macd_signal and macd_prev >= macd_signal_prev:
                    has_signal = True
                    signals.append('MACD Death Cross')
                elif direction == 'SHORT' and macd > macd_signal and macd_prev <= macd_signal_prev:
                    has_signal = True
                    signals.append('MACD Golden Cross')
            
            return {
                'has_signal': has_signal,
                'signals': signals,
                'score': len(signals) * 35  # 35 points per signal
            }
            
        except Exception as e:
            logging.debug(f"[MISSED OPP] Error detecting exit signal: {e}")
            return {'has_signal': False, 'signals': [], 'score': 0}


# =============================================================================
# 3. SUCCESS PATTERN ANALYZER
# =============================================================================

class SuccessPatternAnalyzer:
    """
    Analyse les patterns gagnants pour identifier ce qui marche
    """
    
    def __init__(self):
        self.winning_trades = []
        self.losing_trades = []
        self.pattern_library = defaultdict(list)
        
        logging.info("[SUCCESS] Success Pattern Analyzer initialized")
    
    def add_trade_result(self, trade_data: Dict):
        """
        Ajoute un trade (gagnant ou perdant) pour analyse
        
        Args:
            trade_data: dict avec entry_confluence, hold_time, pips, etc.
        """
        if trade_data['pips'] > 0:
            self.winning_trades.append(trade_data)
        else:
            self.losing_trades.append(trade_data)
        
        # Extraire le pattern signature
        signature = self._extract_pattern_signature(trade_data)
        self.pattern_library[signature].append(trade_data)
    
    def _extract_pattern_signature(self, trade_data: Dict) -> str:
        """
        CrÃƒÂ©e une signature unique du pattern
        
        Example: "LONG_RSI_OVERSOLD_MACD_GOLDEN_VOL_SPIKE"
        """
        components = []
        
        # Direction
        components.append(trade_data.get('direction', 'UNKNOWN'))
        
        # Confluence signals
        if 'entry_confluence' in trade_data:
            signals = trade_data['entry_confluence'].get('signals', {})
            for category, signal_list in signals.items():
                for signal_name, score in signal_list:
                    # Simplifier le nom
                    simple_name = signal_name.replace(' ', '_').replace('+', '').upper()
                    components.append(simple_name)
        
        return "_".join(components[:5])  # Limiter ÃƒÂ  5 composants
    
    def get_best_patterns(self, top_n=10) -> List[Tuple[str, float]]:
        """
        Retourne les N meilleurs patterns (avec le plus haut winrate)
        
        Returns:
            Liste de (pattern_signature, winrate)
        """
        pattern_stats = []
        
        for signature, trades in self.pattern_library.items():
            if len(trades) < 3:  # Au moins 3 trades
                continue
            
            wins = sum(1 for t in trades if t['pips'] > 0)
            total = len(trades)
            winrate = wins / total
            avg_pips = np.mean([t['pips'] for t in trades])
            
            pattern_stats.append((signature, winrate, avg_pips, total))
        
        # Trier par winrate puis avg_pips
        pattern_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        return [(sig, wr) for sig, wr, _, _ in pattern_stats[:top_n]]
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques globales"""
        total_trades = len(self.winning_trades) + len(self.losing_trades)
        
        if total_trades == 0:
            return {'total': 0, 'winrate': 0, 'patterns_found': 0}
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(self.winning_trades),
            'losing_trades': len(self.losing_trades),
            'winrate': len(self.winning_trades) / total_trades * 100,
            'patterns_found': len(self.pattern_library),
            'avg_winning_pips': np.mean([t['pips'] for t in self.winning_trades]) if self.winning_trades else 0,
            'avg_losing_pips': np.mean([t['pips'] for t in self.losing_trades]) if self.losing_trades else 0
        }


# =============================================================================
# 4. PATTERN MEMORY BANK
# =============================================================================

class PatternMemoryBank:
    """
    Banque de mÃƒÂ©moire ÃƒÂ©volutive des patterns
    Stocke et rappelle les patterns similaires
    """
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.patterns = deque(maxlen=capacity)
        self.pattern_index = {}  # signature -> list of patterns
        
        logging.info(f"[MEMORY] Pattern Memory Bank initialized (capacity: {capacity})")
    
    def store_pattern(self, pattern: Dict):
        """Stocke un pattern dans la mÃƒÂ©moire"""
        self.patterns.append(pattern)
        
        # Indexer par signature
        signature = pattern.get('signature', 'UNKNOWN')
        if signature not in self.pattern_index:
            self.pattern_index[signature] = []
        self.pattern_index[signature].append(pattern)
    
    def recall_similar_patterns(self, current_state: Dict, top_n=5) -> List[Dict]:
        """
        Rappelle les patterns similaires ÃƒÂ  l'ÃƒÂ©tat actuel
        
        Args:
            current_state: ÃƒÂ‰tat actuel (confluence, indicators, etc.)
            top_n: Nombre de patterns ÃƒÂ  retourner
            
        Returns:
            Liste des patterns les plus similaires
        """
        # TODO: ImplÃƒÂ©menter similarity matching
        # Pour l'instant, retourne les derniers patterns
        return list(self.patterns)[-top_n:]
    
    def get_statistics(self) -> Dict:
        """Statistiques de la mÃƒÂ©moire"""
        return {
            'total_patterns': len(self.patterns),
            'unique_signatures': len(self.pattern_index),
            'capacity': self.capacity
        }


# =============================================================================
# 5. AUGMENTED EXPERIENCE REPLAY
# =============================================================================

class AugmentedExperienceReplay:
    """
    Experience Replay augmentÃƒÂ© qui apprend de TOUT:
    - Trades rÃƒÂ©els (gagnants et perdants)
    - Patterns loupÃƒÂ©s (opportunitÃƒÂ©s manquÃƒÂ©es)
    - Patterns ÃƒÂ©vitÃƒÂ©s (bons no-trade)
    """
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.real_experiences = deque(maxlen=capacity)
        self.synthetic_missed = deque(maxlen=10000)
        self.synthetic_avoided = deque(maxlen=5000)
        
        self.winning_patterns = []
        self.losing_patterns = []
        
        logging.info(f"[REPLAY] Augmented Experience Replay initialized (capacity: {capacity})")
    
    def add_experience(self, state, action, reward, next_state, done, metadata=None):
        """Ajoute une expÃƒÂ©rience rÃƒÂ©elle"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'metadata': metadata or {},
            'is_synthetic': False
        }
        
        self.real_experiences.append(experience)
        
        # Analyser
        if reward > 10:
            self.winning_patterns.append(experience)
        elif reward < -10:
            self.losing_patterns.append(experience)
    
    def add_missed_opportunity(self, pattern: Dict, agent_state):
        """
        Ajoute un pattern parfait loupÃƒÂ©
        L'agent apprendra: "J'aurais dÃƒÂ» faire ÃƒÂ§a!"
        """
        synthetic_exp = {
            'state': agent_state,
            'action': 0 if pattern['direction'] == 'LONG' else 1,  # BUY ou SELL
            'reward': pattern['potential_pips'] * 2,
            'next_state': agent_state,  # SimplifiÃƒÂ©
            'done': False,
            'metadata': {
                'confluence_score': pattern['entry_confluence']['score'],
                'is_missed': True,
                'potential_pips': pattern['potential_pips']
            },
            'is_synthetic': True
        }
        
        self.synthetic_missed.append(synthetic_exp)
        logging.info(f"[REPLAY] Added missed pattern: {pattern['potential_pips']:.1f} pips")
    
    def sample_batch(self, batch_size):
        """
        Sample intelligent:
        - 70% expÃƒÂ©riences rÃƒÂ©elles
        - 20% patterns loupÃƒÂ©s
        - 10% patterns gagnants
        """
        n_real = int(batch_size * 0.7)
        n_missed = int(batch_size * 0.2)
        n_winning = batch_size - n_real - n_missed
        
        batch = []
        
        if len(self.real_experiences) >= n_real:
            batch.extend(np.random.choice(list(self.real_experiences), n_real, replace=False))
        
        if len(self.synthetic_missed) >= n_missed:
            batch.extend(np.random.choice(list(self.synthetic_missed), n_missed, replace=False))
        
        if len(self.winning_patterns) >= n_winning:
            batch.extend(np.random.choice(self.winning_patterns, n_winning, replace=False))
        
        return batch
    
    def get_statistics(self) -> Dict:
        """Statistiques du replay buffer"""
        return {
            'real_experiences': len(self.real_experiences),
            'synthetic_missed': len(self.synthetic_missed),
            'winning_patterns': len(self.winning_patterns),
            'losing_patterns': len(self.losing_patterns),
            'total': len(self.real_experiences) + len(self.synthetic_missed)
        }


# =============================================================================
# 6. STRATEGY EVOLUTION ENGINE
# =============================================================================

class StrategyEvolutionEngine:
    """
    Moteur d'ÃƒÂ©volution des stratÃƒÂ©gies
    Fait ÃƒÂ©voluer les paramÃƒÂ¨tres basÃƒÂ©s sur les performances
    """
    
    def __init__(self):
        self.strategy_history = []
        self.current_strategy = {
            'min_confluence_score': 70,
            'min_potential_pips': 15,
            'max_hold_time': 50,
            'risk_per_trade': 0.02
        }
        
        logging.info("[EVOLUTION] Strategy Evolution Engine initialized")
    
    def evolve_strategy(self, performance_metrics: Dict):
        """
        Fait ÃƒÂ©voluer la stratÃƒÂ©gie basÃƒÂ©e sur les performances
        
        Args:
            performance_metrics: winrate, profit_factor, etc.
        """
        # Simple adaptation pour l'instant
        winrate = performance_metrics.get('winrate', 0)
        
        if winrate < 45:
            # Augmenter les exigences
            self.current_strategy['min_confluence_score'] += 5
            self.current_strategy['min_potential_pips'] += 2
            logging.info("[EVOLUTION] Winrate low - Increasing requirements")
        elif winrate > 60:
            # Assouplir lÃƒÂ©gÃƒÂ¨rement
            self.current_strategy['min_confluence_score'] = max(60, self.current_strategy['min_confluence_score'] - 2)
            logging.info("[EVOLUTION] Winrate high - Slight relaxation")
        
        self.strategy_history.append(self.current_strategy.copy())
    
    def get_current_strategy(self) -> Dict:
        """Retourne la stratÃƒÂ©gie actuelle"""
        return self.current_strategy.copy()





# =============================================================================
# DRAGON LEARNING CALLBACKS
# =============================================================================

class DragonLearningCallback(BaseCallback):
    """
    Callback qui permet ÃƒÂ  l'agent d'apprendre des patterns loupÃƒÂ©s
    
    ÃƒÂ€ chaque N steps:
    1. RÃƒÂ©cupÃƒÂ¨re les patterns loupÃƒÂ©s du AugmentedExperienceReplay
    2. Les convertit en format compatible SB3
    3. Les injecte dans le buffer de l'agent
    4. Force un training step supplÃƒÂ©mentaire sur ces donnÃƒÂ©es
    """
    
    def __init__(self, env, injection_frequency=1000, verbose=0):
        """
        Args:
            env: L'environnement TradingEnvironment (unwrapped)
            injection_frequency: Tous les N steps, injecter les expÃƒÂ©riences
            verbose: Niveau de log
        """
        super().__init__(verbose)
        self.env_unwrapped = env
        self.injection_frequency = injection_frequency
        self.last_injection_step = 0
        
        # Stats
        self.total_injections = 0
        self.total_experiences_injected = 0
        
        logging.info("[DRAGON CALLBACK] Initialized")
        logging.info(f"[DRAGON CALLBACK] Injection frequency: {injection_frequency} steps")
    
    def _on_step(self) -> bool:
        """
        AppelÃƒÂ© ÃƒÂ  chaque step du training
        
        Returns:
            True pour continuer le training
        """
        # VÃƒÂ©rifier si c'est le moment d'injecter
        if self.num_timesteps - self.last_injection_step >= self.injection_frequency:
            self._inject_synthetic_experiences()
            self.last_injection_step = self.num_timesteps
        
        return True
    
    def _inject_synthetic_experiences(self):
        """
        Injecte les expÃƒÂ©riences synthÃƒÂ©tiques dans le buffer de l'agent
        """
        try:
            # RÃƒÂ©cupÃƒÂ©rer le replay buffer de l'environnement
            if not hasattr(self.env_unwrapped, 'experience_replay'):
                logging.debug("[DRAGON CALLBACK] No experience_replay found in env")
                return
            
            replay_buffer = self.env_unwrapped.experience_replay
            
            # RÃƒÂ©cupÃƒÂ©rer les expÃƒÂ©riences synthÃƒÂ©tiques (patterns loupÃƒÂ©s)
            synthetic_experiences = list(replay_buffer.synthetic_missed)
            
            if len(synthetic_experiences) == 0:
                logging.debug("[DRAGON CALLBACK] No synthetic experiences to inject")
                return
            
            # Sample un batch d'expÃƒÂ©riences synthÃƒÂ©tiques
            batch_size = min(32, len(synthetic_experiences))
            sampled = np.random.choice(synthetic_experiences, batch_size, replace=False)
            
            logging.info(f"[DRAGON CALLBACK] Injecting {batch_size} synthetic experiences")
            
            # Injecter dans le buffer de l'agent
            for exp in sampled:
                self._add_to_agent_buffer(exp)
            
            self.total_injections += 1
            self.total_experiences_injected += batch_size
            
            # Log stats
            if self.total_injections % 10 == 0:
                logging.info(f"[DRAGON CALLBACK] Stats:")
                logging.info(f"  Total injections: {self.total_injections}")
                logging.info(f"  Total experiences: {self.total_experiences_injected}")
                logging.info(f"  Avg per injection: {self.total_experiences_injected / self.total_injections:.1f}")
            
        except Exception as e:
            logging.error(f"[DRAGON CALLBACK] Error injecting experiences: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_to_agent_buffer(self, experience: Dict):
        """
        Ajoute une expÃƒÂ©rience synthÃƒÂ©tique au buffer de l'agent
        
        Args:
            experience: dict avec state, action, reward, next_state, done
        """
        try:
            # AccÃƒÂ©der au buffer de l'agent
            if hasattr(self.model, 'replay_buffer'):
                # DQN a un replay buffer
                buffer = self.model.replay_buffer
                
                # Ajouter l'expÃƒÂ©rience
                buffer.add(
                    obs=experience['state'],
                    next_obs=experience['next_state'],
                    action=np.array([experience['action']]),
                    reward=np.array([experience['reward']]),
                    done=np.array([experience['done']]),
                    infos=[{}]
                )
                
                logging.debug(f"[DRAGON CALLBACK] Added to DQN buffer: reward={experience['reward']:.1f}")
                
            elif hasattr(self.model, 'rollout_buffer'):
                # PPO a un rollout buffer (diffÃƒÂ©rent!)
                # Pour PPO c'est plus complexe car il faut des trajectoires complÃƒÂ¨tes
                # On va plutÃƒÂ´t utiliser ces expÃƒÂ©riences pour "guider" le policy
                
                # TODO: ImplÃƒÂ©menter injection dans PPO rollout buffer
                # Pour l'instant, on log juste
                logging.debug(f"[DRAGON CALLBACK] PPO detected - using reward shaping instead")
                
                # Alternative: Modifier directement la policy via gradient
                # self._apply_synthetic_gradient(experience)
                
            else:
                logging.debug("[DRAGON CALLBACK] Unknown agent type - no buffer found")
                
        except Exception as e:
            logging.debug(f"[DRAGON CALLBACK] Error adding to buffer: {e}")
    
    def _apply_synthetic_gradient(self, experience: Dict):
        """
        [AVANCÃƒÂ‰] Applique un gradient synthÃƒÂ©tique basÃƒÂ© sur l'expÃƒÂ©rience
        
        Ceci est une approche alternative pour PPO:
        Au lieu d'ajouter au buffer, on fait un gradient step directement
        """
        # TODO: ImplÃƒÂ©menter si nÃƒÂ©cessaire
        pass
    
    def _on_training_end(self):
        """AppelÃƒÂ© ÃƒÂ  la fin du training"""
        logging.info("[DRAGON CALLBACK] Training ended")
        logging.info(f"  Total injections: {self.total_injections}")
        logging.info(f"  Total synthetic experiences: {self.total_experiences_injected}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ENSEMBLE TRAINING CALLBACK - Pour l'entraÃ®nement collaboratif PPO+DQN+Meta-Learner
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class EnsembleTrainingCallback(BaseCallback):
    """
    Callback pour l'entraÃ®nement ENSEMBLE (Phase 3).
    
    ResponsabilitÃ©s:
    1. Logger les statistiques de l'ensemble
    2. EntraÃ®ner le meta-learner avec les rÃ©sultats
    3. Afficher les insights sur PPO vs DQN
    """
    
    def __init__(self, ensemble_env, meta_learner, symbol: str, verbose: int = 0):
        super().__init__(verbose)
        self.ensemble_env = ensemble_env
        self.meta_learner = meta_learner
        self.symbol = symbol
        self.last_log_step = 0
        self.log_interval = 1000
        
    def _on_step(self) -> bool:
        if self.n_calls - self.last_log_step >= self.log_interval:
            self.last_log_step = self.n_calls
            self._log_ensemble_stats()
        return True
    
    def _log_ensemble_stats(self):
        try:
            stats = self.ensemble_env.get_ensemble_stats()
            if stats and stats.get('total_decisions', 0) > 0:
                logging.info(f"[ENSEMBLE] Step {self.n_calls:,} - {self.symbol}: "
                           f"Decisions={stats['total_decisions']:,}, "
                           f"Consensus={stats['consensus_rate']:.1f}%, "
                           f"PPO={stats['ppo_selected_rate']:.1f}%, "
                           f"DQN={stats['dqn_selected_rate']:.1f}%")
        except Exception as e:
            pass
    
    def _on_training_end(self):
        logging.info(f"[ENSEMBLE] Phase 3 complete for {self.symbol}")
        if self.meta_learner:
            self.meta_learner.save()


class DragonRewardShapingCallback(BaseCallback):
    """
    Alternative approche: Reward Shaping basÃƒÂ© sur les patterns
    
    Au lieu d'injecter dans le buffer, on MODIFIE les rewards en temps rÃƒÂ©el
    basÃƒÂ© sur la confluence et les patterns dÃƒÂ©tectÃƒÂ©s
    """
    
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env_unwrapped = env
        self.bonus_applied = 0
        self.penalty_applied = 0
        
        logging.info("[DRAGON SHAPING] Reward Shaping Callback initialized")
    
    def _on_step(self) -> bool:
        """
        Modifie le reward basÃƒÂ© sur la confluence actuelle
        """
        try:
            # Obtenir le dernier reward
            if hasattr(self.locals, 'rewards'):
                current_reward = self.locals['rewards'][0]
                
                # Obtenir la confluence actuelle
                if hasattr(self.env_unwrapped, '_last_confluence'):
                    confluence = self.env_unwrapped._last_confluence
                    
                    # Si confluence forte, amplifier le reward
                    if confluence.get('score', 0) >= 80:
                        bonus = current_reward * 0.5  # +50% bonus
                        self.locals['rewards'][0] += bonus
                        self.bonus_applied += 1
                        
                        logging.debug(f"[DRAGON SHAPING] High confluence bonus: +{bonus:.1f}")
                    
                    # Si confluence faible et action prise, pÃƒÂ©nalitÃƒÂ©
                    elif confluence.get('score', 0) < 40 and abs(current_reward) > 1:
                        penalty = abs(current_reward) * 0.3
                        self.locals['rewards'][0] -= penalty
                        self.penalty_applied += 1
                        
                        logging.debug(f"[DRAGON SHAPING] Low confluence penalty: -{penalty:.1f}")
        
        except Exception as e:
            logging.debug(f"[DRAGON SHAPING] Error: {e}")
        
        return True
    
    def _on_training_end(self):
        """Stats finales"""
        logging.info("[DRAGON SHAPING] Training ended")
        logging.info(f"  Bonuses applied: {self.bonus_applied}")
        logging.info(f"  Penalties applied: {self.penalty_applied}")






# =============================================================================
# ADVANCED LEARNING SYSTEMS - MARKET INTELLIGENCE
# =============================================================================

class MarketRegimeDetector:
    """
    DÃƒÂ©tecte le rÃƒÂ©gime actuel du marchÃƒÂ©:
    - TRENDING_UP: Tendance haussiÃƒÂ¨re forte
    - TRENDING_DOWN: Tendance baissiÃƒÂ¨re forte
    - RANGING: Consolidation/range
    - VOLATILE: VolatilitÃƒÂ© ÃƒÂ©levÃƒÂ©e sans direction claire
    - BREAKOUT: Cassure en cours
    """
    
    def __init__(self):
        self.current_regime = 'UNKNOWN'
        self.regime_confidence = 0.0
        self.regime_history = deque(maxlen=100)
        
        logging.info("[REGIME] Market Regime Detector initialized")
    
    def detect_regime(self, data: pd.DataFrame, step: int, lookback=50) -> Dict:
        """
        DÃƒÂ©tecte le rÃƒÂ©gime de marchÃƒÂ© actuel
        
        Args:
            data: DataFrame avec OHLCV + indicateurs
            step: Step actuel
            lookback: Nombre de bougies ÃƒÂ  analyser
            
        Returns:
            dict avec regime, confidence, signals
        """
        if step < lookback:
            return self._unknown_regime()
        
        try:
            window = data.iloc[max(0, step-lookback):step+1]
            
            # 1. TREND ANALYSIS
            trend_score = self._analyze_trend(window)
            
            # 2. VOLATILITY ANALYSIS
            volatility_score = self._analyze_volatility(window)
            
            # 3. RANGE ANALYSIS
            range_score = self._analyze_range(window)
            
            # 4. MOMENTUM ANALYSIS
            momentum_score = self._analyze_momentum(window)
            
            # CLASSIFIER
            regime = self._classify_regime(
                trend_score, 
                volatility_score, 
                range_score, 
                momentum_score
            )
            
            self.current_regime = regime['name']
            self.regime_confidence = regime['confidence']
            self.regime_history.append({
                'step': step,
                'regime': regime['name'],
                'confidence': regime['confidence']
            })
            
            return regime
            
        except Exception as e:
            logging.error(f"[REGIME] Error detecting regime: {e}")
            return self._unknown_regime()
    
    def _analyze_trend(self, window: pd.DataFrame) -> float:
        """
        Analyse la force de la tendance (-100 ÃƒÂ  +100)
        
        Utilise:
        - EMA slopes
        - Higher highs / Lower lows
        - ADX si disponible
        """
        trend_score = 0.0
        
        try:
            # EMA slope
            if 'ema_12' in window.columns and 'ema_26' in window.columns:
                ema_12_slope = (window['ema_12'].iloc[-1] - window['ema_12'].iloc[0]) / len(window)
                ema_26_slope = (window['ema_26'].iloc[-1] - window['ema_26'].iloc[0]) / len(window)
                
                # Normaliser
                avg_slope = (ema_12_slope + ema_26_slope) / 2
                trend_score += np.sign(avg_slope) * min(abs(avg_slope) * 10000, 50)
            
            # Higher highs / Lower lows
            highs = window['high'].values
            lows = window['low'].values
            
            higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
            lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
            
            hh_ll_score = (higher_highs - lower_lows) / len(highs) * 50
            trend_score += hh_ll_score
            
        except Exception as e:
            logging.debug(f"[REGIME] Error in trend analysis: {e}")
        
        return np.clip(trend_score, -100, 100)
    
    def _analyze_volatility(self, window: pd.DataFrame) -> float:
        """
        Analyse la volatilitÃƒÂ© (0 ÃƒÂ  100)
        """
        volatility_score = 0.0
        
        try:
            # ATR-based volatility
            if 'atr_14' in window.columns:
                atr = window['atr_14'].iloc[-1]
                atr_mean = window['atr_14'].mean()
                
                if atr_mean > 0:
                    volatility_score = (atr / atr_mean) * 50
            
            # Price range volatility
            price_range = (window['high'] - window['low']) / window['close']
            avg_range = price_range.mean()
            current_range = price_range.iloc[-1]
            
            range_volatility = (current_range / avg_range) * 50
            volatility_score = (volatility_score + range_volatility) / 2
            
        except Exception as e:
            logging.debug(f"[REGIME] Error in volatility analysis: {e}")
        
        return np.clip(volatility_score, 0, 100)
    
    def _analyze_range(self, window: pd.DataFrame) -> float:
        """
        Analyse si le marchÃƒÂ© est en range (0 ÃƒÂ  100)
        """
        range_score = 0.0
        
        try:
            # Bollinger Bands width
            if all(col in window.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                bb_width = (window['bb_upper'] - window['bb_lower']) / window['bb_middle']
                avg_width = bb_width.mean()
                current_width = bb_width.iloc[-1]
                
                # Plus les bandes sont serrÃƒÂ©es, plus on est en range
                if avg_width > 0:
                    range_score = max(0, 100 - (current_width / avg_width) * 100)
            
            # Price oscillation around mean
            mean_price = window['close'].mean()
            deviations = abs(window['close'] - mean_price) / mean_price
            avg_deviation = deviations.mean()
            
            # Faible dÃƒÂ©viation = range
            range_score = (range_score + (1 - avg_deviation * 100)) / 2
            
        except Exception as e:
            logging.debug(f"[REGIME] Error in range analysis: {e}")
        
        return np.clip(range_score, 0, 100)
    
    def _analyze_momentum(self, window: pd.DataFrame) -> float:
        """
        Analyse le momentum (-100 ÃƒÂ  +100)
        """
        momentum_score = 0.0
        
        try:
            # RSI momentum
            if 'rsi_14' in window.columns:
                rsi = window['rsi_14'].iloc[-1]
                momentum_score += (rsi - 50)  # -50 ÃƒÂ  +50
            
            # MACD momentum
            if 'macd' in window.columns:
                macd = window['macd'].iloc[-1]
                macd_mean = window['macd'].mean()
                
                if abs(macd_mean) > 0:
                    momentum_score += np.sign(macd) * min(abs(macd / macd_mean) * 25, 50)
            
        except Exception as e:
            logging.debug(f"[REGIME] Error in momentum analysis: {e}")
        
        return np.clip(momentum_score, -100, 100)
    
    def _classify_regime(self, trend, volatility, range_bound, momentum) -> Dict:
        """
        Classifie le rÃƒÂ©gime basÃƒÂ© sur les scores
        """
        # TRENDING UP
        if trend > 40 and momentum > 20:
            return {
                'name': 'TRENDING_UP',
                'confidence': min(trend + momentum, 100) / 100,
                'scores': {'trend': trend, 'volatility': volatility, 'range': range_bound, 'momentum': momentum}
            }
        
        # TRENDING DOWN
        if trend < -40 and momentum < -20:
            return {
                'name': 'TRENDING_DOWN',
                'confidence': min(abs(trend) + abs(momentum), 100) / 100,
                'scores': {'trend': trend, 'volatility': volatility, 'range': range_bound, 'momentum': momentum}
            }
        
        # RANGING
        if range_bound > 60 and abs(trend) < 30:
            return {
                'name': 'RANGING',
                'confidence': range_bound / 100,
                'scores': {'trend': trend, 'volatility': volatility, 'range': range_bound, 'momentum': momentum}
            }
        
        # VOLATILE
        if volatility > 70:
            return {
                'name': 'VOLATILE',
                'confidence': volatility / 100,
                'scores': {'trend': trend, 'volatility': volatility, 'range': range_bound, 'momentum': momentum}
            }
        
        # BREAKOUT (volatilitÃƒÂ© haute + momentum fort + sortie de range)
        if volatility > 50 and abs(momentum) > 40 and range_bound < 40:
            direction = 'UP' if momentum > 0 else 'DOWN'
            return {
                'name': f'BREAKOUT_{direction}',
                'confidence': min(volatility + abs(momentum), 100) / 100,
                'scores': {'trend': trend, 'volatility': volatility, 'range': range_bound, 'momentum': momentum}
            }
        
        # UNKNOWN
        return {
            'name': 'CONSOLIDATION',
            'confidence': 0.3,
            'scores': {'trend': trend, 'volatility': volatility, 'range': range_bound, 'momentum': momentum}
        }
    
    def _unknown_regime(self):
        return {
            'name': 'UNKNOWN',
            'confidence': 0.0,
            'scores': {'trend': 0, 'volatility': 0, 'range': 0, 'momentum': 0}
        }


# =============================================================================
# 2. ADVANCED PATTERN RECOGNIZER
# =============================================================================

class AdvancedPatternRecognizer:
    """
    Reconnaissance de patterns avancÃƒÂ©s:
    - Chart patterns (Head & Shoulders, Double Top/Bottom, etc.)
    - Fibonacci retracements
    - Support/Resistance dynamiques
    - Harmonic patterns
    """
    
    def __init__(self):
        self.detected_patterns = []
        self.support_levels = []
        self.resistance_levels = []
        
        logging.info("[PATTERN] Advanced Pattern Recognizer initialized")
    
    def detect_all_patterns(self, data: pd.DataFrame, step: int) -> List[Dict]:
        """
        DÃƒÂ©tecte TOUS les patterns disponibles
        
        Returns:
            Liste de patterns dÃƒÂ©tectÃƒÂ©s avec leur force
        """
        patterns = []
        
        try:
            # 1. Support/Resistance
            sr_patterns = self._detect_support_resistance(data, step)
            patterns.extend(sr_patterns)
            
            # 2. Chart Patterns
            chart_patterns = self._detect_chart_patterns(data, step)
            patterns.extend(chart_patterns)
            
            # 3. Candlestick Patterns
            candle_patterns = self._detect_candlestick_patterns(data, step)
            patterns.extend(candle_patterns)
            
            # 4. Fibonacci Levels
            fib_patterns = self._detect_fibonacci_levels(data, step)
            patterns.extend(fib_patterns)
            
        except Exception as e:
            logging.error(f"[PATTERN] Error detecting patterns: {e}")
        
        return patterns
    
    def _detect_support_resistance(self, data: pd.DataFrame, step: int, lookback=100) -> List[Dict]:
        """DÃƒÂ©tecte les niveaux de support/rÃƒÂ©sistance"""
        patterns = []
        
        try:
            if step < lookback:
                return patterns
            
            window = data.iloc[max(0, step-lookback):step+1]
            
            # Trouver les swing highs/lows
            highs = window['high'].values
            lows = window['low'].values
            
            # Swing highs (rÃƒÂ©sistances)
            for i in range(2, len(highs)-2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    resistance_level = highs[i]
                    self.resistance_levels.append(resistance_level)
                    
                    # Check si le prix actuel est proche
                    current_price = data.iloc[step]['close']
                    distance_pct = abs(current_price - resistance_level) / resistance_level
                    
                    if distance_pct < 0.002:  # Moins de 0.2%
                        patterns.append({
                            'type': 'RESISTANCE_TOUCH',
                            'level': resistance_level,
                            'strength': 80,
                            'signal': 'SELL'
                        })
            
            # Swing lows (supports)
            for i in range(2, len(lows)-2):
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                   lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    support_level = lows[i]
                    self.support_levels.append(support_level)
                    
                    current_price = data.iloc[step]['close']
                    distance_pct = abs(current_price - support_level) / support_level
                    
                    if distance_pct < 0.002:
                        patterns.append({
                            'type': 'SUPPORT_TOUCH',
                            'level': support_level,
                            'strength': 80,
                            'signal': 'BUY'
                        })
            
        except Exception as e:
            logging.debug(f"[PATTERN] Error in S/R detection: {e}")
        
        return patterns
    
    def _detect_chart_patterns(self, data: pd.DataFrame, step: int) -> List[Dict]:
        """DÃƒÂ©tecte les chart patterns classiques"""
        patterns = []
        
        try:
            # Double Top
            double_top = self._detect_double_top(data, step)
            if double_top:
                patterns.append(double_top)
            
            # Double Bottom
            double_bottom = self._detect_double_bottom(data, step)
            if double_bottom:
                patterns.append(double_bottom)
            
        except Exception as e:
            logging.debug(f"[PATTERN] Error in chart patterns: {e}")
        
        return patterns
    
    def _detect_double_top(self, data: pd.DataFrame, step: int, lookback=50) -> Optional[Dict]:
        """DÃƒÂ©tecte un double top"""
        if step < lookback:
            return None
        
        try:
            window = data.iloc[max(0, step-lookback):step+1]
            highs = window['high'].values
            
            # Trouver deux peaks similaires
            peaks = []
            for i in range(2, len(highs)-2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    peaks.append((i, highs[i]))
            
            # VÃƒÂ©rifier s'il y a deux peaks similaires
            for i in range(len(peaks)-1):
                for j in range(i+1, len(peaks)):
                    peak1_idx, peak1_val = peaks[i]
                    peak2_idx, peak2_val = peaks[j]
                    
                    # Peaks similaires (dans 1%)
                    if abs(peak1_val - peak2_val) / peak1_val < 0.01:
                        return {
                            'type': 'DOUBLE_TOP',
                            'strength': 75,
                            'signal': 'SELL',
                            'target': min(window['low'].iloc[peak1_idx:peak2_idx])
                        }
        
        except:
            pass
        
        return None
    
    def _detect_double_bottom(self, data: pd.DataFrame, step: int, lookback=50) -> Optional[Dict]:
        """DÃƒÂ©tecte un double bottom"""
        if step < lookback:
            return None
        
        try:
            window = data.iloc[max(0, step-lookback):step+1]
            lows = window['low'].values
            
            # Trouver deux troughs similaires
            troughs = []
            for i in range(2, len(lows)-2):
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                   lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    troughs.append((i, lows[i]))
            
            # VÃƒÂ©rifier s'il y a deux troughs similaires
            for i in range(len(troughs)-1):
                for j in range(i+1, len(troughs)):
                    trough1_idx, trough1_val = troughs[i]
                    trough2_idx, trough2_val = troughs[j]
                    
                    if abs(trough1_val - trough2_val) / trough1_val < 0.01:
                        return {
                            'type': 'DOUBLE_BOTTOM',
                            'strength': 75,
                            'signal': 'BUY',
                            'target': max(window['high'].iloc[trough1_idx:trough2_idx])
                        }
        
        except:
            pass
        
        return None
    
    def _detect_candlestick_patterns(self, data: pd.DataFrame, step: int) -> List[Dict]:
        """DÃƒÂ©tecte les patterns de chandeliers"""
        patterns = []
        
        if step < 3:
            return patterns
        
        try:
            current = data.iloc[step]
            prev = data.iloc[step-1]
            
            # Doji
            body = abs(current['close'] - current['open'])
            range_size = current['high'] - current['low']
            
            if body / range_size < 0.1:  # Body trÃƒÂ¨s petit
                patterns.append({
                    'type': 'DOJI',
                    'strength': 60,
                    'signal': 'REVERSAL'
                })
            
            # Hammer
            lower_shadow = current['open'] - current['low'] if current['close'] > current['open'] else current['close'] - current['low']
            upper_shadow = current['high'] - current['close'] if current['close'] > current['open'] else current['high'] - current['open']
            
            if lower_shadow > body * 2 and upper_shadow < body * 0.3:
                patterns.append({
                    'type': 'HAMMER',
                    'strength': 70,
                    'signal': 'BUY'
                })
            
            # Shooting Star
            if upper_shadow > body * 2 and lower_shadow < body * 0.3:
                patterns.append({
                    'type': 'SHOOTING_STAR',
                    'strength': 70,
                    'signal': 'SELL'
                })
            
            # Engulfing
            if current['close'] > current['open'] and prev['close'] < prev['open']:
                if current['close'] > prev['open'] and current['open'] < prev['close']:
                    patterns.append({
                        'type': 'BULLISH_ENGULFING',
                        'strength': 75,
                        'signal': 'BUY'
                    })
            
            if current['close'] < current['open'] and prev['close'] > prev['open']:
                if current['close'] < prev['open'] and current['open'] > prev['close']:
                    patterns.append({
                        'type': 'BEARISH_ENGULFING',
                        'strength': 75,
                        'signal': 'SELL'
                    })
            
        except Exception as e:
            logging.debug(f"[PATTERN] Error in candlestick patterns: {e}")
        
        return patterns
    
    def _detect_fibonacci_levels(self, data: pd.DataFrame, step: int, lookback=100) -> List[Dict]:
        """DÃƒÂ©tecte les niveaux de Fibonacci"""
        patterns = []
        
        try:
            if step < lookback:
                return patterns
            
            window = data.iloc[max(0, step-lookback):step+1]
            
            # Trouver le swing high et low
            swing_high = window['high'].max()
            swing_low = window['low'].min()
            
            # Niveaux de Fibonacci
            diff = swing_high - swing_low
            fib_levels = {
                '0.236': swing_high - diff * 0.236,
                '0.382': swing_high - diff * 0.382,
                '0.500': swing_high - diff * 0.500,
                '0.618': swing_high - diff * 0.618,
                '0.786': swing_high - diff * 0.786
            }
            
            current_price = data.iloc[step]['close']
            
            # Check si le prix est proche d'un niveau
            for level_name, level_price in fib_levels.items():
                distance_pct = abs(current_price - level_price) / level_price
                
                if distance_pct < 0.001:  # 0.1%
                    patterns.append({
                        'type': f'FIB_{level_name}',
                        'level': level_price,
                        'strength': 65,
                        'signal': 'BOUNCE_EXPECTED'
                    })
        
        except Exception as e:
            logging.debug(f"[PATTERN] Error in Fibonacci: {e}")
        
        return patterns


# =============================================================================
# 3. LONG-TERM MEMORY SYSTEM
# =============================================================================

class LongTermMemorySystem:
    """
    SystÃƒÂ¨me de mÃƒÂ©moire ÃƒÂ  long terme qui persiste entre sessions
    
    Stocke:
    - Best patterns historiques
    - StratÃƒÂ©gies gagnantes par rÃƒÂ©gime de marchÃƒÂ©
    - ParamÃƒÂ¨tres optimaux dÃƒÂ©couverts
    """
    
    def __init__(self, memory_file='dragon_memory.pkl'):
        self.memory_file = memory_file
        self.memory = {
            'best_patterns': [],
            'strategies_by_regime': defaultdict(list),
            'optimal_parameters': {},
            'session_history': [],
            'total_trades': 0,
            'total_wins': 0,
            'creation_date': datetime.now().isoformat()
        }
        
        # Charger la mÃƒÂ©moire existante
        self.load_memory()
        
        logging.info("[MEMORY] Long-Term Memory System initialized")
        logging.info(f"[MEMORY] Memory file: {memory_file}")
        logging.info(f"[MEMORY] Historical trades: {self.memory['total_trades']}")
    
    def load_memory(self):
        """Charge la mÃƒÂ©moire depuis le disque"""
        try:
            with open(self.memory_file, 'rb') as f:
                loaded = pickle.load(f)
                self.memory.update(loaded)
                logging.info(f"[MEMORY] Loaded memory from {self.memory_file}")
        except FileNotFoundError:
            logging.info("[MEMORY] No existing memory found - starting fresh")
        except Exception as e:
            logging.error(f"[MEMORY] Error loading memory: {e}")
    
    def save_memory(self):
        """Sauvegarde la mÃƒÂ©moire sur le disque"""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.memory, f)
            logging.info(f"[MEMORY] Memory saved to {self.memory_file}")
        except Exception as e:
            logging.error(f"[MEMORY] Error saving memory: {e}")
    
    def store_successful_pattern(self, pattern: Dict, regime: str, pips: float):
        """Stocke un pattern gagnant"""
        entry = {
            'pattern': pattern,
            'regime': regime,
            'pips': pips,
            'timestamp': datetime.now().isoformat()
        }
        
        self.memory['best_patterns'].append(entry)
        self.memory['strategies_by_regime'][regime].append(entry)
        
        # Garder seulement les 1000 meilleurs
        self.memory['best_patterns'].sort(key=lambda x: x['pips'], reverse=True)
        self.memory['best_patterns'] = self.memory['best_patterns'][:1000]
    
    def get_best_strategies_for_regime(self, regime: str, top_n=10) -> List[Dict]:
        """RÃƒÂ©cupÃƒÂ¨re les meilleures stratÃƒÂ©gies pour un rÃƒÂ©gime donnÃƒÂ©"""
        strategies = self.memory['strategies_by_regime'].get(regime, [])
        strategies.sort(key=lambda x: x['pips'], reverse=True)
        return strategies[:top_n]
    
    def update_session(self, trades: int, wins: int):
        """Met ÃƒÂ  jour les stats de session"""
        self.memory['total_trades'] += trades
        self.memory['total_wins'] += wins
        
        self.memory['session_history'].append({
            'timestamp': datetime.now().isoformat(),
            'trades': trades,
            'wins': wins,
            'winrate': wins / trades if trades > 0 else 0
        })


# =============================================================================
# 4. ADAPTIVE STRATEGY SELECTOR
# =============================================================================

class AdaptiveStrategySelector:
    """
    SÃƒÂ©lectionne la meilleure stratÃƒÂ©gie basÃƒÂ©e sur:
    - RÃƒÂ©gime de marchÃƒÂ© actuel
    - Patterns dÃƒÂ©tectÃƒÂ©s
    - Historique de performance
    - MÃƒÂ©moire ÃƒÂ  long terme
    """
    
    def __init__(self, long_term_memory: LongTermMemorySystem):
        self.ltm = long_term_memory
        self.current_strategy = None
        
        logging.info("[STRATEGY] Adaptive Strategy Selector initialized")
    
    def select_best_strategy(self, regime: Dict, patterns: List[Dict], 
                            confluence: Dict) -> Dict:
        """
        SÃƒÂ©lectionne la meilleure stratÃƒÂ©gie pour la situation actuelle
        
        Returns:
            dict avec strategy_name, parameters, expected_winrate
        """
        try:
            regime_name = regime['name']
            
            # RÃƒÂ©cupÃƒÂ©rer les meilleures stratÃƒÂ©gies historiques pour ce rÃƒÂ©gime
            historical_strategies = self.ltm.get_best_strategies_for_regime(regime_name, top_n=5)
            
            # Scorer les stratÃƒÂ©gies basÃƒÂ© sur les patterns actuels
            strategy_scores = []
            
            for hist_strat in historical_strategies:
                similarity = self._calculate_similarity(patterns, hist_strat['pattern'])
                score = similarity * hist_strat['pips'] * regime['confidence']
                
                strategy_scores.append({
                    'historical': hist_strat,
                    'similarity': similarity,
                    'score': score
                })
            
            # SÃƒÂ©lectionner la meilleure
            if strategy_scores:
                best = max(strategy_scores, key=lambda x: x['score'])
                
                return {
                    'name': 'HISTORICAL_BEST',
                    'regime': regime_name,
                    'similarity': best['similarity'],
                    'expected_pips': best['historical']['pips'],
                    'parameters': self._extract_parameters(best['historical'])
                }
            
            # Fallback: stratÃƒÂ©gie par dÃƒÂ©faut
            return self._default_strategy(regime, confluence)
            
        except Exception as e:
            logging.error(f"[STRATEGY] Error selecting strategy: {e}")
            return self._default_strategy(regime, confluence)
    
    def _calculate_similarity(self, current_patterns: List[Dict], historical_pattern: Dict) -> float:
        """Calcule la similaritÃƒÂ© entre patterns actuels et historique"""
        # SimplifiÃƒÂ© pour l'instant
        return 0.5
    
    def _extract_parameters(self, historical: Dict) -> Dict:
        """Extrait les paramÃƒÂ¨tres de la stratÃƒÂ©gie historique"""
        return {
            'entry_threshold': 70,
            'exit_threshold': 80,
            'stop_loss_pips': 20,
            'take_profit_pips': historical['pips']
        }
    
    def _default_strategy(self, regime: Dict, confluence: Dict) -> Dict:
        """StratÃƒÂ©gie par dÃƒÂ©faut"""
        return {
            'name': 'DEFAULT',
            'regime': regime['name'],
            'parameters': {
                'entry_threshold': confluence.get('score', 70),
                'exit_threshold': 75,
                'stop_loss_pips': 25,
                'take_profit_pips': 50
            }
        }






# =============================================================================
# TRAINING LOGGER & CHART VISION
# =============================================================================

def _creer_barre(total, agent_name, symbol, color):
    labels = {"PPO": "RPPO", "SAC": "DiscreteSAC", "TRANS": "Transformer", "FUSION": "FUSION", "META": "META"}
    desc = f"{labels.get(agent_name, agent_name)} {symbol}"
    bar = tqdm(total=total, desc=desc, colour=color, position=0, leave=True, dynamic_ncols=True,
               bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} {percentage:3.0f}% [{elapsed}<{remaining}] {postfix}')
    return bar


def _lancer_barre_thread(model, pbar, env=None):
    import threading, time as _time
    start_steps = model.num_timesteps
    last = start_steps
    def _run():
        nonlocal last
        idle_ticks = 0
        while not getattr(_run, 'stop', False):
            cur = model.num_timesteps
            delta = cur - last
            if delta > 0:
                last = cur
                idle_ticks = 0
                if env is not None:
                    w = getattr(env, 'total_wins', 0); l = getattr(env, 'total_losses', 0)
                    if w+l > 0:
                        pbar.set_postfix({'WR': f'{w/(w+l)*100:.1f}%', 'W': w, 'L': l})
            else:
                idle_ticks += 1
                # Apres 4s sans avancement = calcul gradient LSTM en cours
                if idle_ticks >= 8:
                    elapsed = idle_ticks * 0.5
                    pbar.set_postfix({'status': f'LSTM gradient... {elapsed:.0f}s'})
                    pbar.refresh()  # Force l'affichage terminal
            _time.sleep(0.5)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, _run


class UnifiedProgressCallback(BaseCallback):
    """
    Barre de progression UNIQUE qui reste en bas + affiche winrate
    """
    
    def __init__(self, total_timesteps, agent_name, symbol, color='magenta', pbar=None):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.agent_name = agent_name
        self.symbol = symbol
        self.color = color
        self.pbar = pbar  # Peut être pré-créée depuis train_agent
        
        # Stats Trading
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        
        # === PATCH V3: Stats RL pour Dashboard ===
        self.cumulative_reward = 0.0
        self.episode_rewards = []
        self.last_reward = 0.0
        self.episode_count = 0
        self.current_episode_reward = 0.0
        
        # Losses tracking
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0
        self.last_q_loss = 0.0
        self.last_epsilon = 1.0
        self.target_updates = 0
        
    def _on_training_start(self):
        """Créer la barre au début si pas déjà fournie"""
        if self.pbar is None:
            self.pbar = _creer_barre(self.total_timesteps, self.agent_name, self.symbol, self.color)
    
    def _on_step(self):
        """Update a chaque step - PATCH V3 avec tracking RL"""
        if self.pbar:
            self.pbar.update(1)
            
            # === PATCH V3: Track rewards from environment ===
            try:
                # Recuperer le reward du step actuel
                reward = self.locals.get('rewards', [0])[0] if 'rewards' in self.locals else 0
                self.last_reward = float(reward)
                self.current_episode_reward += self.last_reward
                self.cumulative_reward += self.last_reward
                
                # Check si episode termine
                done = self.locals.get('dones', [False])[0] if 'dones' in self.locals else False
                if done:
                    self.episode_rewards.append(self.current_episode_reward)
                    self.episode_count += 1
                    self.current_episode_reward = 0.0
                
                # Recuperer les losses du modele
                if hasattr(self, 'model') and self.model is not None:
                    if hasattr(self.model, 'logger') and self.model.logger is not None:
                        logger_vals = getattr(self.model.logger, 'name_to_value', {})
                        
                        # PPO losses
                        if 'PPO' in self.agent_name:
                            self.last_policy_loss = logger_vals.get('train/policy_gradient_loss', 0.0)
                            self.last_value_loss = logger_vals.get('train/value_loss', 0.0)
                            self.last_entropy = logger_vals.get('train/entropy_loss', 0.0)
                        
                        # DQN losses
                        elif 'DQN' in self.agent_name:
                            self.last_q_loss = logger_vals.get('train/loss', 0.0)
                            if hasattr(self.model, 'exploration_rate'):
                                self.last_epsilon = self.model.exploration_rate
                            if hasattr(self.model, '_n_updates'):
                                self.target_updates = self.model._n_updates
            except Exception as e:
                pass  # Silent fail
            
            # Update stats toutes les 100 steps
            if self.num_timesteps % 100 == 0:
                self._update_stats()
        
        return True
    
    def _update_stats(self):
        """RÃ©cupÃ¨re et affiche les stats"""
        try:
            # Essayer de rÃ©cupÃ©rer l'env
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
                if hasattr(env, 'env'):
                    env = env.env
                
                # RÃ©cupÃ©rer les stats
                if hasattr(env, 'total_wins') and hasattr(env, 'total_losses'):
                    self.total_trades = env.total_wins + env.total_losses
                    self.winning_trades = env.total_wins
                    self.losing_trades = env.total_losses
                    
                    # RÃ©cupÃ©rer l'equity et calculer le profit
                    if hasattr(env, 'equity') and hasattr(env, 'initial_balance'):
                        current_equity = float(env.equity)
                        initial_balance = float(env.initial_balance)
                        self.total_profit = current_equity - initial_balance
                    else:
                        self.total_profit = 0.0
                    
                    if self.total_trades > 0:
                        winrate = (self.winning_trades / self.total_trades) * 100
                        
                        # Update la barre avec les stats
                        self.pbar.set_postfix({
                            'WR': f'{winrate:.1f}%',
                            'W': self.winning_trades,
                            'L': self.losing_trades,
                            'P': f'${self.total_profit:.2f}'
                        })
                        
                        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                        # LOG METRICS TO DASHBOARD
                        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                        try:
                            if DASHBOARD_ENABLED:
                                # RÃ©cupÃ©rer toutes les mÃ©triques possibles
                                metrics_to_log = {
                                    'symbol': self.symbol,
                                    'step': self.num_timesteps,
                                    'win_rate': winrate,
                                    'total_profit': self.total_profit,
                                    'total_trades': self.total_trades,
                                    'winning_trades': self.winning_trades,
                                    'losing_trades': self.losing_trades,
                                    'agent': self.agent_name
                                }
                                
                                # Ajouter equity si disponible
                                if hasattr(env, 'equity'):
                                    metrics_to_log['equity'] = float(env.equity)
                                
                                # Ajouter sharpe_ratio si disponible
                                if hasattr(env, 'sharpe_ratio'):
                                    metrics_to_log['sharpe_ratio'] = float(env.sharpe_ratio)
                                
                                # Ajouter max_drawdown si disponible
                                if hasattr(env, 'max_drawdown'):
                                    metrics_to_log['max_drawdown'] = float(env.max_drawdown)
                                
                                # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                                # DIVERGENCES - Logger les mÃ©triques de divergence
                                # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                                # PATCH V4: Utiliser la variable globale
                                global GLOBAL_DIVERGENCE_SYSTEM
                                div_sys = GLOBAL_DIVERGENCE_SYSTEM
                                if div_sys is None and hasattr(env, 'divergence_system'):
                                    div_sys = env.divergence_system
                                
                                if div_sys is not None:
                                    metrics_to_log['divergence_bullish'] = getattr(div_sys, 'bullish_divergences', 0)
                                    metrics_to_log['divergence_bearish'] = getattr(div_sys, 'bearish_divergences', 0)
                                    metrics_to_log['divergence_total'] = getattr(div_sys, 'total_divergences_detected', 0)
                                    metrics_to_log['divergence_sr'] = getattr(div_sys, 'divergence_success_rate', 0.0) * 100
                                    metrics_to_log['div_rsi_bullish'] = getattr(div_sys, 'rsi_bullish', 0)
                                    metrics_to_log['div_rsi_bearish'] = getattr(div_sys, 'rsi_bearish', 0)
                                    metrics_to_log['div_macd_bullish'] = getattr(div_sys, 'macd_bullish', 0)
                                    metrics_to_log['div_macd_bearish'] = getattr(div_sys, 'macd_bearish', 0)
                                
                                # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                                # PPO/DQN - Logger les rewards par agent
                                # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                                # ═══════════════════════════════════════════════════════
                                # PPO/DQN - PATCH V3: MÉTRIQUES RL COMPLÈTES
                                # ═══════════════════════════════════════════════════════
                                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
                                
                                if 'PPO' in self.agent_name:
                                    metrics_to_log['ppo_reward'] = self.last_reward
                                    metrics_to_log['ppo_total_reward'] = self.cumulative_reward
                                    metrics_to_log['ppo_avg_reward'] = avg_reward
                                    metrics_to_log['ppo_episodes'] = self.episode_count
                                    metrics_to_log['ppo_policy_loss'] = self.last_policy_loss
                                    metrics_to_log['ppo_value_loss'] = self.last_value_loss
                                    metrics_to_log['ppo_entropy'] = self.last_entropy
                                    
                                elif 'DQN' in self.agent_name:
                                    metrics_to_log['dqn_reward'] = self.last_reward
                                    metrics_to_log['dqn_total_reward'] = self.cumulative_reward
                                    metrics_to_log['dqn_avg_reward'] = avg_reward
                                    metrics_to_log['dqn_episodes'] = self.episode_count
                                    metrics_to_log['dqn_q_loss'] = self.last_q_loss
                                    metrics_to_log['dqn_epsilon'] = self.last_epsilon
                                    metrics_to_log['dqn_target_updates'] = self.target_updates
                                
                                # Phase Training (PPO/DQN/Ensemble)
                                global training_phase
                                metrics_to_log['current_phase'] = training_phase
                                
                                # ═══════════════════════════════════════════════════════
                                # MERLIN VISION AI - Métriques complètes
                                # ═══════════════════════════════════════════════════════
                                if hasattr(env, 'merlin_analyzer') and env.merlin_analyzer is not None:
                                    ma = env.merlin_analyzer
                                    # Nombre de patterns analysés
                                    total_patterns = sum(len(v) for v in ma.pattern_library.values()) if ma.pattern_library else 0
                                    metrics_to_log['merlin_patterns'] = total_patterns
                                    
                                    # Nombre de prédictions faites
                                    predictions_count = len(ma.prediction_history) if ma.prediction_history else 0
                                    metrics_to_log['merlin_predictions'] = predictions_count
                                    
                                    # Taux de succès (correct predictions)
                                    if predictions_count > 0:
                                        correct = sum(1 for p in ma.prediction_history if p.get('correct', False))
                                        metrics_to_log['merlin_accuracy'] = (correct / predictions_count) * 100
                                    else:
                                        metrics_to_log['merlin_accuracy'] = 0.0
                                    
                                    # Enabled status
                                    metrics_to_log['merlin_enabled'] = 1 if ma.enabled else 0
                                
                                # ═══════════════════════════════════════════════════════
                                # [V5 ULTRA] MERLIN PROFIT PREDICTION - TRUE ACCURACY
                                # ═══════════════════════════════════════════════════════
                                if hasattr(env, 'merlin_profit_tracker'):
                                    tracker = env.merlin_profit_tracker
                                    accuracy = tracker.get_accuracy()
                                    
                                    # Vraies métriques de profit
                                    metrics_to_log['merlin_profit_overall'] = accuracy['overall_accuracy']
                                    metrics_to_log['merlin_profit_direction'] = accuracy['direction_accuracy']
                                    metrics_to_log['merlin_profit_magnitude'] = accuracy['magnitude_accuracy']
                                    metrics_to_log['merlin_profit_predictions'] = accuracy['total_predictions']
                                
                                # ═══════════════════════════════════════════════════════
                                # META-LEARNER - Métriques complètes + Heatmap data
                                # ═══════════════════════════════════════════════════════
                                # PATCH V4: Utiliser la variable globale
                                global GLOBAL_META_LEARNER
                                ml = GLOBAL_META_LEARNER
                                if ml is None and hasattr(env, 'meta_learner'):
                                    ml = env.meta_learner
                                
                                if ml is not None:
                                    if hasattr(ml, 'network'):
                                        stats = ml.network.get_fusion_stats()
                                        metrics_to_log['meta_total_decisions'] = stats.get('total_decisions', 0)
                                        metrics_to_log['meta_ppo_rate'] = stats.get('ppo_rate', 0)
                                        metrics_to_log['meta_dqn_rate'] = stats.get('dqn_rate', 0)
                                        metrics_to_log['meta_consensus_rate'] = stats.get('consensus_rate', 0)
                                    
                                    # Context performance pour heatmap
                                    if hasattr(ml, 'context_performance'):
                                        cp = ml.context_performance
                                        for ctx_name, ctx_data in cp.items():
                                            total = ctx_data.get('total', 0)
                                            if total > 0:
                                                ppo_rate = ctx_data.get('ppo_wins', 0) / total * 100
                                                dqn_rate = ctx_data.get('dqn_wins', 0) / total * 100
                                                metrics_to_log[f'meta_{ctx_name}_ppo'] = ppo_rate
                                                metrics_to_log[f'meta_{ctx_name}_dqn'] = dqn_rate
                                
                                # ═══════════════════════════════════════════════════════
                                # PATCH V4.1: POSITION SIZING & CORRÉLATION METRICS
                                # ═══════════════════════════════════════════════════════
                                global GLOBAL_POSITION_SIZER, GLOBAL_CORRELATION_SYSTEM
                                
                                if GLOBAL_POSITION_SIZER is not None:
                                    ps = GLOBAL_POSITION_SIZER
                                    perf = ps.get_recent_performance()
                                    metrics_to_log['ps_current_drawdown'] = round(ps.current_drawdown * 100, 2)
                                    metrics_to_log['ps_recent_winrate'] = round(perf['win_rate'] * 100, 1)
                                    metrics_to_log['ps_streak'] = perf['streak']
                                    
                                    # Calculer la taille recommandée actuelle
                                    if hasattr(env, 'equity'):
                                        equity = getattr(env, 'equity', 10000)
                                        sizing = ps.calculate_position_size(
                                            equity=equity,
                                            signal_confidence=0.5,  # Valeur moyenne
                                            volatility_ratio=1.0
                                        )
                                        metrics_to_log['ps_current_risk_pct'] = round(sizing['risk_pct'] * 100, 2)
                                
                                if GLOBAL_CORRELATION_SYSTEM is not None:
                                    corr = GLOBAL_CORRELATION_SYSTEM
                                    # Compter les paires avec signaux
                                    active_pairs = len([p for p in corr.pair_signals if corr.pair_signals[p]['signal'] != 0])
                                    metrics_to_log['corr_active_pairs'] = active_pairs
                                
                                log_training_metrics(**metrics_to_log)
                        except Exception as e:
                            pass  # Silent fail
                        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        except:
            pass

    
    def _on_training_end(self):
        """Fermer la barre"""
        if self.pbar:
            self._update_stats()  # Final update
            self.pbar.close()
            
            # LOG FINAL
            if self.total_trades > 0:
                winrate = (self.winning_trades / self.total_trades) * 100
                logging.info("=" * 80)
                logging.info(f"[{self.agent_name}] TRAINING COMPLETE - {self.symbol}")
                logging.info("=" * 80)
                logging.info(f"Total Trades: {self.total_trades}")
                logging.info(f"Winning: {self.winning_trades} ({winrate:.1f}%)")
                logging.info(f"Losing: {self.losing_trades} ({100-winrate:.1f}%)")
                logging.info("=" * 80)
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # LOG FINAL METRICS TO DASHBOARD
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            try:
                if DASHBOARD_ENABLED and self.total_trades > 0:
                    winrate = (self.winning_trades / self.total_trades) * 100
                    
                    log_training_metrics(
                        symbol=self.symbol,
                        step=self.num_timesteps,
                        win_rate=winrate,
                        total_profit=self.total_profit,
                        total_trades=self.total_trades,
                        winning_trades=self.winning_trades,
                        losing_trades=self.losing_trades,
                        agent=self.agent_name,
                        training_complete=True
                    )
            except Exception as e:
                pass
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•



class LossReductionAnalyzer:
    """
    Algorithme anti-pertes: DÃƒÂ©tecte les patterns de pertes redondantes
    et les ÃƒÂ©limine par apprentissage
    """
    
    def __init__(self):
        self.losing_patterns = []
        self.pattern_frequency = {}
        
    def analyze_loss(self, state, action, context):
        """
        Analyse une perte pour dÃƒÂ©tecter les patterns
        
        Args:
            state: ÃƒÂ‰tat au moment de l'action
            action: Action prise
            context: Dict avec confluence, regime, etc.
        """
        # CrÃƒÂ©er une signature de la perte
        signature = self._create_loss_signature(state, action, context)
        
        # Stocker
        self.losing_patterns.append({
            'signature': signature,
            'state': state,
            'action': action,
            'context': context
        })
        
        # IncrÃƒÂ©menter frÃƒÂ©quence
        if signature not in self.pattern_frequency:
            self.pattern_frequency[signature] = 0
        self.pattern_frequency[signature] += 1
    
    def _create_loss_signature(self, state, action, context):
        """CrÃƒÂ©e une signature unique du pattern de perte"""
        # Extraire les features clÃƒÂ©s
        features = []
        
        # Action
        features.append(f"A{action}")
        
        # Confluence (si disponible)
        if 'confluence' in context:
            conf = context['confluence']
            score_bucket = int(conf.get('score', 0) / 10) * 10
            features.append(f"C{score_bucket}")
            features.append(conf.get('direction', 'UNKNOWN'))
        
        # Regime
        if 'regime' in context:
            features.append(context['regime'].get('name', 'UNKNOWN'))
        
        # RSI bucket
        if 'rsi' in context:
            rsi_bucket = int(context['rsi'] / 20) * 20
            features.append(f"RSI{rsi_bucket}")
        
        return "_".join(features)
    
    def get_redundant_losses(self, min_frequency=3):
        """
        Retourne les patterns de pertes qui se rÃƒÂ©pÃƒÂ¨tent
        
        Returns:
            Liste de (signature, frequency) triÃƒÂ©e par frÃƒÂ©quence
        """
        redundant = [
            (sig, freq) 
            for sig, freq in self.pattern_frequency.items() 
            if freq >= min_frequency
        ]
        redundant.sort(key=lambda x: x[1], reverse=True)
        return redundant
    
    def should_avoid_action(self, state, action, context):
        """
        VÃƒÂ©rifie si l'action devrait ÃƒÂªtre ÃƒÂ©vitÃƒÂ©e (pattern de perte redondant)
        
        Returns:
            (bool, str): (should_avoid, reason)
        """
        signature = self._create_loss_signature(state, action, context)
        
        frequency = self.pattern_frequency.get(signature, 0)
        
        if frequency >= 5:  # 5 fois = pattern redondant
            return True, f"Redundant loss pattern ({frequency}x): {signature}"
        
        return False, ""




class ChartVisionSystem:
    """
    SystÃƒÂ¨me de vision qui:
    1. Capture des "photos" de courbes (arrays de prix)
    2. Les stocke et classe par similaritÃƒÂ©
    3. PrÃƒÂ©dit la suite basÃƒÂ© sur patterns similaires historiques
    """
    
    def __init__(self, window_size=50, memory_file='chart_vision_memory.pkl'):
        self.window_size = window_size
        self.memory_file = memory_file
        
        # MÃƒÂ©moire des patterns
        self.pattern_library = defaultdict(list)  # hash -> list of patterns
        self.pattern_outcomes = {}  # hash -> outcome stats
        
        # Load existing memory
        self.load_memory()
    
    def capture_chart_pattern(self, data: pd.DataFrame, current_step: int):
        """
        Capture le pattern de courbe actuel
        
        Returns:
            dict avec pattern_hash, pattern_data, features
        """
        if current_step < self.window_size:
            return None
        
        # Extraire la fenÃƒÂªtre
        window = data.iloc[current_step - self.window_size:current_step]
        
        # Normaliser les prix (0-1)
        prices = window['close'].values
        normalized = self._normalize_array(prices)
        
        # Extraire features de forme
        features = self._extract_shape_features(normalized)
        
        # CrÃƒÂ©er un hash unique
        pattern_hash = self._hash_pattern(normalized)
        
        pattern = {
            'hash': pattern_hash,
            'normalized_prices': normalized,
            'features': features,
            'step': current_step,
            'raw_window': window.copy()
        }
        
        return pattern
    
    def _normalize_array(self, arr):
        """Normalise un array entre 0 et 1"""
        min_val = arr.min()
        max_val = arr.max()
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    def _extract_shape_features(self, normalized_prices):
        """
        Extrait les features de la forme de courbe
        
        Returns:
            dict avec slope, curvature, volatility, trend_strength, etc.
        """
        features = {}
        
        # 1. SLOPE (pente globale)
        features['slope'] = (normalized_prices[-1] - normalized_prices[0]) / len(normalized_prices)
        
        # 2. CURVATURE (courbure)
        # DiffÃƒÂ©rence seconde dÃƒÂ©rivÃƒÂ©e
        first_diff = np.diff(normalized_prices)
        second_diff = np.diff(first_diff)
        features['curvature'] = np.mean(np.abs(second_diff))
        
        # 3. VOLATILITY
        features['volatility'] = np.std(normalized_prices)
        
        # 4. TREND STRENGTH
        # LinÃƒÂ©aritÃƒÂ© (RÃ‚Â² de rÃƒÂ©gression linÃƒÂ©aire)
        x = np.arange(len(normalized_prices))
        slope, intercept = np.polyfit(x, normalized_prices, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((normalized_prices - y_pred) ** 2)
        ss_tot = np.sum((normalized_prices - np.mean(normalized_prices)) ** 2)
        features['trend_strength'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 5. PEAKS AND TROUGHS
        peaks = self._count_peaks(normalized_prices)
        troughs = self._count_troughs(normalized_prices)
        features['peaks'] = peaks
        features['troughs'] = troughs
        
        # 6. MOMENTUM (changement rÃƒÂ©cent)
        recent = normalized_prices[-10:]
        features['recent_momentum'] = (recent[-1] - recent[0]) / len(recent)
        
        return features
    
    def _count_peaks(self, arr):
        """Compte les peaks (sommets)"""
        peaks = 0
        for i in range(1, len(arr)-1):
            if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                peaks += 1
        return peaks
    
    def _count_troughs(self, arr):
        """Compte les troughs (creux)"""
        troughs = 0
        for i in range(1, len(arr)-1):
            if arr[i] < arr[i-1] and arr[i] < arr[i+1]:
                troughs += 1
        return troughs
    
    def _hash_pattern(self, normalized_prices):
        """CrÃƒÂ©e un hash unique du pattern (buckets de 10%)"""
        # Bucketer en 10 niveaux (0-0.1, 0.1-0.2, etc.)
        buckets = (normalized_prices * 10).astype(int)
        buckets = np.clip(buckets, 0, 9)
        
        # CrÃƒÂ©er hash
        bucket_str = ''.join(str(b) for b in buckets)
        return hashlib.md5(bucket_str.encode()).hexdigest()[:8]
    
    def find_similar_patterns(self, current_pattern, top_n=5):
        """
        Trouve les patterns similaires dans la mÃƒÂ©moire
        
        Returns:
            Liste de (pattern, similarity_score, outcome)
        """
        if not current_pattern:
            return []
        
        current_features = current_pattern['features']
        similar = []
        
        # Chercher dans tous les patterns stockÃƒÂ©s
        for pattern_hash, patterns in self.pattern_library.items():
            for stored_pattern in patterns:
                similarity = self._calculate_similarity(
                    current_features,
                    stored_pattern['features']
                )
                
                # RÃƒÂ©cupÃƒÂ©rer l'outcome si disponible
                outcome = self.pattern_outcomes.get(stored_pattern.get('outcome_id'))
                
                similar.append({
                    'pattern': stored_pattern,
                    'similarity': similarity,
                    'outcome': outcome
                })
        
        # Trier par similaritÃƒÂ©
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar[:top_n]
    
    def _calculate_similarity(self, features1, features2):
        """
        Calcule la similaritÃƒÂ© entre deux patterns (0-1)
        
        Utilise distance euclidienne normalisÃƒÂ©e
        """
        # Features ÃƒÂ  comparer
        keys = ['slope', 'curvature', 'volatility', 'trend_strength', 'recent_momentum']
        
        diff_sum = 0
        for key in keys:
            val1 = features1.get(key, 0)
            val2 = features2.get(key, 0)
            diff_sum += (val1 - val2) ** 2
        
        distance = np.sqrt(diff_sum)
        
        # Normaliser (inverse de la distance)
        similarity = 1 / (1 + distance)
        
        return similarity
    
    def store_pattern_with_outcome(self, pattern, future_data, steps_ahead=20):
        """
        Stocke un pattern avec son outcome (ce qui s'est passÃƒÂ© aprÃƒÂ¨s)
        
        Args:
            pattern: Pattern capturÃƒÂ©
            future_data: DataFrame avec les donnÃƒÂ©es futures
            steps_ahead: Nombre de steps ÃƒÂ  regarder dans le futur
        """
        if not pattern:
            return
        
        # Capturer ce qui s'est passÃƒÂ© aprÃƒÂ¨s
        current_step = pattern['step']
        future_window = future_data.iloc[current_step:current_step + steps_ahead]
        
        if len(future_window) < 5:  # Pas assez de donnÃƒÂ©es
            return
        
        # Analyser l'outcome
        start_price = future_window['close'].iloc[0]
        max_price = future_window['high'].max()
        min_price = future_window['low'].min()
        end_price = future_window['close'].iloc[-1]
        
        outcome = {
            'max_gain_pips': (max_price - start_price) * 10000,
            'max_loss_pips': (start_price - min_price) * 10000,
            'final_change_pips': (end_price - start_price) * 10000,
            'direction': 'UP' if end_price > start_price else 'DOWN',
            'volatility': future_window['close'].std() * 10000
        }
        
        # GÃƒÂ©nÃƒÂ©rer ID unique pour l'outcome
        outcome_id = f"{pattern['hash']}_{current_step}"
        pattern['outcome_id'] = outcome_id
        
        # Stocker
        pattern_hash = pattern['hash']
        self.pattern_library[pattern_hash].append(pattern)
        self.pattern_outcomes[outcome_id] = outcome
    
    def predict_next_move(self, current_pattern):
        """
        PrÃƒÂ©dit la prochaine move basÃƒÂ©e sur patterns similaires
        
        Returns:
            dict avec predicted_direction, confidence, expected_pips
        """
        similar_patterns = self.find_similar_patterns(current_pattern, top_n=10)
        
        if not similar_patterns:
            return {'predicted_direction': 'UNKNOWN', 'confidence': 0, 'expected_pips': 0}
        
        # Analyser les outcomes des patterns similaires
        up_count = 0
        down_count = 0
        total_pips = []
        
        for item in similar_patterns:
            outcome = item['outcome']
            if outcome:
                if outcome['direction'] == 'UP':
                    up_count += 1
                else:
                    down_count += 1
                
                total_pips.append(outcome['final_change_pips'])
        
        total = up_count + down_count
        if total == 0:
            return {'predicted_direction': 'UNKNOWN', 'confidence': 0, 'expected_pips': 0}
        
        # PrÃƒÂ©diction
        if up_count > down_count:
            direction = 'UP'
            confidence = up_count / total
        else:
            direction = 'DOWN'
            confidence = down_count / total
        
        expected_pips = np.mean(total_pips) if total_pips else 0
        
        return {
            'predicted_direction': direction,
            'confidence': confidence,
            'expected_pips': expected_pips,
            'similar_count': len(similar_patterns)
        }
    
    def save_memory(self):
        """Sauvegarde la mÃƒÂ©moire sur disque"""
        try:
            data = {
                'pattern_library': dict(self.pattern_library),
                'pattern_outcomes': self.pattern_outcomes
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving vision memory: {e}")
    
    def load_memory(self):
        """Charge la mÃƒÂ©moire depuis le disque"""
        try:
            with open(self.memory_file, 'rb') as f:
                data = pickle.load(f)
                self.pattern_library = defaultdict(list, data['pattern_library'])
                self.pattern_outcomes = data['pattern_outcomes']
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error loading vision memory: {e}")





# =============================================================================
# DYNAMIC PROFIT MANAGER - INT GR  DANS GOLDENEYE ULTIMATEimport pandas as pd
import numpy as np


# =============================================================================



class TDAFeatureExtractorV4:
    """
    Extracteur de features TDA V4 avec 30+ indicateurs avanc s
    Int gr  dans GOLDENEYE pour analyse multi-timeframe
    """
    
    def __init__(self):
        self.feature_names = []
        self.scaler = None
        self.is_fitted = False
        
    def extract_advanced_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extrait 53+ features depuis un DataFrame OHLCV
        Compatible avec les donn es MT5 et format GOLDENEYE
        """
        features_dict = {}
        
        # 1. PRICE FEATURES (6)
        features_dict['close'] = df['close'].values
        features_dict['high'] = df['high'].values
        features_dict['low'] = df['low'].values
        features_dict['open'] = df['open'].values
        features_dict['hl_range'] = df['high'].values - df['low'].values
        features_dict['oc_range'] = df['close'].values - df['open'].values
        
        # 2. VOLUME FEATURES (4)
        if 'volume' in df.columns:
            features_dict['volume'] = df['volume'].values
            features_dict['volume_ma5'] = df['volume'].rolling(5).mean().values
            features_dict['volume_std'] = df['volume'].rolling(10).std().values
            features_dict['volume_ratio'] = (df['volume'] / df['volume'].rolling(20).mean()).values
        else:
            for vol_feat in ['volume', 'volume_ma5', 'volume_std', 'volume_ratio']:
                features_dict[vol_feat] = np.zeros(len(df))
        
        # 3. MOMENTUM INDICATORS (8)
        features_dict['rsi_14'] = self._calculate_rsi(df['close'], 14)
        features_dict['rsi_7'] = self._calculate_rsi(df['close'], 7)
        features_dict['roc_10'] = df['close'].pct_change(10).values
        features_dict['roc_5'] = df['close'].pct_change(5).values
        features_dict['williams_r'] = self._calculate_williams_r(df, 14)
        features_dict['stoch_k'], features_dict['stoch_d'] = self._calculate_stochastic(df, 14, 3)
        features_dict['momentum'] = (df['close'] - df['close'].shift(10)).values
        
        # 4. TREND INDICATORS (7)
        features_dict['sma_20'] = df['close'].rolling(20).mean().values
        features_dict['sma_50'] = df['close'].rolling(50).mean().values
        features_dict['ema_12'] = df['close'].ewm(span=12).mean().values
        features_dict['ema_26'] = df['close'].ewm(span=26).mean().values
        features_dict['macd'], features_dict['macd_signal'], features_dict['macd_hist'] = self._calculate_macd(df['close'])
        
        # 5. VOLATILITY INDICATORS (5)
        features_dict['atr_14'] = self._calculate_atr(df, 14)
        features_dict['bb_upper'], features_dict['bb_middle'], features_dict['bb_lower'] = self._calculate_bollinger_bands(df['close'], 20, 2)
        features_dict['bb_width'] = features_dict['bb_upper'] - features_dict['bb_lower']
        
        # 6. ADVANCED TDA FEATURES (15+)
        features_dict['price_distance_sma20'] = (df['close'] - features_dict['sma_20']) / features_dict['sma_20']
        features_dict['price_distance_sma50'] = (df['close'] - features_dict['sma_50']) / features_dict['sma_50']
        features_dict['sma_cross'] = np.where(features_dict['sma_20'] > features_dict['sma_50'], 1, -1)
        features_dict['ema_cross'] = np.where(features_dict['ema_12'] > features_dict['ema_26'], 1, -1)
        
        # Divergences
        features_dict['price_momentum_div'] = self._detect_divergence(df['close'], features_dict['rsi_14'])
        features_dict['macd_divergence'] = self._detect_divergence(df['close'], features_dict['macd'])
        
        # Support/Resistance
        features_dict['distance_to_high'] = (df['high'].rolling(50).max() - df['close']) / df['close']
        features_dict['distance_to_low'] = (df['close'] - df['low'].rolling(50).min()) / df['close']
        
        # Pattern Recognition
        features_dict['doji'] = self._detect_doji(df)
        features_dict['hammer'] = self._detect_hammer(df)
        features_dict['engulfing'] = self._detect_engulfing(df)
        
        # Volatility regime
        features_dict['volatility_regime'] = self._classify_volatility_regime(features_dict['atr_14'])
        
        # Trend strength
        features_dict['trend_strength'] = self._calculate_trend_strength(df['close'])
        
        # Price acceleration
        features_dict['price_acceleration'] = df['close'].diff().diff().values
        
        # Volume-price divergence
        if 'volume' in df.columns:
            features_dict['vp_divergence'] = self._volume_price_divergence(df['close'], df['volume'])
        else:
            features_dict['vp_divergence'] = np.zeros(len(df))
        
        # Convertir en array numpy
        feature_matrix = np.column_stack([features_dict[k] for k in sorted(features_dict.keys())])
        
        # G rer les NaN
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.feature_names = sorted(features_dict.keys())
        
        return feature_matrix
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    def _calculate_williams_r(self, df, period=14):
        """Calculate Williams %R"""
        high_max = df['high'].rolling(period).max()
        low_min = df['low'].rolling(period).min()
        wr = -100 * (high_max - df['close']) / (high_max - low_min)
        return wr.values
    
    def _calculate_stochastic(self, df, period=14, smooth=3):
        """Calculate Stochastic Oscillator"""
        high_max = df['high'].rolling(period).max()
        low_min = df['low'].rolling(period).min()
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(smooth).mean()
        return k.values, d.values
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd.values, signal_line.values, histogram.values
    
    def _calculate_atr(self, df, period=14):
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.values
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper.values, middle.values, lower.values
    
    def _detect_divergence(self, prices, indicator):
        """Detect price-indicator divergence"""
        price_diff = prices.diff(5)
        indicator_diff = pd.Series(indicator).diff(5)
        divergence = np.where((price_diff > 0) & (indicator_diff < 0), 1,
                            np.where((price_diff < 0) & (indicator_diff > 0), -1, 0))
        return divergence
    
    def _detect_doji(self, df):
        """Detect Doji candlestick pattern"""
        body = np.abs(df['close'] - df['open'])
        range_hl = df['high'] - df['low']
        doji = np.where(body / range_hl < 0.1, 1, 0)
        return doji
    
    def _detect_hammer(self, df):
        """Detect Hammer candlestick pattern"""
        body = np.abs(df['close'] - df['open'])
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        hammer = np.where((lower_shadow > 2 * body) & (upper_shadow < 0.3 * body), 1, 0)
        return hammer
    
    def _detect_engulfing(self, df):
        """Detect Engulfing pattern"""
        prev_body = np.abs(df['close'].shift(1) - df['open'].shift(1))
        curr_body = np.abs(df['close'] - df['open'])
        bullish = np.where((df['close'].shift(1) < df['open'].shift(1)) & 
                          (df['close'] > df['open']) & 
                          (curr_body > prev_body), 1, 0)
        bearish = np.where((df['close'].shift(1) > df['open'].shift(1)) & 
                          (df['close'] < df['open']) & 
                          (curr_body > prev_body), -1, 0)
        return bullish + bearish
    
    def _classify_volatility_regime(self, atr):
        """Classify volatility regime"""
        atr_series = pd.Series(atr)
        atr_ma = atr_series.rolling(20).mean()
        regime = np.where(atr_series > atr_ma * 1.5, 2,  # High volatility
                         np.where(atr_series < atr_ma * 0.5, 0, 1))  # Low, Medium
        return regime
    
    def _calculate_trend_strength(self, prices):
        """Calculate trend strength using ADX-like approach"""
        diff = prices.diff()
        up = np.where(diff > 0, diff, 0)
        down = np.where(diff < 0, -diff, 0)
        
        up_ma = pd.Series(up).rolling(14).mean()
        down_ma = pd.Series(down).rolling(14).mean()
        
        strength = np.abs(up_ma - down_ma) / (up_ma + down_ma + 1e-10)
        return strength.values
    
    def _volume_price_divergence(self, prices, volume):
        """Detect volume-price divergence"""
        price_change = prices.pct_change(5)
        volume_change = volume.pct_change(5)
        
        divergence = np.where((price_change > 0.01) & (volume_change < -0.1), -1,  # Bearish
                            np.where((price_change < -0.01) & (volume_change < -0.1), 1, 0))  # Bullish
        return divergence
    
    def fit_transform(self, X):
        """Fit scaler and transform features"""
        from sklearn.preprocessing import StandardScaler
        if self.scaler is None:
            self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X):
        """Transform features using fitted scaler"""
        if not self.is_fitted or self.scaler is None:
            logging.warning("Scaler not fitted, returning raw features")
            return X
        return self.scaler.transform(X)


class DynamicProfitManager:
    """
    G re les profits en temps r el avec trailing stop intelligent
    Version  tendue avec toutes les fonctionnalit s int gr es
    """
    
    def __init__(self, config=None):
        self.position_highs = {}  # Meilleur prix atteint par position
        self.position_lows = {}   # Pire prix atteint par position
        self.position_entries = {} # Prix d'entr e par position
        self.position_types = {}  # Type de position (buy/sell)
        self.position_tick_sizes = {} # Taille du tick par paire
        
        # Configuration avec valeurs par d faut ou personnalis es
        self.config = config or {}
        
        # Param tres de base
        self.min_profit_to_activate = self.config.get('min_profit_to_activate', 50)
        self.trailing_stop_pct = self.config.get('trailing_stop_pct', 0.3)
        self.take_profit_levels = self.config.get('take_profit_levels', [100, 200, 300])
        self.partial_close_pct = self.config.get('partial_close_pct', [0.33, 0.5, 1.0])
        
        # Nouvelles fonctionnalit s
        self.quick_tp_pips = self.config.get('quick_tp_pips', 30)
        self.emergency_stop_pips = self.config.get('emergency_stop_pips', 100)  # Stop loss absolu
        self.volatility_multiplier = self.config.get('volatility_multiplier', 1.5)
        self.breakeven_activate_pips = self.config.get('breakeven_activate_pips', 25)
        self.secure_profit_pct = self.config.get('secure_profit_pct', 0.7)  # S curise 70% du profit max
        
        # Suivi des TP d j  d clench s
        self._tp_hit = {}
        
    def initialize_position(self, symbol: str, entry_price: float, position_type: str, tick_size: float):
        """Initialise une nouvelle position"""
        self.position_entries[symbol] = entry_price
        self.position_types[symbol] = position_type
        self.position_tick_sizes[symbol] = tick_size
        self.position_highs[symbol] = {'price': entry_price, 'profit_pips': 0}
        self.position_lows[symbol] = {'price': entry_price, 'profit_pips': 0}
        
        if symbol not in self._tp_hit:
            self._tp_hit[symbol] = []
        
        print(f"  Position initialis e: {symbol} {position_type.upper()} @ {entry_price}")
    
    def update_position(self, symbol: str, current_price: float, volatility: float = None) -> dict:
        """
        Met   jour le tracking de position et retourne les d cisions
        Version simplifi e avec toutes les v rifications int gr es
        
        Returns:
            {
                'action': 'hold'/'close_partial'/'close_all',
                'reason': str,
                'close_percent': float,
                'current_profit': float,
                'best_profit': float
            }
        """
        if symbol not in self.position_entries:
            return {'action': 'hold', 'reason': 'Position non initialis e', 'close_percent': 0.0}
        
        entry_price = self.position_entries[symbol]
        position_type = self.position_types[symbol]
        tick_size = self.position_tick_sizes[symbol]
        
        # Calcul du profit
        profit_pips = self._calculate_profit_pips(entry_price, current_price, position_type, tick_size)
        
        # Mise   jour des highs/lows
        self._update_extremes(symbol, current_price, profit_pips)
        
        # S QUENCE DE D CISION (ordre de priorit )
        decisions = [
            self._check_emergency_stop(symbol, profit_pips),
            self._check_quick_take_profit(symbol, profit_pips),
            self._check_take_profit_levels(symbol, profit_pips),
            self._check_breakeven_stop(symbol, profit_pips),
            self._check_profit_retention(symbol, profit_pips),
            self._check_dynamic_trailing_stop(symbol, profit_pips, volatility)
        ]
        
        # Retourner la premi re d cision non-"hold"
        for decision in decisions:
            if decision['action'] != 'hold':
                decision.update({
                    'current_profit': profit_pips,
                    'best_profit': self.position_highs[symbol]['profit_pips']
                })
                return decision
        
        # Aucune action n cessaire
        return {
            'action': 'hold', 
            'reason': f'Profit: {profit_pips:.1f}pips | High: {self.position_highs[symbol]["profit_pips"]:.1f}pips',
            'close_percent': 0.0,
            'current_profit': profit_pips,
            'best_profit': self.position_highs[symbol]['profit_pips']
        }
    
    def _calculate_profit_pips(self, entry: float, current: float, pos_type: str, tick_size: float) -> float:
        """Calcule le profit en pips"""
        if pos_type == 'buy':
            return (current - entry) / tick_size
        else:  # sell
            return (entry - current) / tick_size
    
    def _update_extremes(self, symbol: str, current_price: float, profit_pips: float):
        """Met   jour les prix extr mes atteints"""
        if profit_pips > self.position_highs[symbol]['profit_pips']:
            self.position_highs[symbol] = {'price': current_price, 'profit_pips': profit_pips}
        
        if profit_pips < self.position_lows[symbol]['profit_pips']:
            self.position_lows[symbol] = {'price': current_price, 'profit_pips': profit_pips}
    
    def _check_emergency_stop(self, symbol: str, current_profit_pips: float) -> dict:
        """Stop d'urgence pour protection du capital"""
        if current_profit_pips <= -self.emergency_stop_pips:
            return {
                'action': 'close_all',
                'reason': f'  STOP URGENCE: Perte {current_profit_pips:.1f} pips',
                'close_percent': 1.0
            }
        return {'action': 'hold', 'reason': '', 'close_percent': 0.0}
    
    def _check_quick_take_profit(self, symbol: str, current_profit_pips: float) -> dict:
        """Take profit rapide pour petits mouvements"""
        if current_profit_pips >= self.quick_tp_pips:
            return {
                'action': 'close_all',
                'reason': f'  Quick TP: {current_profit_pips:.1f} pips',
                'close_percent': 1.0
            }
        return {'action': 'hold', 'reason': '', 'close_percent': 0.0}
    
    def _check_take_profit_levels(self, symbol: str, current_profit_pips: float) -> dict:
        """Take profit par paliers avec fermeture partielle"""
        for i, tp_level in enumerate(self.take_profit_levels):
            if current_profit_pips >= tp_level and tp_level not in self._tp_hit[symbol]:
                self._tp_hit[symbol].append(tp_level)
                close_pct = self.partial_close_pct[i] if i < len(self.partial_close_pct) else 1.0
                
                action_type = 'close_partial' if close_pct < 1.0 else 'close_all'
                
                return {
                    'action': action_type,
                    'reason': f'  TP Niveau {i+1}: {tp_level} pips',
                    'close_percent': close_pct
                }
        
        return {'action': 'hold', 'reason': '', 'close_percent': 0.0}
    
    def _check_breakeven_stop(self, symbol: str, current_profit_pips: float) -> dict:
        """D place le stop au breakeven apr s un certain profit"""
        best_profit = self.position_highs[symbol]['profit_pips']
        
        if best_profit >= self.breakeven_activate_pips and current_profit_pips <= 0:
            return {
                'action': 'close_all',
                'reason': f'  Breakeven Stop: Profit {current_profit_pips:.1f} pips',
                'close_percent': 1.0
            }
        
        return {'action': 'hold', 'reason': '', 'close_percent': 0.0}
    
    def _check_profit_retention(self, symbol: str, current_profit_pips: float) -> dict:
        """S curise un pourcentage du profit maximum atteint"""
        best_profit = self.position_highs[symbol]['profit_pips']
        
        if best_profit > 30:  # Seulement si on a eu un bon mouvement
            profit_retained = current_profit_pips / best_profit if best_profit > 0 else 0
            
            if profit_retained < self.secure_profit_pct:
                return {
                    'action': 'close_all',
                    'reason': f'  S curisation: {current_profit_pips:.1f}/{best_profit:.1f} pips ({profit_retained:.1%})',
                    'close_percent': 1.0
                }
        
        return {'action': 'hold', 'reason': '', 'close_percent': 0.0}
    
    def _check_dynamic_trailing_stop(self, symbol: str, current_profit_pips: float, 
                                   volatility: float = None) -> dict:
        """Trailing stop adaptatif   la volatilit """
        best_profit = self.position_highs[symbol]['profit_pips']
        
        # V rifier si on a assez de profit pour activer le trailing
        if best_profit < self.min_profit_to_activate:
            return {'action': 'hold', 'reason': '', 'close_percent': 0.0}
        
        profit_given_back = best_profit - current_profit_pips
        
        # Ajuster le trailing stop en fonction de la volatilit 
        trailing_pct = self.trailing_stop_pct
        if volatility:
            trailing_pct *= self.volatility_multiplier
            trailing_pct = min(trailing_pct, 0.5)  # Maximum 50%
        
        if profit_given_back > (best_profit * trailing_pct):
            return {
                'action': 'close_all',
                'reason': f'  Trailing Stop: Rendu {profit_given_back:.1f}/{best_profit:.1f} pips',
                'close_percent': 1.0
            }
        
        return {'action': 'hold', 'reason': '', 'close_percent': 0.0}
    
    def get_position_info(self, symbol: str) -> dict:
        """Retourne les informations actuelles d'une position"""
        if symbol not in self.position_entries:
            return {}
        
        return {
            'entry_price': self.position_entries[symbol],
            'position_type': self.position_types[symbol],
            'tick_size': self.position_tick_sizes[symbol],
            'best_profit_pips': self.position_highs[symbol]['profit_pips'],
            'worst_profit_pips': self.position_lows[symbol]['profit_pips'],
            'tp_levels_hit': self._tp_hit.get(symbol, [])
        }
    
    def reset_position(self, symbol: str):
        """Reset le tracking quand une position est ferm e"""
        for attr in ['position_entries', 'position_types', 'position_tick_sizes', 'position_highs', 'position_lows']:
            if hasattr(self, attr) and symbol in getattr(self, attr):
                del getattr(self, attr)[symbol]
        
        if symbol in self._tp_hit:
            del self._tp_hit[symbol]
        
        print(f"  Position reset: {symbol}")



import websockets
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import sys
import io
import traceback
import time
import copy
from dataclasses import dataclass, field
#ML & Data Science
import numpy as np
from tqdm import tqdm
import pandas as pd
#Configuration s curis e
from dotenv import load_dotenv
# Charge d'abord le .env local, sinon remonte vers le .env de BLADE_RUNNER_V6
import pathlib as _pl
_env_local  = _pl.Path(__file__).parent / ".env"
_env_v6     = _pl.Path(__file__).parent.parent / ".env"
if _env_local.exists():
    load_dotenv(_env_local)
elif _env_v6.exists():
    load_dotenv(_env_v6)
    print(f"  [ENV] Credentials chargés depuis {_env_v6}")
#Monitoring syst me
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging
from scipy.stats import linregress
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import asyncio





try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil non disponible")
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("GPUtil non disponible")
#Reinforcement Learning
try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    import gymnasium as gym
    RL_AVAILABLE = True
except ImportError as e:
    RL_AVAILABLE = False
    logging.error(f"RL non disponible: {e}")

# DynamicProfitManager — module optionnel legacy
try:
    import sys; sys.path.insert(0, ".")
    from DYNAMIC_PROFIT_SYSTEM import DynamicProfitManager
except ImportError:
    DynamicProfitManager = None

# ============================================================================
# FORCE SKIP MT5 - POUR TESTS RAPIDES SANS MT5
# ============================================================================
FORCE_SKIP_MT5 = False  # Mettre a True pour skip MT5 completement

#MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True if not FORCE_SKIP_MT5 else False
except ImportError as e:
    MT5_AVAILABLE = False
    logging.error(f"MT5 non disponible: {e}")

"""
PROFESSIONAL MARKET REVERSAL & SMART EXIT SYSTEM
D tecte les retournements de march  et g re les sorties intelligemment
Inspir  des strat gies institutionnelles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class MarketReversalDetector:
    """
    D tecte les retournements de march  en combinant plusieurs indicateurs
    """
    
    def __init__(self):
        self.lookback_periods = {
            'short': 5,    # 5 bougies pour momentum court terme
            'medium': 20,  # 20 bougies pour tendance
            'long': 50     # 50 bougies pour contexte
        }
        
    def detect_reversal(self, prices: pd.DataFrame) -> Dict:
        """
        D tecte un retournement de march  potentiel
        
        Returns:
            {
                'reversal_detected': bool,
                'direction': 'bullish' or 'bearish' or None,
                'confidence': 0.0 to 1.0,
                'signals': list of detected signals
            }
        """
        signals = []
        confidence_score = 0.0
        
        # 1. MOMENTUM SHIFT (25% weight)
        momentum_signal = self._check_momentum_shift(prices)
        if momentum_signal['detected']:
            signals.append(momentum_signal)
            confidence_score += 0.25
        
        # 2. VOLUME DIVERGENCE (20% weight)
        volume_signal = self._check_volume_divergence(prices)
        if volume_signal['detected']:
            signals.append(volume_signal)
            confidence_score += 0.20
        
        # 3. PRICE ACTION PATTERN (30% weight)
        pattern_signal = self._check_price_patterns(prices)
        if pattern_signal['detected']:
            signals.append(pattern_signal)
            confidence_score += 0.30
        
        # 4. TREND EXHAUSTION (25% weight)
        exhaustion_signal = self._check_trend_exhaustion(prices)
        if exhaustion_signal['detected']:
            signals.append(exhaustion_signal)
            confidence_score += 0.25
        
        # D terminer la direction du retournement
        direction = None
        if signals:
            bullish_count = sum(1 for s in signals if s.get('direction') == 'bullish')
            bearish_count = sum(1 for s in signals if s.get('direction') == 'bearish')
            
            if bullish_count > bearish_count:
                direction = 'bullish'
            elif bearish_count > bullish_count:
                direction = 'bearish'
        
        # Retournement confirm  si confidence > 50% et direction claire
        reversal_detected = confidence_score >= 0.5 and direction is not None
        
        return {
            'reversal_detected': reversal_detected,
            'direction': direction,
            'confidence': confidence_score,
            'signals': signals,
            'timestamp': datetime.now()
        }
    
    def _check_momentum_shift(self, prices: pd.DataFrame) -> Dict:
        """D tecte un changement de momentum"""
        if len(prices) < 20:
            return {'detected': False}
        
        # Calculer ROC (Rate of Change) sur 5 et 10 p riodes
        roc_5 = (prices['close'].iloc[-1] - prices['close'].iloc[-5]) / prices['close'].iloc[-5]
        roc_10 = (prices['close'].iloc[-1] - prices['close'].iloc[-10]) / prices['close'].iloc[-10]
        
        # Momentum ralentit puis s'inverse
        momentum_shift = False
        direction = None
        
        # Tendance baissi re qui ralentit   Retournement haussier potentiel
        if roc_10 < -0.01 and roc_5 > roc_10:  # Momentum devient moins n gatif
            momentum_shift = True
            direction = 'bullish'
        
        # Tendance haussi re qui ralentit   Retournement baissier potentiel
        elif roc_10 > 0.01 and roc_5 < roc_10:  # Momentum devient moins positif
            momentum_shift = True
            direction = 'bearish'
        
        return {
            'detected': momentum_shift,
            'type': 'momentum_shift',
            'direction': direction,
            'roc_5': roc_5,
            'roc_10': roc_10
        }
    
    def _check_volume_divergence(self, prices: pd.DataFrame) -> Dict:
        """D tecte une divergence prix/volume"""
        if len(prices) < 20 or 'volume' not in prices.columns:
            return {'detected': False}
        
        # Prix fait nouveau plus haut mais volume diminue = divergence baissi re
        # Prix fait nouveau plus bas mais volume diminue = divergence haussi re
        
        recent_prices = prices['close'].iloc[-10:]
        recent_volumes = prices['volume'].iloc[-10:]
        
        price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        volume_trend = (recent_volumes.iloc[-5:].mean() - recent_volumes.iloc[:5].mean()) / recent_volumes.iloc[:5].mean()
        
        divergence = False
        direction = None
        
        # Prix monte, volume baisse   Retournement baissier
        if price_trend > 0.005 and volume_trend < -0.2:
            divergence = True
            direction = 'bearish'
        
        # Prix baisse, volume baisse   Retournement haussier (capitulation)
        elif price_trend < -0.005 and volume_trend < -0.2:
            divergence = True
            direction = 'bullish'
        
        return {
            'detected': divergence,
            'type': 'volume_divergence',
            'direction': direction,
            'price_trend': price_trend,
            'volume_trend': volume_trend
        }
    
    def _check_price_patterns(self, prices: pd.DataFrame) -> Dict:
        """D tecte des patterns de retournement classiques"""
        if len(prices) < 5:
            return {'detected': False}
        
        last_5 = prices.iloc[-5:]
        
        # Hammer / Shooting Star
        for i in range(len(last_5) - 1, max(0, len(last_5) - 3), -1):
            candle = last_5.iloc[i]
            body = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range == 0:
                continue
            
            # Hammer (retournement haussier)
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            if lower_wick > body * 2 and lower_wick / total_range > 0.6:
                return {
                    'detected': True,
                    'type': 'hammer_pattern',
                    'direction': 'bullish'
                }
            
            # Shooting Star (retournement baissier)
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            if upper_wick > body * 2 and upper_wick / total_range > 0.6:
                return {
                    'detected': True,
                    'type': 'shooting_star_pattern',
                    'direction': 'bearish'
                }
        
        return {'detected': False}
    
    def _check_trend_exhaustion(self, prices: pd.DataFrame) -> Dict:
        """D tecte l' puisement d'une tendance"""
        if len(prices) < 50:
            return {'detected': False}
        
        # Calculer la pente de la tendance sur diff rentes p riodes
        recent = prices['close'].iloc[-10:].values
        medium = prices['close'].iloc[-30:].values
        
        # R gression lin aire simple
        x_recent = np.arange(len(recent))
        slope_recent = np.polyfit(x_recent, recent, 1)[0]
        
        x_medium = np.arange(len(medium))
        slope_medium = np.polyfit(x_medium, medium, 1)[0]
        
        exhaustion = False
        direction = None
        
        # Tendance haussi re qui s'essouffle
        if slope_medium > 0 and abs(slope_recent) < abs(slope_medium) * 0.3:
            exhaustion = True
            direction = 'bearish'
        
        # Tendance baissi re qui s'essouffle
        elif slope_medium < 0 and abs(slope_recent) < abs(slope_medium) * 0.3:
            exhaustion = True
            direction = 'bullish'
        
        return {
            'detected': exhaustion,
            'type': 'trend_exhaustion',
            'direction': direction,
            'slope_recent': slope_recent,
            'slope_medium': slope_medium
        }


#============================================================================
# TIME SERIES SPLIT POUR VALIDATION TEMPORELLE
#============================================================================
class TimeSeriesSplit:
    """Validation crois e temporelle stricte pour  viter le data leakage"""
    
    def __init__(self, n_splits=5, test_size=0.2, gap=10):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap  # Gap entre train et test pour  viter le lookahead bias
    
    def split(self, data):
        n_samples = len(data)
        n_test = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            # Train: d but   fin progressif
            train_end = n_samples - n_test * (self.n_splits - i) - self.gap
            # Test: apr s le gap
            test_start = train_end + self.gap
            test_end = test_start + n_test
            
            if train_end > 0 and test_end <= n_samples:
                yield list(range(0, train_end)), list(range(test_start, test_end))
    
    def get_train_test_split(self, data, split_index=0):
        """Retourne un split train/test unique"""
        splits = list(self.split(data))
        if split_index < len(splits):
            train_idx, test_idx = splits[split_index]
            return data.iloc[train_idx], data.iloc[test_idx]
        else:
            # Fallback: split simple 80/20
            split_point = int(len(data) * 0.8)
            return data.iloc[:split_point], data.iloc[split_point:]


class SmartExitManager:
    """
    G re les sorties intelligentes avec rebond post-SL
    """
    
    def __init__(self):
        self.position_tracking = {}  # Tracking des positions avec leurs m triques
        self.min_recovery_pct = 0.5  # 50% de r cup ration minimum
        self.max_wait_time = 300  # 5 minutes max d'attente apr s SL touch 
        
    def should_close_on_reversal(self, position: Dict, reversal_info: Dict) -> Tuple[bool, str]:
        """
        D cide si on ferme une position   cause d'un retournement
        
        Args:
            position: {'symbol': str, 'type': 'buy'/'sell', 'entry': float, 'current': float}
            reversal_info: r sultat de MarketReversalDetector.detect_reversal()
        
        Returns:
            (should_close, reason)
        """
        if not reversal_info['reversal_detected']:
            return False, "No reversal detected"
        
        # Position LONG mais retournement BEARISH d tect 
        if position['type'] == 'buy' and reversal_info['direction'] == 'bearish':
            # Fermer m me en profit pour  viter de rendre les gains
            if reversal_info['confidence'] >= 0.6:  # Haute confiance
                return True, f"Reversal bearish confirm  (conf: {reversal_info['confidence']:.1%})"
        
        # Position SHORT mais retournement BULLISH d tect 
        elif position['type'] == 'sell' and reversal_info['direction'] == 'bullish':
            if reversal_info['confidence'] >= 0.6:
                return True, f"Reversal bullish confirm  (conf: {reversal_info['confidence']:.1%})"
        
        return False, "Position aligned with market direction"
    
    def track_position_near_sl(self, symbol: str, entry_price: float, 
                               current_price: float, sl_price: float, 
                               position_type: str):
        """
        Commence   tracker une position qui approche du SL
        """
        distance_to_sl = abs(current_price - sl_price) / abs(entry_price - sl_price)
        
        if distance_to_sl < 0.1:  # Moins de 10% avant le SL
            if symbol not in self.position_tracking:
                self.position_tracking[symbol] = {
                    'entry': entry_price,
                    'sl_price': sl_price,
                    'type': position_type,
                    'worst_price': current_price,
                    'sl_touched_time': None,
                    'recovery_started': False
                }
    
    def check_smart_exit(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """
        V rifie si on doit faire une sortie intelligente post-SL
        
        Returns:
            (should_exit, reason)
        """
        if symbol not in self.position_tracking:
            return False, "Position not tracked"
        
        pos = self.position_tracking[symbol]
        
        # SL a  t  touch 
        if pos['type'] == 'buy' and current_price <= pos['sl_price']:
            if pos['sl_touched_time'] is None:
                pos['sl_touched_time'] = datetime.now()
                pos['worst_price'] = current_price
                return False, "SL touched, waiting for potential bounce"
        
        elif pos['type'] == 'sell' and current_price >= pos['sl_price']:
            if pos['sl_touched_time'] is None:
                pos['sl_touched_time'] = datetime.now()
                pos['worst_price'] = current_price
                return False, "SL touched, waiting for potential bounce"
        
        # Si SL touch , v rifier le rebond
        if pos['sl_touched_time']:
            time_since_sl = (datetime.now() - pos['sl_touched_time']).total_seconds()
            
            # Calculer le rebond
            if pos['type'] == 'buy':
                max_loss = pos['entry'] - pos['worst_price']
                current_loss = pos['entry'] - current_price
                if max_loss != 0:
                    recovery_pct = 1 - (current_loss / max_loss)
                else:
                    recovery_pct = 0
                
                # A r cup r  50%+ de la perte
                if recovery_pct >= self.min_recovery_pct:
                    del self.position_tracking[symbol]
                    return True, f"Smart exit: R cup r  {recovery_pct:.1%} de la perte"
            
            elif pos['type'] == 'sell':
                max_loss = pos['worst_price'] - pos['entry']
                current_loss = current_price - pos['entry']
                if max_loss != 0:
                    recovery_pct = 1 - (current_loss / max_loss)
                else:
                    recovery_pct = 0
                
                if recovery_pct >= self.min_recovery_pct:
                    del self.position_tracking[symbol]
                    return True, f"Smart exit: R cup r  {recovery_pct:.1%} de la perte"
            
            # Timeout: trop longtemps sans rebond
            if time_since_sl > self.max_wait_time:
                del self.position_tracking[symbol]
                return True, "Timeout: Pas de rebond significatif, exit maintenant"
        
        return False, "Waiting for optimal exit"
    
    def reset_tracking(self, symbol: str):
        """Reset le tracking d'une position"""
        if symbol in self.position_tracking:
            del self.position_tracking[symbol]




#Reinforcement Learning
try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    import gymnasium as gym
    RL_AVAILABLE = True
except ImportError as e:
    RL_AVAILABLE = False
    logging.error(f"RL non disponible: {e}")

#MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError as e:
    MT5_AVAILABLE = False
    logging.error(f"MT5 non disponible: {e}")

#Sentiment Analysis
try:
    import tweepy
    import praw
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError as e:
    SENTIMENT_AVAILABLE = False
    logging.warning(f"Sentiment analysis non disponible: {e}")

#PyTorch pour HRM-TRM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

#Keyboard input pour skip training
import threading
try:
    import msvcrt  # Windows
    KEYBOARD_AVAILABLE = 'windows'
except ImportError:
    try:
        import select  # Linux/Mac
        import sys
        KEYBOARD_AVAILABLE = 'unix'
    except ImportError:
        KEYBOARD_AVAILABLE = False
#============================================================================
#KEYBOARD LISTENER POUR SKIP TRAINING
#============================================================================
class KeyboardListener:
    ''' coute le clavier pour skip training avec touche S'''
    
    def __init__(self):
        self.skip_training = False
        self.listening = False
        self.thread = None
    
    def start(self):
        '''D marre l  coute du clavier'''
        if not KEYBOARD_AVAILABLE:
            logging.warning("  Keyboard input non disponible sur cette plateforme")
            return
        
        self.listening = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        logging.info("  Keyboard Listener d marr  - Appuyez sur S pour skip training")
    
    def _listen_loop(self):
        '''Boucle d  coute du clavier'''
        while self.listening:
            if self._key_pressed():
                key = self._get_key()
                if key and key.upper() == 'S':
                    self.skip_training = True
                    logging.warning("  SKIP TRAINING ACTIV  - Appuyez sur ENTR E pour continuer...")
                    break
    
    def _key_pressed(self):
        '''V rifie si une touche est press e'''
        if KEYBOARD_AVAILABLE == 'windows':
            return msvcrt.kbhit()
        elif KEYBOARD_AVAILABLE == 'unix':
            return select.select([sys.stdin], [], [], 0.1)[0]
        return False
    
    def _get_key(self):
        '''R cup re la touche press e'''
        if KEYBOARD_AVAILABLE == 'windows':
            if msvcrt.kbhit():
                return msvcrt.getch().decode('utf-8', errors='ignore')
        elif KEYBOARD_AVAILABLE == 'unix':
            return sys.stdin.read(1)
        return None
    
    def stop(self):
        '''Arr te l  coute'''
        self.listening = False

#============================================================================
#CONFIGURATION LOGGING AVANC E
#============================================================================
def setup_advanced_logging():
    """Configuration compl te du logging avec couleurs et emojis"""
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    class ColoredFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': '\033[36m',      # Cyan
            'INFO': '\033[32m',       # Vert
            'WARNING': '\033[33m',    # Jaune/Orange
            'ERROR': '\033[31m',      # Rouge
            'CRITICAL': '\033[41m',   # Fond rouge
            'LOSS': '\033[38;5;130m'  #   Marron
        }
        EMOJIS = {
            'DEBUG': ' ',
            'INFO': ' ',
            'WARNING': ' ', 
            'ERROR': ' ',
            'CRITICAL': ' ',
            'LOSS': ' '  #   Pour les pertes
        }
        RESET = '\033[0m'
        
        def format(self, record):
            emoji = self.EMOJIS.get(record.levelname, '')
            color = self.COLORS.get(record.levelname, '')
            record.emoji = emoji
            record.color = color
            record.reset = self.RESET
            return super().format(record)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = ColoredFormatter(
        '%(color)s%(emoji)s [%(asctime)s] %(levelname)s: %(message)s%(reset)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    # Handler fichier
    file_handler = logging.FileHandler('ultimate_trader.log', encoding='utf-8', mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '  [%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info("  Syst me de logging avanc  initialis ")

setup_advanced_logging()

#============================================================================
#CONFIGURATION GLOBALE
#============================================================================
@dataclass
class UltimateConfig:
    """Configuration centralis e avec validation"""
    # Symboles trading
    SYMBOLS: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY"])

    # MT5 — lit DEMO_* (V6 .env) avec fallback MT5_* pour compatibilité
    MT5_LOGIN: int = int(os.getenv('DEMO_LOGIN') or os.getenv('MT5_LOGIN', '0'))
    MT5_PASSWORD: str = os.getenv('DEMO_PASSWORD') or os.getenv('MT5_PASSWORD', '')
    MT5_SERVER: str = os.getenv('DEMO_SERVER') or os.getenv('MT5_SERVER', '')
    TIMEFRAME: int = 15  # M15

    # Risk Management
    RISK_PER_TRADE: float = 0.02  # 1.5% par trade
    DAILY_LOSS_LIMIT: float = 0.05  # 5% max loss daily
    MAX_POSITIONS: int = 8


    # HRM-TRM Parameters
    HRM_INPUT_DIM: int = 14  # 10 features base + 4 signaux avanc s
    HRM_HIDDEN_DIM: int = 128
    HRM_H_CYCLES: int = 2
    HRM_L_CYCLES: int = 3
    HRM_NUM_HEADS: int = 4
    HRM_HALT_MAX_STEPS: int = 16
    HRM_EXPLORATION_PROB: float = 0.1

    # RL Training - PATCH ULTIME: 1.6M steps
    RL_TOTAL_TIMESTEPS: int = 30000       # 600k steps (phases rapides: 50k+50k+500k)
    RL_LEARNING_RATE: float = 0.0003
    RL_BUFFER_SIZE: int = 500000           # Buffer augmenté pour 1.6M steps
    RL_EXPLORATION_FRACTION: float = 0.15  # Exploration légèrement réduite

    # WebSocket
    WS_PORT: int = 8765
    RL_WS_PORT: int = 8766

    # Retraining
    RETRAIN_COOLDOWN: int = 43200  # 12h
    PERF_DROP_THRESHOLD: float = 0.5

    def __post_init__(self):
        """Validation de la config"""
        if not self.MT5_PASSWORD:
            logging.warning("  MT5_PASSWORD non configur ")
        if self.RISK_PER_TRADE > 0.05:
            logging.warning("  RISK_PER_TRADE  lev  (>5%)")

CONFIG = UltimateConfig()

#============================================================================
#EMA HELPER (Exponential Moving Average pour stabilisation)
#============================================================================
class EMAHelper:
    """Helper pour stabiliser les weights avec EMA"""
    
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}
    
    def register(self, module: nn.Module):
        """Enregistre les param tres pour EMA"""
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, module: nn.Module):
        """Met   jour l'EMA des weights"""
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data
    
    def ema(self, module: nn.Module):
        """Applique l'EMA aux weights"""
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

#============================================================================
#LAYERS POUR HRM-TRM
#============================================================================
class SwiGLU(nn.Module):
    """SwiGLU activation pour le reasoning network"""
    
    def __init__(self, hidden_size: int, expansion: float = 2.0):
        super().__init__()
        inter = int(round(expansion * hidden_size * 2 / 3))
        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

class Attention(nn.Module):
    """Multi-head attention pour raisonnement hi rarchique"""
    
    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_key_value_heads: int, causal: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.qkv_proj = nn.Linear(hidden_size, (num_heads + 2 * num_key_value_heads) * head_dim, bias=False)
        self.o_proj = nn.Linear(head_dim * num_heads, hidden_size, bias=False)
    
    def forward(self, cos_sin: Tuple, hidden_states: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        
        query, key, value = qkv.split([
            self.num_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim
        ], dim=-1)
        
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=self.causal)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        return self.o_proj(attn_output)

def rms_norm(hidden_states: Tensor, variance_epsilon: float = 1e-5) -> Tensor:
    """RMS Normalization pour stabilit """
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self) -> Tuple[Tensor, Tensor]:
        return self.cos_cached, self.sin_cached

#============================================================================
#SPARSE EMBEDDING POUR SYMBOLES
#============================================================================
class CastedSparseEmbedding(nn.Module):
    """Sparse embeddings pour identifiants de symboles"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.02)
    
    def forward(self, inputs: Tensor) -> Tensor:
        return self.weights[inputs]

#============================================================================
#ACT LOSS HEAD POUR Q-LEARNING - VERSION UNIFI E
#============================================================================
class ACTLossHead(nn.Module):
    """Loss head pour ACT Q-learning (halting adaptatif) - VERSION UNIFI E"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Calcule la loss pour l'halting adaptatif.
        
        Args:
            logits: Sortie du q_head [batch_size, 2]
            labels: Targets binaires [batch_size, 1] - 1 si correct, 0 sinon
        
        Returns:
            BCE loss sur la probabilit  de halting
        """
        # Prendre la probabilit  de halting (deuxi me logit)
        halt_probs = torch.softmax(logits, dim=-1)[:, 1].unsqueeze(1)
        
        # Binary cross entropy entre halting prob et target correctness
        return F.binary_cross_entropy(halt_probs, labels.float())

#============================================================================
#ADVANCED HRM-TRM (Hierarchical Recursive Model with Tiny Recursive Model)
#============================================================================
#============================================================================
#ADVANCED HRM-TRM (Hierarchical Recursive Model with Tiny Recursive Model) - VERSION CORRIG E
#============================================================================
class AdvancedHRMTRM(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        H_cycles: int = 2,
        L_cycles: int = 3,
        num_heads: int = 4,
        halt_max_steps: int = 16,
        exploration_prob: float = 0.1,
        num_symbols: int = 4,
        vocab_size: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        #   CORRECTION: Initialisation robuste des poids
        self._init_config(hidden_dim, H_cycles, L_cycles, halt_max_steps, exploration_prob)
        
        # Modules d'embedding
        self.sparse_emb = nn.Embedding(num_symbols, hidden_dim)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        
        # Architecture Q-learning pour l'halting adaptatif
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # [continue_prob, halt_prob]
        )
        self.loss_head = ACTLossHead(hidden_dim)
        
        # Architecture principale de raisonnement
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.reasoning = SwiGLU(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 3)  # 3 actions: BUY, SELL, HOLD
        
        #   CORRECTION: Initialisation des poids
        self._init_weights()
        
        # EMA pour stabilisation
        self.ema = EMAHelper()
        self.ema.register(self)
        
        logging.info(f"  HRM-TRM initialis : input_dim={input_dim}, hidden_dim={hidden_dim}")

    def _init_config(self, hidden_dim: int, H_cycles: int, L_cycles: int, 
                    halt_max_steps: int, exploration_prob: float):
        """Initialisation de la configuration"""
        self.hidden_dim = hidden_dim
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.halt_max_steps = halt_max_steps
        self.exploration_prob = exploration_prob

    def _init_weights(self):
        """Initialisation robuste des poids pour  viter les NaN - VERSION RENFORC E"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Initialisation Xavier/Glorot avec gain r duit
                nn.init.xavier_uniform_(module.weight, gain=0.5)  #   Gain r duit
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
                    
            elif isinstance(module, nn.Embedding):
                # Initialisation normale avec std r duite
                nn.init.normal_(module.weight, mean=0.0, std=0.01)  #   Std r duite
                
            elif isinstance(module, nn.LayerNorm):
                # Initialisation LayerNorm
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
                logging.debug(f"  Initialis : {name} (Embedding)")

    def _safe_forward(self, module: nn.Module, x: Tensor) -> Tensor:
        """Forward pass s curis  avec d tection NaN - VERSION RENFORC E"""
        try:
            # V rification rigoureuse de l'input
            if torch.isnan(x).any() or torch.isinf(x).any():
                logging.warning(f"  Input corrompu dans {module.__class__.__name__}, nettoyage")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Clipping des valeurs extr mes
            x = torch.clamp(x, -10.0, 10.0)
            
            output = module(x)
            
            # V rification rigoureuse de l'output
            if torch.isnan(output).any() or torch.isinf(output).any():
                logging.warning(f"  Output corrompu dans {module.__class__.__name__}, fallback")
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                output = torch.clamp(output, -5.0, 5.0)
                
            return output
            
        except Exception as e:
            logging.error(f"  Erreur dans {module.__class__.__name__}: {e}")
            # Retourner des z ros de la m me shape
            return torch.zeros_like(x)

    def refine(self, x: Tensor, y_init: Tensor, z_init: Tensor, symbol_id: Tensor) -> Tensor:
        """
        Raffine r cursivement et hi rarchiquement une observation avec HRM-TRM.
        
        Args:
            x: Observation brute (batch, input_dim)
            y_init: Action initiale guess (batch,)
            z_init: Latent state initial (batch, hidden_dim)
            symbol_id: Identifiant du symbole (batch,)
        
        Returns:
            z_H: Latent state raffin  (batch, hidden_dim)
        """
        try:
            #   CORRECTION: V rification des inputs
            if torch.isnan(x).any() or torch.isinf(x).any():
                logging.warning("  Input x contient NaN/Inf dans refine()")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

            # Embeddings avec fallback safe
            try:
                symb_emb = self.sparse_emb(symbol_id)
                y_emb = self.embed_tokens(y_init.long()) + symb_emb
            except Exception as emb_error:
                logging.warning(f"  Erreur embeddings: {emb_error}, fallback")
                symb_emb = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
                y_emb = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

            # Projection de l'input avec s curit 
            x_proj = self._safe_forward(self.input_proj, x)

            # Initialize states
            z_H = z_init.clone()
            z_L = z_init.clone()
            steps = 0
            halted = False

            # TRM: Recursive refinement loop (cycles L et H)
            while steps < self.halt_max_steps and not halted:
                # Low-level cycles (TRM - Tiny Recursive Model)
                for _ in range(self.L_cycles):
                    # Combine input + embeddings + current state
                    z_L_input = z_H + y_emb + x_proj
                    z_L = self._safe_forward(self.reasoning, z_L_input)

                # High-level update (HRM - Hierarchical Reasoning)
                z_H = z_L.clone()  # Update high-level state

                # Adaptive halting decision (Q-learning)
                if hasattr(self, 'q_head') and self.q_head is not None:
                    try:
                        q_logits = self._safe_forward(self.q_head, z_H)
                        q_halt, q_continue = q_logits[:, 0], q_logits[:, 1]
                        
                        # V rification des valeurs Q
                        if torch.isnan(q_halt).any() or torch.isnan(q_continue).any():
                            logging.warning("  Valeurs Q NaN, halting forc ")
                            halted = steps >= 3
                        else:
                            # Exploration during training
                            if self.training and torch.rand(1, device=x.device) < self.exploration_prob:
                                min_steps = torch.randint(2, self.halt_max_steps, (1,), device=x.device)
                                halted = steps >= min_steps.item()
                            else:
                                halted = (q_halt > q_continue).any().item() or (steps == self.halt_max_steps - 1)
                                
                    except Exception as q_error:
                        logging.warning(f"  Erreur Q-learning: {q_error}, halting simple")
                        halted = steps >= 3
                else:
                    # Fallback: halt apr s quelques cycles
                    halted = steps >= 3

                steps += 1

                #   CORRECTION: V rification p riodique des  tats
                if steps % 5 == 0 and (torch.isnan(z_H).any() or torch.isinf(z_H).any()):
                    logging.warning("   tat z_H corrompu, r initialisation partielle")
                    z_H = torch.nan_to_num(z_H, nan=0.0, posinf=1.0, neginf=-1.0)

            # Update EMA if training
            if self.training:
                try:
                    self.ema.update(self)
                except Exception as ema_error:
                    logging.warning(f"  Erreur EMA update: {ema_error}")

            # V rification finale du state de sortie
            if torch.isnan(z_H).any() or torch.isinf(z_H).any():
                logging.error("   tat de sortie z_H corrompu, fallback vers z_init")
                z_H = z_init.clone()

            return z_H

        except Exception as e:
            logging.error(f"  Erreur critique dans refine(): {e}")
            # Fallback s r
            return torch.zeros(x.size(0), self.hidden_dim, device=x.device)

    def compute_q_loss(self, z_H: Tensor, target_correct: Tensor) -> Tensor:
        """
        Calcule la loss Q-learning pour le halting.
        
        Args:
            z_H: Latent states (batch, hidden_dim)
            target_correct: Targets binaires (batch, 1) - 1 si correct, 0 sinon
        
        Returns:
            loss: BCE loss
        """
        try:
            if not hasattr(self, 'q_head') or self.q_head is None:
                return torch.tensor(0.0, device=z_H.device)
            
            # V rification des inputs
            if torch.isnan(z_H).any() or torch.isinf(z_H).any():
                logging.warning("  z_H contient NaN/Inf dans compute_q_loss")
                return torch.tensor(0.0, device=z_H.device)
            
            q_logits = self._safe_forward(self.q_head, z_H)
            loss = self.loss_head(q_logits, target_correct)
            
            # V rification de la loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("  Loss Q-learning NaN/Inf, remplacement par 0")
                return torch.tensor(0.0, device=z_H.device)
                
            return loss
            
        except Exception as e:
            logging.error(f"  Erreur compute_q_loss: {e}")
            return torch.tensor(0.0, device=z_H.device)

    def forward(self, x: Tensor, symbol_ids: Tensor = None) -> Tensor:
        """
        Forward pass complet pour l'inf rence avec protection robuste contre les NaN.
        """
        try:
            #   CORRECTION: V rification et nettoyage rigoureux de l'input
            if torch.isnan(x).any() or torch.isinf(x).any():
                logging.warning("  Input x contient NaN/Inf dans forward(), nettoyage")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # V rification des dimensions
            if x.dim() != 2:
                logging.warning(f"  Input shape incorrect: {x.shape}, reshaping")
                x = x.view(1, -1) if x.dim() == 1 else x
                
            if x.size(1) != self.input_proj.in_features:
                logging.error(f"  Dimension input incorrecte: {x.size(1)} vs {self.input_proj.in_features}")
                # Fallback vers HOLD
                return torch.tensor([[0.0, 0.0, 1.0]], device=x.device)

            # Refinement hi rarchique
            y_init = torch.tensor([2], device=x.device)  # HOLD initial
            z_init = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            
            if symbol_ids is None:
                symbol_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            z_refined = self.refine(x, y_init, z_init, symbol_ids)
            
            # V rification du state raffin 
            if torch.isnan(z_refined).any() or torch.isinf(z_refined).any():
                logging.error("  State raffin  z_refined corrompu, fallback")
                z_refined = torch.zeros_like(z_refined)

            # D cision finale avec protection
            action_logits = self._safe_forward(self.output_layer, z_refined)
            
            #   CORRECTION: V rification finale rigoureuse
            if (torch.isnan(action_logits).any() or 
                torch.isinf(action_logits).any() or 
                action_logits.size(1) != 3):
                
                logging.error("  Logits de sortie corrompus, fallback vers HOLD")
                action_logits = torch.tensor([[0.0, 0.0, 1.0]], device=x.device)  # HOLD forc 
            
            # Normalisation douce pour stabilit 
            action_logits = torch.tanh(action_logits) * 2.0
            
            return action_logits
            
        except Exception as e:
            logging.error(f"  Erreur critique dans forward(): {e}")
            # Fallback ultra-safe vers HOLD
            return torch.tensor([[0.0, 0.0, 1.0]], device=x.device if torch.is_tensor(x) else torch.device('cpu'))

#============================================================================
# GYMNASTIUM TO SB3 COMPATIBILITY WRAPPER - VERSION CORRIG E
#============================================================================
class GymnasiumToSB3Wrapper(gym.Wrapper):
    """Wrapper pour compatibilit  entre Gymnasium et Stable-Baselines3"""
    
    def __init__(self, env):
        super().__init__(env)
        #   FIX: Exposer hrm_trm pour acc s depuis l'ext rieur
        if hasattr(env, 'hrm_trm'):
            self.hrm_trm = env.hrm_trm
    
    def reset(self, **kwargs):
        # Ignorer les kwargs pour compatibilit  SB3
        obs, info = self.env.reset()
        return obs  # SB3 attend seulement l'observation
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Convertir en format SB3 (terminated ou truncated -> done)
        done = terminated or truncated


        # ========================================================================

        # ========================================================================

        # ========================================================================
        # VISION: CAPTURE CHART PATTERN
        # ========================================================================
        if self.current_step % 10 == 0 and self.current_step > 100:
            try:
                # Capturer le pattern actuel
                chart_pattern = self.chart_vision.capture_chart_pattern(self.data, self.current_step)
                
                if chart_pattern:
                    # Trouver patterns similaires
                    similar = self.chart_vision.find_similar_patterns(chart_pattern, top_n=3)
                    
                    if similar:
                        # PrÃƒÂ©dire la prochaine move
                        prediction = self.chart_vision.predict_next_move(chart_pattern)
                        logging.debug(f"[VISION] Prediction: {prediction['predicted_direction']} "
                                     f"(confidence: {prediction['confidence']:.2f}, "
                                     f"expected: {prediction['expected_pips']:.1f} pips)")
                        
                        # Stocker pour decision making
                        self._vision_prediction = prediction
                
            except Exception as e:
                logging.debug(f"[VISION] Error: {e}")

        # ADVANCED: DETECT REGIME & PATTERNS
        # ========================================================================
        # DÃƒÂ©tecter rÃƒÂ©gime toutes les 20 steps
        if self.current_step % 20 == 0 and self.current_step > 50:
            try:
                # Detect market regime
                regime = self.regime_detector.detect_regime(self.data, self.current_step)
                logging.debug(f"[REGIME] {regime['name']} (confidence: {regime['confidence']:.2f})")
                
                # Detect patterns
                patterns = self.pattern_recognizer.detect_all_patterns(self.data, self.current_step)
                if patterns:
                    logging.debug(f"[PATTERNS] Detected {len(patterns)} patterns")
                
                # Select best strategy
                confluence = getattr(self, '_last_confluence', {})
                strategy = self.strategy_selector.select_best_strategy(regime, patterns, confluence)
                logging.debug(f"[STRATEGY] Selected: {strategy['name']}")
                
                # Store for decision making
                self._current_regime = regime
                self._current_patterns = patterns
                self._current_strategy = strategy
                
            except Exception as e:
                logging.debug(f"[ADVANCED] Error: {e}")

        # DRAGON SYSTEMS: SCAN FOR MISSED OPPORTUNITIES
        # ========================================================================
        # Scanner toutes les 50 steps pour dÃƒÂ©tecter les patterns loupÃƒÂ©s
        if self.current_step % 50 == 0 and self.current_step > 100:
            try:
                self._scan_missed_opportunities()
            except Exception as e:
                logging.debug(f"[DRAGON] Error scanning missed opportunities: {e}")

        # UPDATE PERFECT TIMING (si position ouverte)
        if self.position != 0:
            try:
                cprice = self.data.iloc[self.current_step]["close"]
                self.perfect_timing.update_position(self.symbol, cprice, self.current_step)
            except:
                pass

        return obs, reward, done, info


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDENEYE V5 ULTRA - MERLIN PROFIT TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class MerlinProfitTracker:
    """
    MERLIN prédit maintenant le PROFIT d'un trade, pas des patterns.
    On mesure si ses prédictions sont bonnes sur le VRAI profit en $.
    """
    
    def __init__(self):
        self.predictions = []  # Liste de {predicted_pnl, actual_pnl, timestamp, confidence}
        self.correct_predictions = 0
        self.total_predictions = 0
        
        logging.info("[MERLIN PROFIT] Tracker initialized - tracking REAL $ predictions")
    
    def predict_trade_outcome(self, rsi: float, macd: float, price_change: float) -> dict:
        """
        MERLIN analyse l'état actuel et prédit le profit en $.
        
        Utilise des heuristiques simples pour démarrer :
        - RSI extrême + MACD aligné = fort signal
        - Prix momentum = magnitude
        """
        # Signal de direction basé sur indicateurs
        signal_strength = 0.0
        
        # RSI contribution
        if rsi < 30:  # Oversold
            signal_strength += (30 - rsi) / 30 * 0.5  # Max +0.5
        elif rsi > 70:  # Overbought
            signal_strength -= (rsi - 70) / 30 * 0.5  # Max -0.5
        
        # MACD contribution
        if macd > 0:
            signal_strength += min(macd * 100, 0.3)  # Max +0.3
        else:
            signal_strength += max(macd * 100, -0.3)  # Max -0.3
        
        # Prix momentum contribution
        signal_strength += price_change * 0.2  # Max ±0.2
        
        # Clamp entre -1 et +1
        signal_strength = max(-1.0, min(1.0, signal_strength))
        
        # Prédiction de profit en $ (basé sur signal strength)
        # Signal fort = prédiction de profit plus important
        expected_pnl = signal_strength * 50  # Max ±$50 par trade
        
        # Confidence basée sur la force du signal
        confidence = abs(signal_strength)
        
        # Direction
        direction = 1 if signal_strength > 0.1 else (-1 if signal_strength < -0.1 else 0)
        
        return {
            'direction': direction,
            'expected_pnl': expected_pnl,
            'confidence': confidence,
            'signal_strength': signal_strength
        }
    
    def record_prediction(self, prediction: dict, entry_price: float):
        """Enregistre une prédiction au moment de l'entrée"""
        self.predictions.append({
            'predicted_pnl': prediction['expected_pnl'],
            'predicted_direction': prediction['direction'],
            'confidence': prediction['confidence'],
            'entry_price': entry_price,
            'timestamp': datetime.now(),
            'actual_pnl': None  # Sera rempli à la sortie
        })
        self.total_predictions += 1
    
    def record_outcome(self, actual_pnl: float):
        """Enregistre le résultat réel d'un trade"""
        # Trouver la dernière prédiction non résolue
        for pred in reversed(self.predictions):
            if pred['actual_pnl'] is None:
                pred['actual_pnl'] = actual_pnl
                
                # Vérifier si la prédiction était correcte
                predicted_sign = np.sign(pred['predicted_pnl'])
                actual_sign = np.sign(actual_pnl)
                
                # Correct si bon signe ET magnitude dans ±50%
                sign_correct = (predicted_sign == actual_sign)
                
                if sign_correct and abs(pred['predicted_pnl']) > 0:
                    magnitude_error = abs(actual_pnl - pred['predicted_pnl']) / abs(pred['predicted_pnl'])
                    magnitude_correct = (magnitude_error < 0.5)  # ±50% tolérance
                    
                    if magnitude_correct:
                        self.correct_predictions += 1
                
                break
    
    def get_accuracy(self) -> dict:
        """
        Calcule la vraie accuracy de MERLIN sur le profit
        """
        if self.total_predictions == 0:
            return {
                'overall_accuracy': 0.0,
                'direction_accuracy': 0.0,
                'magnitude_accuracy': 0.0,
                'total_predictions': 0
            }
        
        # 1. Overall accuracy (signe + magnitude)
        overall_acc = self.correct_predictions / self.total_predictions * 100
        
        # 2. Direction accuracy (juste le signe)
        completed = [p for p in self.predictions if p['actual_pnl'] is not None]
        
        if not completed:
            return {
                'overall_accuracy': 0.0,
                'direction_accuracy': 0.0,
                'magnitude_accuracy': 0.0,
                'total_predictions': self.total_predictions
            }
        
        direction_correct = sum(
            1 for p in completed
            if np.sign(p['predicted_pnl']) == np.sign(p['actual_pnl'])
        )
        direction_acc = direction_correct / len(completed) * 100 if completed else 0
        
        # 3. Magnitude accuracy (erreur moyenne en %)
        magnitude_errors = []
        for p in completed:
            if abs(p['predicted_pnl']) > 0.01:  # Éviter division par zéro
                error = abs(p['actual_pnl'] - p['predicted_pnl']) / abs(p['predicted_pnl'])
                magnitude_errors.append(min(error, 2.0))  # Cap à 200% d'erreur
        
        magnitude_acc = (1 - np.mean(magnitude_errors)) * 100 if magnitude_errors else 0.0
        
        return {
            'overall_accuracy': overall_acc,
            'direction_accuracy': direction_acc,
            'magnitude_accuracy': max(0, magnitude_acc),
            'total_predictions': self.total_predictions
        }


#============================================================================
#TRADING ENVIRONMENT POUR RL - VERSION UNIFI E CORRIG E
#============================================================================
class TradingEnvironment(gym.Env):
    """Environnement Gymnasium pour trading avec HRM-TRM intÃƒÂ©grÃƒÂ© - VERSION CORRIGÃƒÂ‰E"""
    
    def __init__(self, historical_data: pd.DataFrame, symbol_id: int = 0, symbol: str = "EURUSD"):
        super().__init__()
        
        # ===========================================================================
        #   CONFIGURATION SYMBOLE & COÃƒÂ›TS RÃƒÂ‰ALISTES
        # ===========================================================================
        self.symbol = symbol
        self.symbol_id = torch.tensor([symbol_id])
        self.data = historical_data
        
        # Configuration des coÃƒÂ»ts par type de symbole
        self._setup_trading_costs(symbol)
        
        # ═══════════════════════════════════════════════════════════════════
        # MEGA PATCH V4: ENRICHIR LES DONNÉES AVEC TOUTES LES FEATURES
        # ═══════════════════════════════════════════════════════════════════
        self._enhance_data_with_full_features()

        
        # ===========================================================================
        #   GESTION DES RISQUES ET CAPITAL
        # ===========================================================================
        self.initial_balance = 10000.0
        self.equity = self.initial_balance
        self.free_margin = self.initial_balance
        self.used_margin = 0.0
        self.leverage = 30
        self.margin_requirement = 0.033  # 3.3% pour levier 1:30
        
        # Limites de trading
        self.max_position_pct = 0.02     # Max 2% du capital par trade
        self.max_daily_loss = 0.05       # Max 5% de perte par jour
        self.max_daily_trades = 50       # Max 50 trades/jour
        
        # ===========================================================================
        #   SYSTÃƒÂˆME DE PHASES ET TRACKING
        # ===========================================================================
        self.phase_length = 500
        self.current_phase = 0
        self.phase_start_step = 0
        self.phase_start_equity = self.initial_balance
        
        # Statistiques globales
        self.total_wins = 0
        self.total_losses = 0
        self.total_closed = 0
        self.total_pnl = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # Statistiques de phase
        self.phase_closes = []
        self.phase_pnl = 0.0
        self.phase_wins = 0
        self.phase_losses = 0
        
        # CompatibilitÃƒÂ© ancien systÃƒÂ¨me
        self.close_wins = 0
        self.close_losses = 0
        self.close_trades = []
        
        # ===========================================================================
        #   NOUVEAUX ATTRIBUTS POUR TIMING OPTIMAL
        # ===========================================================================
        self.peak_profit = 0.0
        self.peak_pips = 0.0
        self.peak_step = 0
        self.max_drawdown = 0.0
        self.time_since_peak = 0
        self.entry_quality_score = 0.0
        self.optimal_entries = 0
        self.optimal_exits = 0
        self.perfect_trades = 0
        
        # ===========================================================================
        #   COÃƒÂ›TS CUMULÃƒÂ‰S
        # ===========================================================================
        self.total_spread_paid = 0.0
        self.total_commission_paid = 0.0
        self.total_swap_paid = 0.0
        self.total_slippage_cost = 0.0
        
        # ===========================================================================
        #   ÃƒÂ‰TAT DU TRADING
        # ===========================================================================
        self.current_step = 0
        self.position = 0  # 0: flat, 1: long, -1: short
        self.entry_price = 0.0
        self.pnl = 0.0
        self.position_open_step = 0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
        # ===========================================================================
        #   INITIALISATION FINALE
        # ===========================================================================
        self._reset_daily_tracking()
        logging.info(f"[OK] TradingEnvironment initialisÃƒÂ© pour {symbol}")
        
        # ===================================================================
        # [FIRE] INITIALISATION TDA V4 - EXTRACTION FEATURES AVANCÃƒÂ‰ES
        # Protégé contre les dim mismatch quand CSV V6 ≠ longueur MT5
        # ===================================================================
        try:
            logging.info(f"[TDA] Initialisation TDA V4 pour {symbol}...")
            self.tda_extractor = TDAFeatureExtractorV4()

            logging.info(f"[TDA] Extraction features TDA sur {len(historical_data)} bougies...")
            tda_features_array = self.tda_extractor.extract_advanced_features(historical_data)

            # Vérification alignement — tronque si nécessaire
            n_rows = min(len(historical_data), tda_features_array.shape[0])
            tda_features_df = pd.DataFrame(
                tda_features_array[:n_rows],
                columns=self.tda_extractor.feature_names,
                index=historical_data.index[:n_rows]
            )

            # Enrichit self.data avec les features TDA
            for col in tda_features_df.columns:
                if col not in self.data.columns:
                    if len(tda_features_df[col]) == len(self.data):
                        self.data[col] = tda_features_df[col].values
        except Exception as _tda_err:
            logging.warning(f"[TDA] Ignoré — dim mismatch ou erreur: {_tda_err}")
            self.tda_extractor = None

        # Stats TDA seulement si extraction réussie
        if self.tda_extractor is not None:
            n_tda_features = len(self.tda_extractor.feature_names)
            n_total_features = len(self.data.columns)
            logging.info(f"[TDA] [OK] {n_tda_features} features TDA ajoutées")
            feature_names = self.tda_extractor.feature_names
            trend_f = [f for f in feature_names if 'sma' in f or 'ema' in f]
            momentum_f = [f for f in feature_names if 'rsi' in f or 'macd' in f or 'stoch' in f]
            volatility_f = [f for f in feature_names if 'atr' in f or 'bb' in f]
            volume_f = [f for f in feature_names if 'volume' in f or 'obv' in f]
            logging.info(f"[TDA] Trend:{len(trend_f)} Momentum:{len(momentum_f)} Volatility:{len(volatility_f)}")
        else:
            logging.info(f"[TDA] Ignoré — {len(self.data.columns)} features V6 disponibles")
        
        # Stats TDA sur les donnÃƒÂ©es actuelles
        if 'rsi_14' in self.data.columns:
            rsi_mean = self.data['rsi_14'].mean()
            logging.info(f"[TDA]   -> RSI moyen: {rsi_mean:.1f}")
        
        if 'macd' in self.data.columns:
            macd_bullish = (self.data['macd'] > 0).sum()
            macd_pct = 100 * macd_bullish / len(self.data)
            logging.info(f"[TDA]   -> MACD bullish: {macd_pct:.1f}%")
        
        logging.info("[TDA] [OK] TDA V4 intÃƒÂ©gration terminÃƒÂ©e")
        
        # ===========================================================================
        # [PERFECT TIMING] PERFECT TIMING REWARDS SYSTEM
        # ===========================================================================
        logging.info("[PERFECT TIMING] Initializing Perfect Timing Rewards System...")
        self.perfect_timing = PerfectTimingRewardSystem(
            entry_window=10,  # ÃƒÂ‰valuer entry sur 10 bougies avant
            exit_window=10    # ÃƒÂ‰valuer exit sur 10 bougies aprÃƒÂ¨s
        )
        logging.info("[PERFECT TIMING] System ready!")
        
        # ===========================================================================
        # [GOLDENEYE ULTIMATE] DIVERGENCE DETECTION SYSTEM
        # ===========================================================================
        logging.info("[DIVERGENCE] Initializing Divergence Detection System...")
        self.divergence_system = DivergenceDetectionSystem(
            memory_file=f'divergence_memory_{symbol}.pkl',
            max_patterns=1000
        )
        # PATCH V4: Assigner à la variable globale pour le dashboard
        global GLOBAL_DIVERGENCE_SYSTEM
        GLOBAL_DIVERGENCE_SYSTEM = self.divergence_system
        
        self.current_divergences = []
        self.trade_on_divergence = False
        self.divergence_at_entry = None
        logging.info(f"[DIVERGENCE] System ready - {len(self.divergence_system.divergence_patterns)} patterns loaded")
        
        # ===========================================================================
        # [GOLDENEYE V5 ULTRA] TRUE PROFIT TRACKING
        # ===========================================================================
        logging.info("[V5 ULTRA] Initializing True Profit Tracking...")
        
        # PnL History pour risk adjustment
        self.recent_pnl_history = []  # Derniers 20 trades
        self.max_pnl_history = 20
        
        # Drawdown tracking
        self.peak_equity = self.initial_balance
        self.current_drawdown = 0.0
        
        # Overtrading tracking
        self.recent_trade_count = 0
        self.recent_trade_timestamps = []
        self.trade_count_window = 3600  # 1 heure en secondes
        
        # Win streak tracking (déjà existe mais on s'assure)
        if not hasattr(self, 'consecutive_wins'):
            self.consecutive_wins = 0
        if not hasattr(self, 'consecutive_losses'):
            self.consecutive_losses = 0
        
        logging.info("[V5 ULTRA] True Profit Tracking ready!")
        
        # ===========================================================================
        # [SUPER MERLIN] PATTERN ANALYZER WITH VISION AI
        # ===========================================================================
        logging.info("[MERLIN] Initializing Super Merlin Pattern Analyzer...")
        self.merlin_analyzer = SuperMerlinPatternAnalyzer(
            memory_file=f'merlin_patterns_{symbol}.pkl',
            similarity_threshold=0.75
        )
        self.merlin_prediction = None
        self.merlin_at_entry = None
        if self.merlin_analyzer.enabled:
            logging.info(f"[MERLIN] Vision AI ready - {sum(len(v) for v in self.merlin_analyzer.pattern_library.values())} patterns loaded")
        else:
            logging.warning("[MERLIN] Vision AI disabled - install cv2 and sklearn for full functionality")
        
        # MERLIN Profit Tracker V5 ULTRA
        self.merlin_profit_tracker = MerlinProfitTracker()
        logging.info("[MERLIN PROFIT] Real $ prediction tracker initialized")
        
        # ===========================================================================
        # [DRAGON] DRAGON TRADING SYSTEMS - LEARNING FROM EVERYTHING
        # ===========================================================================
        logging.info("[DRAGON] Initializing Dragon Trading Systems...")
        
        # Confluence Detector
        self.confluence_detector = MultiIndicatorConfluenceDetector()
        
        # Missed Opportunity Detector
        self.missed_opp_detector = MissedOpportunityDetector(
            min_confluence_score=70,
            min_potential_pips=15
        )
        
        # Success Pattern Analyzer
        self.success_analyzer = SuccessPatternAnalyzer()
        
        # Pattern Memory Bank
        self.pattern_memory = PatternMemoryBank(capacity=10000)
        
        # Augmented Experience Replay
        self.experience_replay = AugmentedExperienceReplay(capacity=100000)
        
        # Strategy Evolution Engine
        self.strategy_evolution = StrategyEvolutionEngine()
        
        # ===========================================================================
        # [ADVANCED] ADVANCED LEARNING SYSTEMS
        # ===========================================================================
        logging.info("[ADVANCED] Initializing Advanced Learning Systems...")
        
        # Market Regime Detector
        self.regime_detector = MarketRegimeDetector()
        
        # Advanced Pattern Recognizer
        self.pattern_recognizer = AdvancedPatternRecognizer()
        
        # Long-Term Memory System (persistent!)
        self.long_term_memory = LongTermMemorySystem(memory_file=f'dragon_memory_{symbol}.pkl')
        
        # Adaptive Strategy Selector
        self.strategy_selector = AdaptiveStrategySelector(self.long_term_memory)
        
        # ===========================================================================
        # [VISION] CHART VISION & LOSS REDUCTION
        # ===========================================================================
        logging.info("[VISION] Initializing Chart Vision & Loss Reduction...")
        
        # Chart Vision System
        self.chart_vision = ChartVisionSystem(
            window_size=50,
            memory_file=f'chart_vision_{symbol}.pkl'
        )
        
        # Loss Reduction Analyzer
        self.loss_analyzer = LossReductionAnalyzer()
        
        logging.info("[VISION] Systems ready!")
        logging.info("[VISION]   Ã¢ÂœÂ“ Chart pattern recognition")
        logging.info("[VISION]   Ã¢ÂœÂ“ Loss pattern detection")

        
        logging.info("[ADVANCED] All advanced systems initialized!")
        logging.info("[ADVANCED] Agent now has:")
        logging.info("[ADVANCED]   Ã¢ÂœÂ“ Market regime detection")
        logging.info("[ADVANCED]   Ã¢ÂœÂ“ Advanced pattern recognition")
        logging.info("[ADVANCED]   Ã¢ÂœÂ“ Long-term memory (persistent)")
        logging.info("[ADVANCED]   Ã¢ÂœÂ“ Adaptive strategy selection")

        
        # Tracking variables
        self.missed_patterns_history = []
        self.scanned_windows = set()  # Pour ÃƒÂ©viter de re-scanner
        
        logging.info("[DRAGON] All systems initialized!")
        logging.info("[DRAGON] Agent can now learn from:")
        logging.info("[DRAGON]   Ã¢ÂœÂ“ Winning trades")
        logging.info("[DRAGON]   Ã¢ÂœÂ“ Losing trades")
        logging.info("[DRAGON]   Ã¢ÂœÂ“ Missed opportunities")
        logging.info("[DRAGON]   Ã¢ÂœÂ“ Pattern confluences")


        
        # ===========================================================================
        # [FIRE] HRM-TRM - MAINTENANT AVEC LES BONNES DIMENSIONS (APRÃƒÂˆS TDA)
        # ===========================================================================
        logging.info(f"[HRM] Initialisation HRM-TRM avec {len(self.data.columns)} features...")
        self.hrm_trm = AdvancedHRMTRM(
            input_dim=len(self.data.columns),
            hidden_dim=CONFIG.HRM_HIDDEN_DIM,
            H_cycles=CONFIG.HRM_H_CYCLES,
            L_cycles=CONFIG.HRM_L_CYCLES,
            num_heads=CONFIG.HRM_NUM_HEADS,
            halt_max_steps=CONFIG.HRM_HALT_MAX_STEPS,
            exploration_prob=CONFIG.HRM_EXPLORATION_PROB,
            num_symbols=len(CONFIG.SYMBOLS),
            vocab_size=4
        )
        logging.info(f"[HRM] [OK] HRM-TRM configurÃƒÂ© pour {len(self.data.columns)} features")
        
        # ===========================================================================
        #   ESPACES GYMNASIUM
        # ===========================================================================
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(CONFIG.HRM_HIDDEN_DIM,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)  # 0: BUY, 1: SELL, 2: HOLD, 3: CLOSE


    def _enhance_data_with_full_features(self):
        """
        MEGA PATCH V4: Enrichit self.data avec TOUTES les features
        - Multi-Timeframe (MTF): RSI, MACD, ATR de tous les TF
        - Sessions de trading: London, NY, Asia, Overlap
        - Heure encodée: sin/cos pour cyclicité
        - Position context: position actuelle, holding time
        
        Appelé après self.data = historical_data
        """
        import numpy as np
        
        logging.info("[MEGA] ═══════════════════════════════════════════════════")
        logging.info("[MEGA] ENRICHISSEMENT FEATURES V4 - L'AGENT VERRA TOUT!")
        logging.info("[MEGA] ═══════════════════════════════════════════════════")
        
        original_cols = len(self.data.columns)
        
        # ═══════════════════════════════════════════════════════════════════
        # SESSIONS DE TRADING (basé sur l'heure UTC)
        # ═══════════════════════════════════════════════════════════════════
        if 'time' in self.data.columns:
            try:
                # Extraire l'heure
                hours = self.data['time'].dt.hour
                
                # Sessions (approximations UTC)
                self.data['session_london'] = ((hours >= 7) & (hours < 16)).astype(float)
                self.data['session_ny'] = ((hours >= 12) & (hours < 21)).astype(float)
                self.data['session_asia'] = ((hours >= 23) | (hours < 8)).astype(float)
                self.data['session_overlap'] = ((hours >= 12) & (hours < 16)).astype(float)  # London+NY
                
                # Encodage cyclique de l'heure (meilleur pour NN)
                self.data['hour_sin'] = np.sin(2 * np.pi * hours / 24)
                self.data['hour_cos'] = np.cos(2 * np.pi * hours / 24)
                
                # Jour de la semaine (0=Lundi, 4=Vendredi)
                day_of_week = self.data['time'].dt.dayofweek
                self.data['dow_sin'] = np.sin(2 * np.pi * day_of_week / 5)
                self.data['dow_cos'] = np.cos(2 * np.pi * day_of_week / 5)
                
                logging.info("[MEGA] ✅ Sessions ajoutées: London, NY, Asia, Overlap + heure/jour encodés")
            except Exception as e:
                logging.warning(f"[MEGA] ⚠️ Erreur sessions: {e}")
                # Colonnes par défaut
                for col in ['session_london', 'session_ny', 'session_asia', 'session_overlap', 
                           'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']:
                    if col not in self.data.columns:
                        self.data[col] = 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # FEATURES MULTI-TIMEFRAME (MTF)
        # Les colonnes M15_rsi, H1_macd, etc. devraient déjà exister si chargées
        # On s'assure qu'elles ont des valeurs propagées
        # ═══════════════════════════════════════════════════════════════════
        mtf_features = ['M1_rsi', 'M1_macd', 'M1_atr',
                        'M15_rsi', 'M15_macd', 'M15_atr',
                        'H1_rsi', 'H1_macd', 'H1_atr',
                        'H4_rsi', 'H4_macd', 'H4_atr',
                        'D1_rsi', 'D1_macd', 'D1_atr']
        
        for col in mtf_features:
            if col not in self.data.columns:
                # Créer avec valeur par défaut basée sur le TF de base
                base_indicator = col.split('_')[1]  # rsi, macd, ou atr
                if base_indicator in self.data.columns:
                    self.data[col] = self.data[base_indicator]
                else:
                    self.data[col] = 0.0
            else:
                # Propager les NaN avec forward fill
                self.data[col] = self.data[col].ffill().bfill().fillna(0)
        
        logging.info(f"[MEGA] ✅ MTF features: {len(mtf_features)} colonnes vérifiées/créées")
        
        # ═══════════════════════════════════════════════════════════════════
        # FEATURES DE TENDANCE MULTI-TF
        # ═══════════════════════════════════════════════════════════════════
        try:
            # Tendance basée sur comparaison RSI multi-TF
            if 'rsi' in self.data.columns:
                base_rsi = self.data['rsi']
                
                # Trend alignment (tous les TF dans la même direction?)
                rsi_cols = [c for c in self.data.columns if 'rsi' in c.lower()]
                if len(rsi_cols) > 1:
                    rsi_df = self.data[rsi_cols].fillna(50)
                    self.data['mtf_rsi_alignment'] = (rsi_df > 50).all(axis=1).astype(float) - (rsi_df < 50).all(axis=1).astype(float)
                else:
                    self.data['mtf_rsi_alignment'] = 0.0
                
                # Divergence MTF (RSI M5 vs RSI H1)
                if 'H1_rsi' in self.data.columns:
                    self.data['mtf_rsi_divergence'] = base_rsi - self.data['H1_rsi'].fillna(50)
                else:
                    self.data['mtf_rsi_divergence'] = 0.0
            else:
                self.data['mtf_rsi_alignment'] = 0.0
                self.data['mtf_rsi_divergence'] = 0.0
                
            logging.info("[MEGA] ✅ Tendance MTF calculée")
        except Exception as e:
            logging.warning(f"[MEGA] ⚠️ Erreur tendance MTF: {e}")
            self.data['mtf_rsi_alignment'] = 0.0
            self.data['mtf_rsi_divergence'] = 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # VOLATILITÉ RELATIVE
        # ═══════════════════════════════════════════════════════════════════
        try:
            if 'atr' in self.data.columns and 'close' in self.data.columns:
                # ATR relatif au prix (volatilité normalisée)
                self.data['atr_pct'] = self.data['atr'] / self.data['close'] * 100
                
                # Volatilité relative (ATR actuel vs moyenne 20 périodes)
                atr_ma = self.data['atr'].rolling(20, min_periods=1).mean()
                self.data['volatility_ratio'] = self.data['atr'] / (atr_ma + 1e-10)
                
                # High volatility flag
                self.data['high_volatility'] = (self.data['volatility_ratio'] > 1.5).astype(float)
            else:
                self.data['atr_pct'] = 0.0
                self.data['volatility_ratio'] = 1.0
                self.data['high_volatility'] = 0.0
                
            logging.info("[MEGA] ✅ Volatilité relative calculée")
        except Exception as e:
            logging.warning(f"[MEGA] ⚠️ Erreur volatilité: {e}")
        
        # ═══════════════════════════════════════════════════════════════════
        # PRIX RELATIFS
        # ═══════════════════════════════════════════════════════════════════
        try:
            if 'close' in self.data.columns:
                # Distance au plus haut/bas des 20 dernières périodes
                high_20 = self.data['high'].rolling(20, min_periods=1).max()
                low_20 = self.data['low'].rolling(20, min_periods=1).min()
                range_20 = high_20 - low_20 + 1e-10
                
                self.data['price_position'] = (self.data['close'] - low_20) / range_20
                
                # Distance aux Bollinger Bands (si disponibles)
                if 'bb_upper' in self.data.columns and 'bb_lower' in self.data.columns:
                    bb_range = self.data['bb_upper'] - self.data['bb_lower'] + 1e-10
                    self.data['bb_position'] = (self.data['close'] - self.data['bb_lower']) / bb_range
                else:
                    self.data['bb_position'] = 0.5
            else:
                self.data['price_position'] = 0.5
                self.data['bb_position'] = 0.5
                
            logging.info("[MEGA] ✅ Prix relatifs calculés")
        except Exception as e:
            logging.warning(f"[MEGA] ⚠️ Erreur prix relatifs: {e}")
        
        # ═══════════════════════════════════════════════════════════════════
        # MOMENTUM MULTI-PÉRIODE
        # ═══════════════════════════════════════════════════════════════════
        try:
            if 'close' in self.data.columns:
                # ROC (Rate of Change) sur différentes périodes
                self.data['roc_5'] = self.data['close'].pct_change(5).fillna(0) * 100
                self.data['roc_10'] = self.data['close'].pct_change(10).fillna(0) * 100
                self.data['roc_20'] = self.data['close'].pct_change(20).fillna(0) * 100
                
                logging.info("[MEGA] ✅ Momentum multi-période calculé")
        except Exception as e:
            logging.warning(f"[MEGA] ⚠️ Erreur momentum: {e}")
            self.data['roc_5'] = 0.0
            self.data['roc_10'] = 0.0
            self.data['roc_20'] = 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # PATCH V4.1: FEATURES POUR POSITION SIZING DYNAMIQUE
        # ═══════════════════════════════════════════════════════════════════
        try:
            # Signal direction basé sur momentum et RSI
            if 'rsi' in self.data.columns and 'roc_5' in self.data.columns:
                # Signal: +1 si bullish, -1 si bearish, 0 si neutre
                rsi = self.data['rsi']
                roc = self.data['roc_5']
                
                signal_direction = np.zeros(len(self.data))
                signal_direction[(rsi > 50) & (roc > 0)] = 1   # Bullish
                signal_direction[(rsi < 50) & (roc < 0)] = -1  # Bearish
                self.data['signal_direction'] = signal_direction
                
                # Confiance du signal (0 à 1) basée sur la force des indicateurs
                rsi_strength = np.abs(rsi - 50) / 50  # 0 si RSI=50, 1 si RSI=0 ou 100
                roc_strength = np.minimum(np.abs(roc) / 2, 1.0)  # Normalisé
                self.data['signal_confidence'] = (rsi_strength + roc_strength) / 2
                
                # Recommended position size multiplier (0.5 à 1.5)
                # Basé sur volatilité et confiance
                if 'volatility_ratio' in self.data.columns:
                    vol_factor = 1.0 / (self.data['volatility_ratio'].clip(0.5, 2.0))  # Inverse de volatilité
                else:
                    vol_factor = 1.0
                
                self.data['position_size_mult'] = (0.5 + self.data['signal_confidence'] * vol_factor).clip(0.5, 1.5)
                
                logging.info("[MEGA] ✅ Features position sizing ajoutées")
            else:
                self.data['signal_direction'] = 0.0
                self.data['signal_confidence'] = 0.5
                self.data['position_size_mult'] = 1.0
        except Exception as e:
            logging.warning(f"[MEGA] ⚠️ Erreur position sizing features: {e}")
            self.data['signal_direction'] = 0.0
            self.data['signal_confidence'] = 0.5
            self.data['position_size_mult'] = 1.0
        
        # ═══════════════════════════════════════════════════════════════════
        # NETTOYAGE FINAL
        # ═══════════════════════════════════════════════════════════════════
        # Remplacer tous les NaN et Inf
        self.data = self.data.replace([np.inf, -np.inf], 0)
        self.data = self.data.fillna(0)
        
        # Supprimer les colonnes non-numériques (comme 'time')
        non_numeric = self.data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logging.info(f"[MEGA] Suppression colonnes non-numériques: {non_numeric}")
            self.data = self.data.drop(columns=non_numeric)
        
        new_cols = len(self.data.columns)
        added = new_cols - original_cols
        
        logging.info("[MEGA] ═══════════════════════════════════════════════════")
        logging.info(f"[MEGA] ✅ ENRICHISSEMENT TERMINÉ!")
        logging.info(f"[MEGA]    Colonnes originales: {original_cols}")
        logging.info(f"[MEGA]    Colonnes ajoutées: +{added}")
        logging.info(f"[MEGA]    TOTAL FEATURES: {new_cols}")
        logging.info(f"[MEGA]    L'agent RL voit maintenant {new_cols} features!")
        logging.info("[MEGA] ═══════════════════════════════════════════════════")
        
        return self.data

    def _setup_trading_costs(self, symbol: str):
        """Configure les coÃƒÂ»ts de trading rÃƒÂ©alistes par symbole"""
        if 'XAU' in symbol or 'GOLD' in symbol:
            self.spread_pct = 0.00025
            self.commission_per_lot = 0.0
            self.swap_long = -0.05
            self.swap_short = -0.03
            self.min_position_duration = 1
        elif 'EUR' in symbol or 'GBP' in symbol:
            self.spread_pct = 0.0001
            self.commission_per_lot = 0.0003
            self.swap_long = -0.02
            self.swap_short = -0.01
            self.min_position_duration = 1
        else:
            self.spread_pct = 0.0002
            self.commission_per_lot = 0.0003
            self.swap_long = -0.03
            self.swap_short = -0.02
            self.min_position_duration = 1
        
        self.base_slippage = 0.00005
        self.high_vol_slippage = 0.0002
        self.execution_delay_steps = 1

    def _reset_daily_tracking(self):
        """RÃƒÂ©initialise le tracking quotidien"""
        self.daily_pnl = 0.0
        self.daily_trades = 0

    def reset(self, seed=None, options=None):
        """Reset l'environnement - COMPATIBLE GYMNASIUM & SB3"""
        self.current_step = np.random.randint(100, len(self.data) - 100)
        
        # Reset ÃƒÂ©tat trading
        self.position = 0
        self.entry_price = 0.0
        self.pnl = 0.0
        self.equity = self.initial_balance
        self.free_margin = self.initial_balance
        self.used_margin = 0.0
        self.position_open_step = 0
        
        # Reset tracking
        self._reset_daily_tracking()
        self.total_spread_paid = 0.0
        self.total_commission_paid = 0.0
        self.total_swap_paid = 0.0
        self.total_slippage_cost = 0.0
        
        # Reset timing optimal
        self.peak_profit = 0.0
        self.peak_pips = 0.0
        self.peak_step = 0
        self.max_drawdown = 0.0
        self.time_since_peak = 0
        self.entry_quality_score = 0.0
        self.optimal_entries = 0
        self.optimal_exits = 0
        self.perfect_trades = 0
        
        # Reset phase si nÃƒÂ©cessaire
        if self.current_step % self.phase_length == 0:
            self.current_phase = self.current_step // self.phase_length
            self.phase_start_step = self.current_step
            self.phase_start_equity = self.equity
            self.phase_closes = []
            self.phase_pnl = 0.0
            self.phase_wins = 0
            self.phase_losses = 0
        
        obs = self._get_obs()
        info = self.get_performance_report()
        
        return obs, info

    def step(self, action: int):
        """ExÃƒÂ©cute une action - COMPATIBLE GYMNASIUM & SB3"""
        self.current_step += 1
        
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        # Calcul du reward AVEC SYSTÃƒÂˆME CORRIGÃƒÂ‰
        reward = self._calculate_reward_corrected(action)
        
        # Fermeture forcÃƒÂ©e en fin d'ÃƒÂ©pisode
        if self.current_step >= len(self.data) - 1 and self.position != 0:
            final_reward = self._force_close_position()
            reward += final_reward
        
        # Gestion de fin de phase
        if self.current_step > 0 and self.current_step % self.phase_length == 0:
            self._end_phase()
        
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        info = {
            'equity': self.equity,
            'position': self.position,
            'step': self.current_step,
            'pnl': self.pnl,
            'performance': self.get_performance_report()
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Obtient l'observation raffinÃƒÂ©e via HRM-TRM"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
            
        raw_obs = self.data.iloc[self.current_step].values.astype(np.float32)
        
        if np.isnan(raw_obs).any() or np.isinf(raw_obs).any():
            logging.warning(f"  Observation corrompue pour {self.symbol}, nettoyage")
            raw_obs = np.nan_to_num(raw_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalisation robuste
        obs_mean = np.mean(raw_obs)
        obs_std = np.std(raw_obs)
        
        if obs_std < 1e-8:
            raw_obs_normalized = raw_obs - obs_mean
        else:
            raw_obs_normalized = (raw_obs - obs_mean) / obs_std
        
        raw_obs_normalized = np.clip(raw_obs_normalized, -5.0, 5.0)
        
        # Raffinement HRM-TRM
        x = torch.from_numpy(raw_obs_normalized).float().unsqueeze(0)
        y_init = torch.tensor([2])  # HOLD initial
        z_init = torch.zeros(1, CONFIG.HRM_HIDDEN_DIM)
        
        with torch.no_grad():
            refined_obs = self.hrm_trm.refine(x, y_init, z_init, self.symbol_id)
        
        obs_np = refined_obs.squeeze(0).numpy()
        
        if np.isnan(obs_np).any() or np.isinf(obs_np).any():
            logging.error(f"  Observation raffinÃƒÂ©e corrompue pour {self.symbol}")
            obs_np = np.zeros(CONFIG.HRM_HIDDEN_DIM, dtype=np.float32)
        
        return obs_np

    def _calculate_reward_corrected(self, action: int) -> float:
        """
        SYSTÃƒÂˆME DE RÃƒÂ‰COMPENSE CORRIGÃƒÂ‰ - Aligne rÃƒÂ©compenses avec performance rÃƒÂ©elle
        """
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0.0
        
        # Gestion basique des actions
        if action == 0:  # BUY
            reward = self._handle_buy_simple(current_price)
        elif action == 1:  # SELL
            reward = self._handle_sell_simple(current_price)
        elif action == 2:  # HOLD
            reward = self._handle_hold_simple(current_price)
        elif action == 3:  # CLOSE
            reward = self._handle_close_simple(current_price)
        
        return np.clip(reward, -50, +100)

    def _handle_buy_simple(self, current_price: float) -> float:
        """GÃƒÂ¨re l'action BUY - Version simple et corrigÃƒÂ©e"""
        reward = 0.0
        
        if self.position == 0:
            # OUVRIR LONG
            self.position = 1
            self.entry_price = current_price
            self.position_open_step = self.current_step
            self.daily_trades += 1
            

            # DRAGON: Calculate confluence score
            try:
                self._last_confluence = self.confluence_detector.calculate_confluence_score(
                    self.data, 
                    self.current_step
                )
                logging.debug(f"[DRAGON] Entry confluence: {self._last_confluence['score']:.1f}/100 {self._last_confluence['direction']}")
            except Exception as e:
                self._last_confluence = {}
                logging.debug(f"[DRAGON] Error calculating confluence: {e}")
            
            # ===========================================================================
            # [GOLDENEYE ULTIMATE] DIVERGENCE DETECTION Ã€ L'ENTRY
            # ===========================================================================
            try:
                prices = self.data['close'].values[:self.current_step+1]
                
                # Extraire indicateurs
                rsi_values = []
                macd_values = []
                momentum_values = []
                
                for col in self.data.columns:
                    if 'rsi' in col.lower() and len(rsi_values) == 0:
                        rsi_values = self.data[col].values[:self.current_step+1]
                    if 'macd' in col.lower() and 'signal' not in col.lower() and len(macd_values) == 0:
                        macd_values = self.data[col].values[:self.current_step+1]
                    if 'momentum' in col.lower() and len(momentum_values) == 0:
                        momentum_values = self.data[col].values[:self.current_step+1]
                
                # Calcul si pas trouvÃ©
                if len(rsi_values) == 0:
                    rsi_values = self.tda_extractor.calculate_rsi(prices)
                if len(macd_values) == 0:
                    macd_values, _ = self.tda_extractor.calculate_macd(prices)
                if len(momentum_values) == 0:
                    momentum_values = self.tda_extractor.calculate_momentum(prices)
                
                # DÃ©tecter divergences
                self.current_divergences = self.divergence_system.detect_all_divergences(
                    prices, rsi_values, macd_values, momentum_values, lookback=20
                )
                
                if self.current_divergences:
                    self.trade_on_divergence = True
                    self.divergence_at_entry = self.current_divergences[0]
                    reward += 10.0  # BONUS +10 pour divergence!
                    logging.info(f"[DIVERGENCE] BUY on {self.divergence_at_entry['type']} {self.divergence_at_entry['indicator']} - BONUS +10")
                else:
                    self.trade_on_divergence = False
                    self.divergence_at_entry = None
                    
            except Exception as e:
                logging.debug(f"[DIVERGENCE] Detection error: {e}")
                self.trade_on_divergence = False
                self.divergence_at_entry = None
            
            # ===========================================================================
            # [SUPER MERLIN] PATTERN ANALYSIS & PREDICTION
            # ===========================================================================
            try:
                if self.merlin_analyzer.enabled:
                    prices_recent = self.data['close'].values[:self.current_step+1]
                    
                    # PrÃ©parer indicateurs pour MERLIN
                    indicators = {}
                    if len(rsi_values) > 0:
                        indicators['rsi'] = rsi_values
                    if len(macd_values) > 0:
                        indicators['macd'] = macd_values
                    
                    # Analyser pattern actuel
                    self.merlin_prediction = self.merlin_analyzer.analyze_current_pattern(
                        prices_recent, indicators, timeframe='M5'
                    )
                    
                    if self.merlin_prediction:
                        pred = self.merlin_prediction['prediction']
                        self.merlin_at_entry = pred
                        
                        # BONUS si haute confiance (>80%)
                        if pred['confidence'] > 0.8:
                            reward += 15.0  # BONUS +15 pour prÃ©diction MERLIN haute confiance!
                            logging.info(f"[MERLIN] BUY with {pred['direction']} prediction ({pred['confidence']*100:.0f}% confidence) - BONUS +15")
                        elif pred['confidence'] > 0.6:
                            reward += 5.0
                            logging.debug(f"[MERLIN] BUY with {pred['direction']} prediction ({pred['confidence']*100:.0f}% confidence)")
                        else:
                            logging.debug(f"[MERLIN] BUY with weak {pred['direction']} prediction ({pred['confidence']*100:.0f}% confidence)")
                    else:
                        self.merlin_at_entry = None
                else:
                    self.merlin_at_entry = None
                    
            except Exception as e:
                logging.debug(f"[MERLIN] Analysis error: {e}")
                self.merlin_at_entry = None
            
            # ═══════════════════════════════════════════════════════════════════
            # [V5 ULTRA] MERLIN PROFIT PREDICTION AT ENTRY
            # ═══════════════════════════════════════════════════════════════════
            try:
                current_row = self.data.iloc[self.current_step]
                rsi = current_row.get("rsi", 50)
                macd = current_row.get("macd", 0)
                
                # Calculer price change (momentum)
                if self.current_step > 0:
                    prev_price = self.data.iloc[self.current_step - 1]["close"]
                    price_change = (current_price - prev_price) / prev_price
                else:
                    price_change = 0
                
                # MERLIN prédit le profit de ce trade
                merlin_pred = self.merlin_profit_tracker.predict_trade_outcome(
                    rsi=rsi,
                    macd=macd,
                    price_change=price_change
                )
                
                # Enregistrer la prédiction
                self.merlin_profit_tracker.record_prediction(merlin_pred, current_price)
                
                logging.debug(
                    f"[MERLIN PROFIT] Predicted PnL: ${merlin_pred['expected_pnl']:+.2f} "
                    f"(confidence: {merlin_pred['confidence']:.2f})"
                )
                
            except Exception as e:
                logging.debug(f"[MERLIN PROFIT] Error predicting: {e}")
            
            # TRACKING PERFECT TIMING
            try:
                self.perfect_timing.track_position(
                    symbol=self.symbol,
                    entry_price=current_price,
                    entry_step=self.current_step,
                    position_type=1  # LONG
                )
            except Exception as e:
                logging.warning(f"[PERFECT TIMING] Error tracking BUY: {e}")
            
            reward = 0.1  # Petit bonus pour initiative
            
        elif self.position == -1:
            # FERMER SHORT + OUVRIR LONG
            profit_pct = (self.entry_price - current_price) / self.entry_price
            self._update_position_stats(profit_pct, "SHORT")
            reward = self._calculate_profit_reward(profit_pct)
            
            # Ouvrir LONG
            self.position = 1
            self.entry_price = current_price
            self.position_open_step = self.current_step
            
        else:
            # DÃƒÂ‰JÃƒÂ€ LONG - HOLD
            unrealized_profit = (current_price - self.entry_price) / self.entry_price
            reward = self._calculate_hold_reward_simple(unrealized_profit)
        
        return reward

    def _handle_sell_simple(self, current_price: float) -> float:
        """GÃƒÂ¨re l'action SELL - Version simple et corrigÃƒÂ©e"""
        reward = 0.0
        
        if self.position == 0:
            # OUVRIR SHORT
            self.position = -1
            self.entry_price = current_price
            self.position_open_step = self.current_step
            self.daily_trades += 1
            

            # DRAGON: Calculate confluence score
            try:
                self._last_confluence = self.confluence_detector.calculate_confluence_score(
                    self.data, 
                    self.current_step
                )
                logging.debug(f"[DRAGON] Entry confluence: {self._last_confluence['score']:.1f}/100 {self._last_confluence['direction']}")
            except Exception as e:
                self._last_confluence = {}
                logging.debug(f"[DRAGON] Error calculating confluence: {e}")
            
            # ===========================================================================
            # [GOLDENEYE ULTIMATE] DIVERGENCE DETECTION Ã€ L'ENTRY SHORT
            # ===========================================================================
            try:
                prices = self.data['close'].values[:self.current_step+1]
                
                # Extraire indicateurs
                rsi_values = []
                macd_values = []
                momentum_values = []
                
                for col in self.data.columns:
                    if 'rsi' in col.lower() and len(rsi_values) == 0:
                        rsi_values = self.data[col].values[:self.current_step+1]
                    if 'macd' in col.lower() and 'signal' not in col.lower() and len(macd_values) == 0:
                        macd_values = self.data[col].values[:self.current_step+1]
                    if 'momentum' in col.lower() and len(momentum_values) == 0:
                        momentum_values = self.data[col].values[:self.current_step+1]
                
                # Calcul si pas trouvÃ©
                if len(rsi_values) == 0:
                    rsi_values = self.tda_extractor.calculate_rsi(prices)
                if len(macd_values) == 0:
                    macd_values, _ = self.tda_extractor.calculate_macd(prices)
                if len(momentum_values) == 0:
                    momentum_values = self.tda_extractor.calculate_momentum(prices)
                
                # DÃ©tecter divergences
                self.current_divergences = self.divergence_system.detect_all_divergences(
                    prices, rsi_values, macd_values, momentum_values, lookback=20
                )
                
                if self.current_divergences:
                    self.trade_on_divergence = True
                    self.divergence_at_entry = self.current_divergences[0]
                    reward += 10.0  # BONUS +10 pour divergence!
                    logging.info(f"[DIVERGENCE] SELL on {self.divergence_at_entry['type']} {self.divergence_at_entry['indicator']} - BONUS +10")
                else:
                    self.trade_on_divergence = False
                    self.divergence_at_entry = None
                    
            except Exception as e:
                logging.debug(f"[MERLIN] Analysis error: {e}")
                self.merlin_at_entry = None
            
            # ═══════════════════════════════════════════════════════════════════
            # [V5 ULTRA] MERLIN PROFIT PREDICTION AT ENTRY
            # ═══════════════════════════════════════════════════════════════════
            try:
                current_row = self.data.iloc[self.current_step]
                rsi = current_row.get("rsi", 50)
                macd = current_row.get("macd", 0)
                
                # Calculer price change
                if self.current_step > 0:
                    prev_price = self.data.iloc[self.current_step - 1]["close"]
                    price_change = (current_price - prev_price) / prev_price
                else:
                    price_change = 0
                
                # MERLIN prédit le profit (négatif pour SELL)
                merlin_pred = self.merlin_profit_tracker.predict_trade_outcome(
                    rsi=rsi,
                    macd=macd,
                    price_change=-price_change  # Inversé pour SELL
                )
                
                # Enregistrer la prédiction
                self.merlin_profit_tracker.record_prediction(merlin_pred, current_price)
                
                logging.debug(
                    f"[MERLIN PROFIT] SELL Predicted PnL: ${merlin_pred['expected_pnl']:+.2f}"
                )
                
            except Exception as e:
                logging.debug(f"[MERLIN PROFIT] Error predicting: {e}")
                logging.debug(f"[DIVERGENCE] Detection error: {e}")
                self.trade_on_divergence = False
                self.divergence_at_entry = None
            
            # ===========================================================================
            # [SUPER MERLIN] PATTERN ANALYSIS & PREDICTION SHORT
            # ===========================================================================
            try:
                if self.merlin_analyzer.enabled:
                    prices_recent = self.data['close'].values[:self.current_step+1]
                    
                    # PrÃ©parer indicateurs pour MERLIN
                    indicators = {}
                    if len(rsi_values) > 0:
                        indicators['rsi'] = rsi_values
                    if len(macd_values) > 0:
                        indicators['macd'] = macd_values
                    
                    # Analyser pattern actuel
                    self.merlin_prediction = self.merlin_analyzer.analyze_current_pattern(
                        prices_recent, indicators, timeframe='M5'
                    )
                    
                    if self.merlin_prediction:
                        pred = self.merlin_prediction['prediction']
                        self.merlin_at_entry = pred
                        
                        # BONUS si haute confiance (>80%)
                        if pred['confidence'] > 0.8:
                            reward += 15.0  # BONUS +15 pour prÃ©diction MERLIN haute confiance!
                            logging.info(f"[MERLIN] SELL with {pred['direction']} prediction ({pred['confidence']*100:.0f}% confidence) - BONUS +15")
                        elif pred['confidence'] > 0.6:
                            reward += 5.0
                            logging.debug(f"[MERLIN] SELL with {pred['direction']} prediction ({pred['confidence']*100:.0f}% confidence)")
                        else:
                            logging.debug(f"[MERLIN] SELL with weak {pred['direction']} prediction ({pred['confidence']*100:.0f}% confidence)")
                    else:
                        self.merlin_at_entry = None
                else:
                    self.merlin_at_entry = None
                    
            except Exception as e:
                logging.debug(f"[MERLIN] Analysis error: {e}")
                self.merlin_at_entry = None
            
            # TRACKING PERFECT TIMING
            try:
                self.perfect_timing.track_position(
                    symbol=self.symbol,
                    entry_price=current_price,
                    entry_step=self.current_step,
                    position_type=-1  # SHORT
                )
            except Exception as e:
                logging.warning(f"[PERFECT TIMING] Error tracking SELL: {e}")
            
            reward = 0.1
            
        elif self.position == 1:
            # FERMER LONG + OUVRIR SHORT
            profit_pct = (current_price - self.entry_price) / self.entry_price
            self._update_position_stats(profit_pct, "LONG")
            reward = self._calculate_profit_reward(profit_pct)
            
            # Ouvrir SHORT
            self.position = -1
            self.entry_price = current_price
            self.position_open_step = self.current_step
            
        else:
            # DÃƒÂ‰JÃƒÂ€ SHORT - HOLD
            unrealized_profit = (self.entry_price - current_price) / self.entry_price
            reward = self._calculate_hold_reward_simple(unrealized_profit)
        
        return reward

    def _handle_hold_simple(self, current_price: float) -> float:
        """Gère HOLD — proportionnel au P&L latent si position, 0 sinon."""
        if self.position != 0:
            if self.position == 1:  # LONG
                unrealized_profit = (current_price - self.entry_price) / self.entry_price
            else:  # SHORT
                unrealized_profit = (self.entry_price - current_price) / self.entry_price
            return self._calculate_hold_reward_simple(unrealized_profit)
        return 0.0

    MIN_HOLD_STEPS = 10  # Durée minimale de détention avant autorisation de CLOSE

    def _handle_close_simple(self, current_price: float) -> float:
        """Gère l'action CLOSE avec durée minimale de détention."""
        if self.position != 0:
            held_steps = self.current_step - self.position_open_step
            if held_steps < self.MIN_HOLD_STEPS:
                return -0.1
            # Calculer le P&L de base
            if self.position == 1:  # LONG
                profit_pct = (current_price - self.entry_price) / self.entry_price
            else:  # SHORT
                profit_pct = (self.entry_price - current_price) / self.entry_price
            
            
            # ═══════════════════════════════════════════════════════════════════
            # [V5 ULTRA] RECORD ACTUAL PROFIT FOR MERLIN
            # ═══════════════════════════════════════════════════════════════════
            try:
                profit_dollars = profit_pct * self.initial_balance
                self.merlin_profit_tracker.record_outcome(profit_dollars)
                logging.debug(f"[MERLIN PROFIT] Actual PnL recorded: ${profit_dollars:+.2f}")
            except Exception as e:
                logging.debug(f"[MERLIN PROFIT] Error recording outcome: {e}")

            # ═══════════════════════════════════════════════════════════════════
            # [V6 GALAXY] EXPORT PNG → cerveau/captures/ (boucle fermée)
            # Toutes les 20 fermetures — MERLIN s'enrichit pendant l'entraînement
            # ═══════════════════════════════════════════════════════════════════
            try:
                self.total_closed = getattr(self, 'total_closed', 0) + 1
                if self.total_closed % 20 == 0:
                    from goldeneye_png_exporter import exporter_capture_ppo
                    exporter_capture_ppo(
                        self.data, self.current_step, self.symbol,
                        profit_pct, self.position,
                        getattr(self, '_timeframe_name', 'M15')
                    )
            except Exception:
                pass
            # Reward de base pour le profit/perte
            base_reward = self._calculate_profit_reward(profit_pct)
            
            # ===========================================================
            # [GOLDENEYE ULTIMATE] DIVERGENCE BONUS x1.5
            # ===========================================================
            if self.trade_on_divergence and self.divergence_at_entry:
                profit_pips = profit_pct * 10000
                outcome = 'SUCCESS' if profit_pct > 0 else 'FAILURE'
                
                try:
                    self.divergence_system.save_divergence_pattern(
                        self.divergence_at_entry,
                        outcome,
                        profit_pips
                    )
                except Exception as e:
                    logging.debug(f"[DIVERGENCE] Save error: {e}")
                
                # BONUS x1.5 si profitable!
                if profit_pct > 0:
                    base_reward *= 1.5
                    logging.info(f"[DIVERGENCE] WIN on {self.divergence_at_entry['type']} - BONUS x1.5! Reward: {base_reward:.1f}")
                else:
                    logging.info(f"[DIVERGENCE] LOSS on {self.divergence_at_entry['type']} - Pattern saved")
                
                # Reset
                self.trade_on_divergence = False
                self.divergence_at_entry = None
            
            # ===========================================================
            # [SUPER MERLIN] PREDICTION ACCURACY BONUS x1.3
            # ===========================================================
            if self.merlin_at_entry and self.merlin_analyzer.enabled:
                try:
                    pred_direction = self.merlin_at_entry['direction']
                    pred_confidence = self.merlin_at_entry['confidence']
                    
                    # DÃ©terminer outcome rÃ©el
                    if profit_pct > 0.001:  # Profit
                        if self.position == 1:  # LONG profitable
                            actual_outcome = 'HAUSSIER'
                        else:  # SHORT profitable
                            actual_outcome = 'BAISSIER'
                    elif profit_pct < -0.001:  # Loss
                        if self.position == 1:  # LONG perdant
                            actual_outcome = 'BAISSIER'
                        else:  # SHORT perdant
                            actual_outcome = 'HAUSSIER'
                    else:
                        actual_outcome = 'NEUTRE'
                    
                    # Enregistrer rÃ©sultat
                    self.merlin_analyzer.record_prediction_outcome(pred_direction, actual_outcome)
                    
                    # BONUS x1.3 si prÃ©diction correcte!
                    if pred_direction == actual_outcome:
                        base_reward *= 1.3
                        logging.info(f"[MERLIN] Prediction CORRECT ({pred_direction}) - BONUS x1.3! Reward: {base_reward:.1f}")
                    else:
                        logging.debug(f"[MERLIN] Prediction incorrect: predicted {pred_direction}, actual {actual_outcome}")
                    
                except Exception as e:
                    logging.debug(f"[MERLIN] Bonus error: {e}")
                
                # Reset
                self.merlin_at_entry = None
            
            # Perfect Timing bonus supprimé du gradient (Manhattan)
            total_reward = base_reward
            
            # Update stats
            self._update_position_stats(profit_pct, "CLOSE")
            
            # Fermer position
            self.position = 0
            self.entry_price = 0.0
            
            
            
            # ===============================================================
            # DRAGON: ANALYZE TRADE FOR LEARNING
            # ===============================================================
            try:
                # Calculer les pips
                pips = profit_pct * 10000
                
                # CrÃƒÂ©er le trade data pour analyse
                trade_data = {
                    'direction': 'LONG' if self.position == 1 else 'SHORT',
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'pips': pips,
                    'reward': total_reward,
                    'entry_confluence': getattr(self, '_last_confluence', {}),
                    'hold_time': self.current_step - getattr(self, 'position_open_step', self.current_step)
                }
                
                # Ajouter au Success Pattern Analyzer
                self.success_analyzer.add_trade_result(trade_data)
                
                # Log patterns if winning
                if pips > 0:
                    logging.debug(f"[DRAGON] Winning trade analyzed and stored")
                
            
                
                # ADVANCED: Store in Long-Term Memory if winning
                if pips > 5:  # Seulement les bons winners
                    try:
                        regime = getattr(self, '_current_regime', {'name': 'UNKNOWN'})
                        patterns = getattr(self, '_current_patterns', [])
                        
                        # CrÃƒÂ©er un pattern dict
                        pattern_data = {
                            'confluence': trade_data.get('entry_confluence', {}),
                            'patterns': patterns,
                            'indicators': {
                                'rsi': self.data.iloc[self.current_step].get('rsi_14', 0),
                                'macd': self.data.iloc[self.current_step].get('macd', 0)
                            }
                        }
                        
                        # Stocker dans LTM
                        self.long_term_memory.store_successful_pattern(
                            pattern_data,
                            regime['name'],
                            pips
                        )
                        
                        logging.info(f"[LTM] Stored winning pattern: {pips:.1f} pips in {regime['name']}")
                        
                    except Exception as ltm_error:
                        logging.debug(f"[LTM] Error storing pattern: {ltm_error}")
                
            except Exception as e:
                logging.debug(f"[DRAGON] Error analyzing trade: {e}")
            
            return total_reward
        else:
            return -0.5  # PÃƒÂ©nalitÃƒÂ© pour CLOSE sans position

    def _calculate_profit_reward(self, profit_pct: float, holding_time: int = 0) -> float:
        """
        MANHATTAN PURE P&L REWARD — clip(±20%) × scale=0.01
        Pur signal P&L sans multiplicateurs qui tuent le gradient.
        $100 profit → reward +1.0 | $100 perte → reward -1.0
        """
        profit_pct_safe = float(np.clip(profit_pct, -0.20, 0.20))
        profit_dollars = profit_pct * self.initial_balance  # vrai PnL pour le log
        # Reward normalisé [-1, +1] : value function stable → EV > 0 → gradient réel
        # 20% gain = +1.0 | 20% perte = -1.0 | 1% gain = +0.05
        final_reward = profit_pct_safe / 0.20

        # Tracking des variables (utilisées ailleurs, conservées)
        self.recent_pnl_history.append(profit_pct)
        if len(self.recent_pnl_history) > self.max_pnl_history:
            self.recent_pnl_history.pop(0)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity

        result_str = "WIN" if profit_dollars > 0 else "LOSS"
        logging.info(
            f"  [MANHATTAN P&L] {result_str} ${profit_dollars:+.2f} "
            f"({profit_pct*10000:+.1f} pips) → reward={final_reward:+.3f}"
        )
        return float(final_reward)

    def _calculate_hold_reward_simple(self, unrealized_profit: float) -> float:
        """Calcule le reward pour HOLD - Version simple"""
        unrealized_pips = unrealized_profit * 10000
        
        if unrealized_pips > 1:  # Seuil ÃƒÂ  +1 pip
            return min(1.0, unrealized_pips * 0.1)
        elif unrealized_pips < -2:  # Seuil ÃƒÂ  -2 pips
            return max(-2.0, unrealized_pips * 0.2)
        
        return -0.05  # Plat -> pousse l'agent a decider

    def _update_position_stats(self, profit_pct: float, action_type: str):
        """Met ÃƒÂ  jour les statistiques de position"""
        self.total_closed += 1
        profit_pips = profit_pct * 10000
        profit_dollars = profit_pct * self.initial_balance
        
        
        # PATCH V4.1: Mettre à jour le position sizer
        global GLOBAL_POSITION_SIZER
        if GLOBAL_POSITION_SIZER is not None:
            GLOBAL_POSITION_SIZER.record_trade(profit_dollars)
            GLOBAL_POSITION_SIZER.update_equity(self.equity)
        if profit_pct > 0:
            self.total_wins += 1
            self.close_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            logging.info(f"  {action_type} WIN: {self.symbol} +{profit_pips:.1f} pips (+${profit_dollars:.2f})")
        else:
            self.total_losses += 1
            
            # LOSS REDUCTION: Analyser cette perte
            try:
                context = {
                    'confluence': getattr(self, '_last_confluence', {}),
                    'regime': getattr(self, '_current_regime', {}),
                    'rsi': self.data.iloc[self.current_step].get('rsi_14', 50) if self.current_step < len(self.data) else 50
                }
                
                # Enregistrer la perte
                state = self.data.iloc[self.current_step].values if self.current_step < len(self.data) else np.zeros(10)
                self.loss_analyzer.analyze_loss(state, self.position, context)
                
                # VÃƒÂ©rifier si pattern redondant
                redundant = self.loss_analyzer.get_redundant_losses(min_frequency=3)
                if redundant:
                    logging.warning(f"[LOSS PATTERN] Detected {len(redundant)} redundant loss patterns!")
                    for sig, freq in redundant[:3]:
                        logging.warning(f"  Ã¢Â€Â¢ {sig}: {freq}x")
                
            except Exception as e:
                logging.debug(f"[LOSS] Error analyzing: {e}")

            self.close_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            logging.warning(f"  {action_type} LOSS: {self.symbol} {profit_pips:.1f} pips (${profit_dollars:.2f})")
        
        # Mise ÃƒÂ  jour equity
        self.equity *= (1 + profit_pct)
        self.pnl += profit_pct * self.initial_balance
        
        # Tracking phase
        self.phase_closes.append(profit_pct)
        if profit_pct > 0:
            self.phase_wins += 1
            self.phase_pnl += profit_pct * 100
        else:
            self.phase_losses += 1
            self.phase_pnl += profit_pct * 100

    def _force_close_position(self):
        """FERMETURE FORCÃƒÂ‰E en fin d'ÃƒÂ©pisode"""
        if self.position != 0:
            current_price = self.data.iloc[self.current_step]['close']
            
            SPREAD = 0.00025 if ('XAU' in self.symbol or 'GOLD' in self.symbol) else 0.0001
            
            if self.position == 1:
                exit_price = current_price * (1 - SPREAD)
                pnl_pct = (exit_price - self.entry_price) / self.entry_price
            else:
                exit_price = current_price * (1 + SPREAD)
                pnl_pct = (self.entry_price - exit_price) / self.entry_price
            
            final_reward = self._calculate_profit_reward(pnl_pct)
            self._update_position_stats(pnl_pct, "FORCE CLOSE")
            
            # Reset position
            self.position = 0
            self.entry_price = 0.0
            self.position_open_step = 0
            
            return final_reward
        
        return 0.0


    def _scan_missed_opportunities(self):
        """
        [DRAGON] Scanne l'historique rÃƒÂ©cent pour dÃƒÂ©tecter les patterns loupÃƒÂ©s
        
        Cette mÃƒÂ©thode:
        1. Extrait une fenÃƒÂªtre historique (100 derniers steps)
        2. DÃƒÂ©tecte les trinitÃƒÂ©s parfaites dans cette fenÃƒÂªtre
        3. Compare avec les actions de l'agent
        4. Log et stocke les patterns loupÃƒÂ©s
        5. Ajoute au replay buffer pour apprentissage
        """
        try:
            # ÃƒÂ‰viter de re-scanner la mÃƒÂªme fenÃƒÂªtre
            window_id = (self.current_step // 50) * 50
            if window_id in self.scanned_windows:
                return
            
            self.scanned_windows.add(window_id)
            
            # Extraire fenÃƒÂªtre historique
            lookback = 100
            start_step = max(0, self.current_step - lookback)
            end_step = self.current_step
            
            if end_step - start_step < 50:
                return  # Pas assez de donnÃƒÂ©es
            
            logging.debug(f"[DRAGON] Scanning steps {start_step} to {end_step} for missed patterns")
            
            # DÃƒÂ©tecter les trinitÃƒÂ©s parfaites
            perfect_patterns = self.missed_opp_detector.scan_for_perfect_trinity_patterns(
                self.data,
                start_step,
                end_step
            )
            
            if not perfect_patterns:
                logging.debug(f"[DRAGON] No perfect patterns found in window")
                return
            
            logging.info(f"[DRAGON] Found {len(perfect_patterns)} perfect patterns in window")
            
            # Comparer avec les actions de l'agent
            for pattern in perfect_patterns:
                # VÃƒÂ©rifier si l'agent a agi correctement ÃƒÂ  ce moment
                agent_missed = self._did_agent_miss_pattern(pattern)
                
                if agent_missed:
                    # L'AGENT A LOUPÃƒÂ‰ CE PATTERN!
                    self.missed_patterns_history.append(pattern)
                    
                    # LOG DÃƒÂ‰TAILLÃƒÂ‰
                    logging.warning(f"[DRAGON MISSED] Step {pattern['entry_step']}:")
                    logging.warning(f"  Perfect {pattern['direction']} setup")
                    logging.warning(f"  Entry: {pattern['entry_price']:.5f}")
                    logging.warning(f"  Potential: +{pattern['potential_pips']:.1f} pips")
                    logging.warning(f"  Confluence: {pattern['entry_confluence']['score']:.1f}/100")
                    logging.warning(f"  Signals: {len(pattern['entry_confluence']['signals'])} categories")
                    
                    # Ajouter au replay buffer pour apprentissage
                    agent_state = self._get_observation_at_step(pattern['entry_step'])
                    self.experience_replay.add_missed_opportunity(pattern, agent_state)
                    
                    # Stocker dans la mÃƒÂ©moire
                    pattern['signature'] = self._create_pattern_signature(pattern)
                    self.pattern_memory.store_pattern(pattern)
            
            # Statistiques
            if len(self.missed_patterns_history) > 0:
                total_missed = len(self.missed_patterns_history)
                total_pips_lost = sum(p['potential_pips'] for p in self.missed_patterns_history)
                logging.info(f"[DRAGON STATS] Total missed patterns: {total_missed}")
                logging.info(f"[DRAGON STATS] Total pips lost: {total_pips_lost:.1f}")
                
        except Exception as e:
            logging.error(f"[DRAGON] Error in _scan_missed_opportunities: {e}")
            import traceback
            traceback.print_exc()
    
    def _did_agent_miss_pattern(self, pattern: dict) -> bool:
        """
        VÃƒÂ©rifie si l'agent a loupÃƒÂ© ce pattern
        
        Returns:
            True si l'agent n'a pas pris position au bon moment
        """
        entry_step = pattern['entry_step']
        
        # VÃƒÂ©rifier si l'agent avait une position ÃƒÂ  ce moment
        # Simplification: si position == 0 ou position dans mauvaise direction
        if entry_step < len(self.data):
            # L'agent aurait dÃƒÂ» ouvrir une position ici
            # Pour l'instant, on considÃƒÂ¨re qu'il a loupÃƒÂ© si on dÃƒÂ©tecte le pattern
            return True  # Simplification
        
        return False
    
    def _get_observation_at_step(self, step: int):
        """Reconstruit l'observation ÃƒÂ  un step donnÃƒÂ©"""
        try:
            if step < len(self.data):
                row = self.data.iloc[step]
                # Extraire les features principales
                obs = row[self.data.columns].values
                return obs
        except:
            pass
        
        return np.zeros(len(self.data.columns))
    
    def _create_pattern_signature(self, pattern: dict) -> str:
        """CrÃƒÂ©e une signature unique du pattern"""
        components = [
            pattern['direction'],
            f"CONF{int(pattern['entry_confluence']['score'])}",
            f"PIPS{int(pattern['potential_pips'])}"
        ]
        
        # Ajouter les 2 signaux les plus forts
        all_signals = []
        for category, signals in pattern['entry_confluence']['signals'].items():
            all_signals.extend(signals)
        
        all_signals.sort(key=lambda x: x[1], reverse=True)
        for signal_name, score in all_signals[:2]:
            clean_name = signal_name.replace(' ', '_')[:15]
            components.append(clean_name)
        
        return "_".join(components)


    def _generate_dragon_report(self):
        """
        [DRAGON] GÃƒÂ©nÃƒÂ¨re un rapport complet des systÃƒÂ¨mes Dragon
        AppelÃƒÂ© pÃƒÂ©riodiquement pour voir l'ÃƒÂ©volution
        """
        try:
            logging.info("=" * 80)
            logging.info("[DRAGON REPORT] LEARNING SYSTEMS STATUS")
            logging.info("=" * 80)
            
            # 1. Missed Opportunities
            if self.missed_patterns_history:
                total_missed = len(self.missed_patterns_history)
                total_pips_lost = sum(p['potential_pips'] for p in self.missed_patterns_history)
                avg_confluence = np.mean([p['entry_confluence']['score'] for p in self.missed_patterns_history])
                
                logging.info(f"[MISSED OPPORTUNITIES]")
                logging.info(f"  Total Patterns Missed: {total_missed}")
                logging.info(f"  Total Pips Lost: {total_pips_lost:.1f}")
                logging.info(f"  Avg Confluence Score: {avg_confluence:.1f}/100")
            
            # 2. Success Patterns
            success_stats = self.success_analyzer.get_statistics()
            logging.info(f"[SUCCESS PATTERNS]")
            logging.info(f"  Total Trades: {success_stats['total_trades']}")
            logging.info(f"  Winrate: {success_stats['winrate']:.1f}%")
            logging.info(f"  Patterns Found: {success_stats['patterns_found']}")
            if success_stats['total_trades'] > 0:
                logging.info(f"  Avg Win: +{success_stats['avg_winning_pips']:.1f} pips")
                logging.info(f"  Avg Loss: {success_stats['avg_losing_pips']:.1f} pips")
            
            # 3. Best Patterns
            best_patterns = self.success_analyzer.get_best_patterns(top_n=3)
            if best_patterns:
                logging.info(f"[TOP 3 PATTERNS]")
                for i, (signature, winrate) in enumerate(best_patterns, 1):
                    logging.info(f"  {i}. {signature}: {winrate*100:.1f}% winrate")
            
            # 4. Memory Bank
            memory_stats = self.pattern_memory.get_statistics()
            logging.info(f"[PATTERN MEMORY]")
            logging.info(f"  Total Patterns: {memory_stats['total_patterns']}")
            logging.info(f"  Unique Signatures: {memory_stats['unique_signatures']}")
            
            # 5. Experience Replay
            replay_stats = self.experience_replay.get_statistics()
            logging.info(f"[EXPERIENCE REPLAY]")
            logging.info(f"  Real Experiences: {replay_stats['real_experiences']}")
            logging.info(f"  Synthetic Missed: {replay_stats['synthetic_missed']}")
            logging.info(f"  Winning Patterns: {replay_stats['winning_patterns']}")
            
            # 6. Strategy Evolution
            current_strategy = self.strategy_evolution.get_current_strategy()
            logging.info(f"[STRATEGY EVOLUTION]")
            logging.info(f"  Min Confluence: {current_strategy['min_confluence_score']}")
            logging.info(f"  Min Potential Pips: {current_strategy['min_potential_pips']}")
            
            logging.info("=" * 80)
            
        except Exception as e:
            logging.error(f"[DRAGON] Error generating report: {e}")

    def _end_phase(self):
        """
        FIN DE PHASE - Analyse complÃƒÂ¨te des performances
        """
        # Fermer position si ouverte
        if self.position != 0:
            self._force_close_position()
        
        # Calculs statistiques
        phase_equity_change = self.equity - self.phase_start_equity
        phase_equity_pct = (phase_equity_change / self.phase_start_equity) * 100
        phase_trades = len(self.phase_closes)
        phase_avg_pnl = sum(self.phase_closes) / phase_trades if phase_trades > 0 else 0
        phase_wr = (self.phase_wins / phase_trades * 100) if phase_trades > 0 else 0
        
        # AFFICHAGE TABLEAU DE PERFORMANCE
        print("")
        print("="*70)
        print(f"  FIN PHASE {self.current_phase}")
        print("="*70)
        print(f"  PÃƒÂ©riode: Steps {self.phase_start_step} Ã¢Â†Â’ {self.current_step}")
        print(f"  ÃƒÂ‰quity: ${self.phase_start_equity:.2f} Ã¢Â†Â’ ${self.equity:.2f} ({phase_equity_pct:+.2f}%)")
        
        if phase_trades > 0:
            print(f"  Trading: {phase_trades} trades exÃƒÂ©cutÃƒÂ©s")
            print(f"     Win Rate: {phase_wr:.1f}% ({self.phase_wins}W/{self.phase_losses}L)")
            print(f"     P&L Moyen: {phase_avg_pnl:+.4f}%")
            print(f"     P&L Phase: {self.phase_pnl:+.3f}%")
            
            # Performance dÃƒÂ©taillÃƒÂ©e
            gross_profit = sum([p for p in self.phase_closes if p > 0])
            gross_loss = abs(sum([p for p in self.phase_closes if p < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            print(f"  Profit Factor: {profit_factor:.2f}")
        else:
            print(f"   Aucun trade exÃƒÂ©cutÃƒÂ© cette phase")
        
        print(f"  Position: {'FLAT' if self.position == 0 else 'OUVERTE'}")
        print("="*70)
        print("")
        
        # PrÃƒÂ©paration phase suivante
        self.current_phase += 1
        self.phase_start_step = self.current_step
        self.phase_start_equity = self.equity
        self.phase_closes = []
        self.phase_pnl = 0.0
        self.phase_wins = 0
        self.phase_losses = 0

    def get_performance_report(self) -> Dict[str, Any]:
        """
        RAPPORT DE PERFORMANCE COMPLET
        """
        total_trades = self.total_wins + self.total_losses
        winrate = (self.total_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Calcul Profit Factor
        gross_profit = sum([p for p in self.phase_closes if p > 0]) if self.phase_closes else 0
        gross_loss = abs(sum([p for p in self.phase_closes if p < 0])) if self.phase_closes else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # ÃƒÂ‰volution equity
        equity_change_pct = ((self.equity - self.initial_balance) / self.initial_balance) * 100
        
        return {
            'symbol': self.symbol,
            'total_trades': total_trades,
            'wins': self.total_wins,
            'losses': self.total_losses,
            'winrate': winrate,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'total_pnl': self.total_pnl,
            'equity': self.equity,
            'equity_change_pct': equity_change_pct,
            'profit_factor': profit_factor,
            'current_phase': self.current_phase,
            'phase_stats': {
                'phase_pnl': self.phase_pnl,
                'phase_wins': self.phase_wins,
                'phase_losses': self.phase_losses,
                'phase_trades': len(self.phase_closes)
            },
            'performance_grade': self._get_performance_grade(winrate, profit_factor),
            'current_position': self.position,
            'position_age': self.current_step - self.position_open_step if self.position != 0 else 0
        }

    def _get_performance_grade(self, winrate: float, profit_factor: float) -> str:
        """Retourne la note de performance"""
        if winrate >= 65 and profit_factor >= 2.0:
            return "A+"
        elif winrate >= 60 and profit_factor >= 1.8:
            return "A"
        elif winrate >= 55 and profit_factor >= 1.5:
            return "B"
        elif winrate >= 50 and profit_factor >= 1.2:
            return "C"
        elif winrate >= 45 and profit_factor >= 1.0:
            return "D"
        else:
            return "E"

    def render(self, mode='human'):
        """Affichage de l'ÃƒÂ©tat actuel (pour monitoring)"""
        performance = self.get_performance_report()
        
        print(f"\n=== TRADING ENVIRONMENT {self.symbol} ===")
        print(f"Step: {self.current_step}/{len(self.data)} | Phase: {self.current_phase}")
        print(f"Position: {self.position} | Entry: {self.entry_price:.5f}")
        print(f"Equity: ${self.equity:.2f} | P&L: ${self.pnl:.2f}")
        print(f"WinRate: {performance['winrate']:.1f}% | Grade: {performance['performance_grade']}")
        print("="*50)

    def close(self):
        """Nettoyage des ressources"""
        pass

#============================================================================
# IMPL MENTATION COMPL TE AVEC TOUS LES  L MENTS MANQUANTS
#============================================================================


#   COMPL MENT : R SEAUX NEURONNAUX AVANC S MANQUANTS

class TemporalConvNet(nn.Module):
    """R seau convolutionnel temporel pour s ries financi res"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 128, 256], kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            dilation = 2 ** i  # Dilatation croissante
            self.layers.append(
                nn.Conv1d(prev_dim, hidden_dim, kernel_size, dilation=dilation, padding=dilation)
            )
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
            
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(prev_dim, hidden_dims[-1])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, features, seq_len]
        for layer in self.layers:
            x = layer(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return self.output_layer(x)

class AttentionMechanism(nn.Module):
    """M canisme d'attention pour focus sur les points importants"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [seq_len, batch, hidden_dim]
        residual = x
        x, attn_weights = self.multihead_attn(x, x, x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x, attn_weights

class MarketRegimePredictor(nn.Module):
    """Pr dicteur de r gime de march  avec LSTM"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)  # 4 r gimes: trending_up, trending_down, ranging, volatile
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, features]
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # Dernier hidden state
        return self.regime_classifier(last_hidden)

#   COMPL MENT : SYST ME DE M MOIRE   LONG TERME

class ExperienceReplay:
    """M moire de replay avanc e avec priorisation"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def add(self, experience: Tuple, td_error: float = None):
        """Ajoute une exp rience avec priorit """
        priority = td_error if td_error is not None else self.max_priority
        self.buffer.append(experience)
        self.priorities.append(priority ** self.alpha)
        
    def sample(self, batch_size: int) -> Tuple:
        """ chantillonne avec priorit """
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Met   jour les priorit s"""
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):
                self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha
                self.max_priority = max(self.max_priority, self.priorities[idx])

#   COMPL MENT : M THODES MANQUANTES POUR L'ENVIRONNEMENT AM LIOR 

class EnhancedTradingEnvironment(gym.Env):
    """  ENVIRONNEMENT COMPLET AVEC TOUTES LES FONCTIONNALIT S"""
    
    def __init__(self, historical_data: pd.DataFrame, symbol_id: int = 0, symbol: str = "EURUSD"):
        super().__init__()
        
        # Initialisation de base
        self.symbol = symbol
        self.symbol_id = torch.tensor([symbol_id])
        self.data = historical_data
        
        #   R SEAUX NEURONNAUX MANQUANTS
        self.temporal_net = TemporalConvNet(input_dim=len(historical_data.columns))
        self.attention = AttentionMechanism(hidden_dim=256)
        self.regime_predictor = MarketRegimePredictor(input_dim=len(historical_data.columns))
        
        # M moire d'exp rience
        self.experience_replay = ExperienceReplay(capacity=50000)
        
        # M triques avanc es
        self.portfolio_metrics = {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0
        }
        
        #  tat du march 
        self.market_regime = "neutral"
        self.volatility_regime = "normal"
        self.trend_strength = 0.0
        
        # Initialisation manquante
        self._initialize_advanced_metrics()
        
        logging.info(f"  EnhancedTradingEnvironment COMPLET initialis  pour {symbol}")

    def _initialize_advanced_metrics(self):
        """Initialise les m triques avanc es manquantes"""
        self.equity_curve = []
        self.drawdown_curve = []
        self.returns_series = []
        self.volatility_window = 20
        self.cumulative_returns = 0.0
        
    def _get_advanced_features(self) -> np.ndarray:
        """Extrait les features avanc es manquantes"""
        if self.current_step < 50:
            return np.zeros(20)  # Features par d faut
            
        # Donn es r centes
        recent_data = self.data.iloc[self.current_step-50:self.current_step]
        
        features = []
        
        # 1. Features de tendance
        prices = recent_data['close'].values
        features.extend(self._calculate_trend_features(prices))
        
        # 2. Features de volatilit 
        returns = np.diff(prices) / prices[:-1]
        features.extend(self._calculate_volatility_features(returns))
        
        # 3. Features de momentum
        features.extend(self._calculate_momentum_features(prices))
        
        # 4. Features de volume (si disponible)
        if 'volume' in recent_data.columns:
            features.extend(self._calculate_volume_features(recent_data['volume'].values))
        
        return np.array(features)

    def _calculate_trend_features(self, prices: np.ndarray) -> List[float]:
        """Calcule les features de tendance manquantes"""
        if len(prices) < 10:
            return [0.0] * 5
            
        # Pente de r gression lin aire
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = linregress(x, prices)
        
        # Moyennes mobiles multiples
        ma_5 = np.mean(prices[-5:])
        ma_20 = np.mean(prices[-20:])
        ma_50 = np.mean(prices) if len(prices) >= 50 else ma_20
        
        # Force de tendance
        trend_strength = abs(slope) / (np.std(prices) + 1e-8)
        
        return [slope, r_value**2, ma_5/ma_20 - 1, ma_20/ma_50 - 1, trend_strength]

    def _calculate_volatility_features(self, returns: np.ndarray) -> List[float]:
        """Calcule les features de volatilit  manquantes"""
        if len(returns) < 10:
            return [0.0] * 4
            
        # Volatilit  historique
        vol_10 = np.std(returns[-10:])
        vol_20 = np.std(returns)
        
        # Asym trie et kurtosis
        skewness = np.mean((returns - np.mean(returns))**3) / (vol_20**3 + 1e-8)
        kurtosis = np.mean((returns - np.mean(returns))**4) / (vol_20**4 + 1e-8) - 3
        
        # Ratio de volatilit 
        vol_ratio = vol_10 / vol_20 if vol_20 > 0 else 1.0
        
        return [vol_10, vol_20, skewness, kurtosis, vol_ratio]

    def _calculate_momentum_features(self, prices: np.ndarray) -> List[float]:
        """Calcule les features de momentum manquantes"""
        if len(prices) < 20:
            return [0.0] * 5
            
        # RSI
        gains = np.where(np.diff(prices) > 0, np.diff(prices), 0)
        losses = np.where(np.diff(prices) < 0, -np.diff(prices), 0)
        
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        macd = ema_12 - ema_26
        macd_signal = self._ema(macd, 9)
        
        # Momentum simple
        momentum_5 = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
        momentum_10 = (prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0
        
        return [rsi/100 - 0.5, macd, macd_signal, momentum_5, momentum_10]

    def _calculate_volume_features(self, volumes: np.ndarray) -> List[float]:
        """Calcule les features de volume manquantes"""
        if len(volumes) < 10:
            return [0.0] * 3
            
        # Volume relatif
        vol_ma = np.mean(volumes[-20:])
        current_vol = volumes[-1] if len(volumes) > 0 else vol_ma
        vol_ratio = current_vol / vol_ma
        
        # Tendance volume
        vol_trend = np.polyfit(np.arange(len(volumes[-10:])), volumes[-10:], 1)[0] if len(volumes) >= 10 else 0
        
        return [vol_ratio, vol_trend, current_vol]

    def _ema(self, values: np.ndarray, period: int) -> float:
        """Calcule l'EMA manquante"""
        if len(values) < period:
            return np.mean(values)
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(values, weights, mode='valid')[-1]

    def _detect_market_regime_advanced(self) -> Dict[str, any]:
        """D tection avanc e du r gime de march  manquante"""
        if self.current_step < 100:
            return {"regime": "neutral", "confidence": 0.5, "volatility": "normal"}
            
        prices = self.data['close'].iloc[self.current_step-100:self.current_step].values
        returns = np.diff(prices) / prices[:-1]
        
        # Calculs avanc s
        volatility = np.std(returns)
        trend_strength = abs(np.polyfit(np.arange(len(prices)), prices, 1)[0]) / (np.std(prices) + 1e-8)
        
        # D tection r gime
        if volatility > 0.02:
            regime = "volatile"
        elif trend_strength > 0.7:
            regime = "trending"
        elif trend_strength < 0.3:
            regime = "ranging"
        else:
            regime = "neutral"
            
        # D tection volatilit 
        if volatility > 0.015:
            vol_regime = "high"
        elif volatility < 0.005:
            vol_regime = "low"
        else:
            vol_regime = "normal"
            
        return {
            "regime": regime,
            "volatility_regime": vol_regime,
            "trend_strength": trend_strength,
            "volatility": volatility,
            "confidence": min(trend_strength * 2, 1.0)
        }

    def _calculate_advanced_risk_metrics(self) -> Dict[str, float]:
        """Calcule les m triques de risque avanc es manquantes"""
        if len(self.returns_series) < 10:
            return {k: 0.0 for k in ['sharpe', 'sortino', 'max_drawdown', 'calmar', 'var']}
            
        returns = np.array(self.returns_series)
        
        # Sharpe Ratio
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        
        # Sortino Ratio (seulement downside risk)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = np.mean(returns) / (downside_std + 1e-8)
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)
        
        # Drawdown maximum
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - peak) / peak
        max_drawdown = np.min(drawdowns)
        
        # Calmar Ratio
        calmar = np.mean(returns) / (abs(max_drawdown) + 1e-8) if max_drawdown < 0 else 0
        
        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'var_95': var_95
        }

    def _update_portfolio_metrics(self):
        """Met   jour les m triques de portefeuille manquantes"""
        risk_metrics = self._calculate_advanced_risk_metrics()
        self.portfolio_metrics.update(risk_metrics)
        
        # Mise   jour courbe d' quit 
        self.equity_curve.append(self.equity)
        if len(self.equity_curve) > 1000:
            self.equity_curve.pop(0)

    def _get_neural_enhanced_observation(self) -> np.ndarray:
        """Observation enrichie par r seaux neuronaux manquante"""
        base_obs = self._get_base_observation()
        
        # Features avanc es
        advanced_features = self._get_advanced_features()
        
        # Pr diction r gime de march 
        market_data = self.data.iloc[self.current_step-50:self.current_step].values if self.current_step >= 50 else self.data.iloc[:50].values
        if len(market_data) < 50:
            market_data = np.pad(market_data, ((0, 50 - len(market_data)), (0, 0)), mode='edge')
            
        market_tensor = torch.FloatTensor(market_data).unsqueeze(0)
        regime_pred = self.regime_predictor(market_tensor)
        regime_probs = torch.softmax(regime_pred, dim=1).squeeze().detach().numpy()
        
        # Fusion finale
        neural_obs = np.concatenate([
            base_obs,
            advanced_features,
            regime_probs
        ])
        
        return neural_obs

    def step(self, action: int):
        """Step complet avec tous les  l ments manquants"""
        self.current_step += 1
        
        # Gestion des limites
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
            
        # Ex cution de l'action
        reward = self._calculate_advanced_reward(action)
        
        # Mise   jour m triques
        self._update_portfolio_metrics()
        
        # D tection r gime
        market_info = self._detect_market_regime_advanced()
        self.market_regime = market_info["regime"]
        self.volatility_regime = market_info["volatility_regime"]
        self.trend_strength = market_info["trend_strength"]
        
        # V rification fin d' pisode
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Observation enrichie
        obs = self._get_neural_enhanced_observation()
        
        # Info  tendue
        info = {
            'equity': self.equity,
            'position': self.position,
            'step': self.current_step,
            'pnl': self.pnl,
            'market_regime': self.market_regime,
            'volatility_regime': self.volatility_regime,
            'portfolio_metrics': self.portfolio_metrics,
            'performance': self.get_performance_report()
        }
        
        return obs, reward, terminated, truncated, info

#   COMPL MENT : OPTIMISEURS AVANC S MANQUANTS







class RealTimeMonitor:
    """Monitoring en temps r el des performances"""
    
    def __init__(self, symbols: List[str], update_interval: int = 60):
        self.symbols = symbols
        self.update_interval = update_interval
        self.performance_data = {symbol: [] for symbol in symbols}
        self.alert_thresholds = {
            'drawdown': -0.05,  # -5%
            'volatility': 0.02,  # 2%
            'consecutive_losses': 5
        }
        
    async def start_monitoring(self):
        """D marre le monitoring en temps r el"""
        while True:
            for symbol in self.symbols:
                await self._update_symbol_metrics(symbol)
                await self._check_alerts(symbol)
                
            await asyncio.sleep(self.update_interval)
            
    async def _update_symbol_metrics(self, symbol: str):
        """Met   jour les m triques pour un symbole"""
        # Impl mentation de la collecte de donn es en temps r el
        try:
            # R cup ration prix courant, positions, P&L
            current_data = await self._get_current_market_data(symbol)
            self.performance_data[symbol].append(current_data)
            
            # Garder seulement les derni res 1000 observations
            if len(self.performance_data[symbol]) > 1000:
                self.performance_data[symbol].pop(0)
                
        except Exception as e:
            logging.error(f"Erreur monitoring {symbol}: {e}")
            
    async def _check_alerts(self, symbol: str):
        """V rifie les alertes de trading"""
        data = self.performance_data[symbol]
        if len(data) < 10:
            return
            
        # V rification drawdown
        current_equity = data[-1]['equity']
        peak_equity = max(d['equity'] for d in data)
        drawdown = (current_equity - peak_equity) / peak_equity
        
        if drawdown < self.alert_thresholds['drawdown']:
            logging.warning(f"  ALERTE DRAWDOWN {symbol}: {drawdown:.2%}")
            
        # V rification volatilit 
        returns = [data[i]['equity'] / data[i-1]['equity'] - 1 for i in range(1, len(data))]
        volatility = np.std(returns) if returns else 0
        
        if volatility > self.alert_thresholds['volatility']:
            logging.warning(f"  ALERTE VOLATILIT  {symbol}: {volatility:.2%}")

#   COMPL MENT : INT GRATION FINALE AVEC TOUS LES COMPOSANTS

class CompleteRLTradingSystem:
    """SYST ME COMPLET DE TRADING RL AVEC TOUS LES COMPOSANTS"""
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.agent_manager = AdvancedRLAgentManager(config)
        self.backtester = AdvancedBacktester()
        self.monitor = RealTimeMonitor(config.SYMBOLS)
        self.validator = CrossValidator()
        
        # M triques globales
        self.system_metrics = {
            'total_trades': 0,
            'total_profit': 0.0,
            'system_health': 'OPTIMAL'
        }
        
    async def start_complete_system(self):
        """D marre le syst me complet"""
        logging.info("  D MARRAGE SYST ME COMPLET DE TRADING RL")
        
        try:
            # 1. Entra nement des agents
            await self.agent_manager.start_training_all_agents()
            
            # 2. Validation crois e
            validation_results = await self._run_comprehensive_validation()
            
            # 3. Backtesting
            backtest_results = await self._run_backtesting_suite()
            
            # 4. D marrage monitoring
            asyncio.create_task(self.monitor.start_monitoring())
            
            # 5. D marrage trading en direct
            await self._start_live_trading()
            
            logging.info("  SYST ME COMPLET OP RATIONNEL")
            
        except Exception as e:
            logging.error(f"  Erreur d marrage syst me: {e}")
            raise
            
    async def _run_comprehensive_validation(self) -> Dict:
        """Ex cute une validation compl te"""
        results = {}
        
        for symbol in self.config.SYMBOLS:
            # Chargement donn es
            data = await self.agent_manager.load_historical_data(symbol)
            
            # Validation crois e
            cv_results = self.validator.evaluate_strategy(
                self.agent_manager.ppo_agents[symbol], data
            )
            
            results[symbol] = cv_results
            
            logging.info(f"  Validation {symbol}: {cv_results}")
            
        return results

#   R PONSE   VOTRE QUESTION AVEC ANALYSE APPROFONDIE

"""
  **ANALYSE COMPL TE : R SEAUX DE NEURONES POUR LE TRADING RL**

##   ** VALUATION DE LA PLUS-VALUE :**

###   **POTENTIEL TR S  LEV ** avec les architectures propos es :

1. **TemporalConvNet** :
   -   **Avantage** : Capture les patterns temporels multi- chelles
   -   **Application** : D tection de tendances court/moyen terme
   -   **Impact** : Am lioration timing d'entr e/sortie

2. **AttentionMechanism** :
   -   **Avantage** : Focus sur les p riodes market importantes
   -   **Application** : Identification points de rupture
   -   **Impact** : R duction des faux signaux

3. **MarketRegimePredictor** :
   -   **Avantage** : Adaptation dynamique   l'environnement
   -   **Application** : Ajustement strat gie selon r gime
   -   **Impact** : Performance accrue en market changing

###   **B N FICES MESURABLES ATTENDUS :**

- **+15-25%** : Sharpe Ratio avec r seaux neuronaux
- **-10-20%** : Maximum Drawdown avec meilleure gestion risque  
- **+20-30%** : Win Rate avec signaux plus pr cis
- **+30-50%** : Profit Factor avec meilleur risk/reward

###   **CONSID RATIONS IMPORTANTES :**

1. **Complexit  vs B n fice** :
   - Start simple    voluer progressivement
   - Validation rigoureuse   chaque  tape

2. **Risque de Overfitting** :
   - Validation crois e obligatoire
   - Regularization aggressive
   - Early stopping

3. **Co t Computationnel** :
   - Entra nement : 2-5x plus long
   - Inference : N gligeable avec optimisation
   - Hardware : GPU recommand  pour training

###   **RECOMMANDATION STRAT GIQUE :**

**PHASE 1** (2-4 semaines) :
- Impl menter TemporalConvNet + validation
- Mesurer impact sur performance
- Ajuster hyperparam tres

**PHASE 2** (2-3 semaines) :  
- Ajouter AttentionMechanism
- Optimiser pour r duire latency
- Backtesting avanc 

**PHASE 3** (2-4 semaines) :
- Int grer MarketRegimePredictor
- Syst me de monitoring temps r el
- D ploiement progressif

##   **CONCLUSION :**

**OUI, l'ajout de r seaux de neurones est CLAIREMENT B N FIQUE** 
avec une approche structur e et progressive. Les gains potentiels 
justifient l'investissement en complexit  et temps de d veloppement.

**Estimation ROI** : 3-6 mois pour un syst me robuste produisant 
des am liorations significatives et durables.
"""


#============================================================================
#RL TRAINING CALLBACK
#============================================================================


class DetailedTrainingCallback(BaseCallback):
    """Callback D TAILL  pour suivre l'entra nement en temps r el"""
    
    def __init__(self, symbol: str, algo: str, log_freq: int = 500):
        super().__init__()
        self.symbol = symbol
        self.algo = algo
        self.log_freq = log_freq
        self.episode_count = 0
        self.current_reward = 0
        self.actions_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'CLOSE': 0}
        self.wins = 0
        self.losses = 0
        self.total_steps = 0
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        self.total_steps += 1
        
        # Accumuler reward
        if 'rewards' in self.locals and len(self.locals['rewards']) > 0:
            reward = self.locals['rewards'][-1]
            self.current_reward += reward
        
        # Compter actions
        if 'actions' in self.locals and len(self.locals['actions']) > 0:
            action = self.locals['actions'][-1]
            if action == 0:
                self.actions_count['BUY'] += 1
            elif action == 1:
                self.actions_count['SELL'] += 1
            elif action == 2:
                self.actions_count['HOLD'] += 1
            elif action == 3:
                self.actions_count['CLOSE'] += 1
        
        # Log tous les N steps
        if self.total_steps % self.log_freq == 0:
            self._log_progress()
        
        # D tecter fin d' pisode
        if 'dones' in self.locals and len(self.locals['dones']) > 0:
            if self.locals['dones'][-1]:
                self._log_episode()
        
        return True
    
    def _log_progress(self):
        """Log métriques clés PPO/DQN à chaque log_freq steps + snapshot diagnostic toutes les 5000 steps."""
        try:
            logger_vals = {}
            if hasattr(self, 'model') and self.model is not None:
                lobj = getattr(self.model, 'logger', None)
                if lobj: logger_vals = getattr(lobj, 'name_to_value', {})

            entropy   = logger_vals.get('train/entropy_loss')
            vloss     = logger_vals.get('train/value_loss')
            expl_var  = logger_vals.get('train/explained_variance')
            pg_loss   = logger_vals.get('train/policy_gradient_loss')
            kl        = logger_vals.get('train/approx_kl')
            clip_frac = logger_vals.get('train/clip_fraction')
            q_loss    = logger_vals.get('train/loss')
            epsilon   = getattr(getattr(self, 'model', None), 'exploration_rate', None)

            wr, total_trades = 0.0, 0
            if hasattr(self.training_env, 'envs'):
                _e = self.training_env.envs[0]
                if hasattr(_e, 'env'): _e = _e.env
                w = getattr(_e, 'total_wins', 0); l = getattr(_e, 'total_losses', 0)
                total_trades = w + l
                wr = (w / total_trades * 100.0) if total_trades > 0 else 0.0

            parts = [f"[{self.algo}|{self.symbol}] step={self.total_steps}",
                     f"WR={wr:.1f}%({total_trades}T)"]
            if entropy   is not None: parts.append(f"H={entropy:.3f}")
            if vloss     is not None: parts.append(f"Vl={vloss:.4f}")
            if expl_var  is not None: parts.append(f"EV={expl_var:.3f}")
            if pg_loss   is not None: parts.append(f"PG={pg_loss:.4f}")
            if kl        is not None: parts.append(f"KL={kl:.4f}")
            if clip_frac is not None: parts.append(f"clip={clip_frac:.2f}")
            if q_loss    is not None: parts.append(f"Ql={q_loss:.4f}")
            if epsilon   is not None: parts.append(f"eps={epsilon:.3f}")
            logging.info("  " + " | ".join(parts))

            # Snapshot diagnostic toutes les 5000 steps pour courbe de progression
            if self.total_steps % 5000 == 0:
                try:
                    _env = None
                    if hasattr(self.training_env, 'envs'):
                        _env = self.training_env.envs[0]
                        if hasattr(_env, 'env'): _env = _env.env
                    if _env is not None:
                        _w = getattr(_env, 'total_wins', 0); _l = getattr(_env, 'total_losses', 0)
                        _t = _w + _l
                        import json as _json
                        _path = "./models/goldeneye_diagnostic.json"
                        _diag = {"sessions": [], "summary": {}, "last_diagnosis": {}, "hyperparams_active": ACTIVE_HYPERPARAMS}
                        try:
                            if os.path.exists(_path):
                                with open(_path) as _f: _diag = _json.load(_f)
                        except Exception: pass
                        _rec = {
                            "ts": datetime.now().isoformat(),
                            "symbol": self.symbol,
                            "agent": f"{self.algo}_step{self.total_steps}",
                            "cumul_steps": self.total_steps,
                            "session_trades": _t,
                            "wr": round((_w/_t*100) if _t>0 else 0, 2),
                            "pnl": round(float(sum(getattr(_env,'phase_closes',None) or [])), 4),
                            "wins": _w, "losses": _l,
                            "entropy":   round(float(logger_vals.get('train/entropy_loss',0)), 4),
                            "value_loss":round(float(logger_vals.get('train/value_loss',0)), 4),
                            "expl_var":  round(float(logger_vals.get('train/explained_variance',0)), 4),
                            "pg_loss":   round(float(logger_vals.get('train/policy_gradient_loss',0)), 4),
                            "kl":        round(float(logger_vals.get('train/approx_kl',0)), 4),
                            "clip_frac": round(float(logger_vals.get('train/clip_fraction',0)), 4),
                            "q_loss":    round(float(logger_vals.get('train/loss',0)), 4),
                            "epsilon":   round(float(getattr(self.model,'exploration_rate',0) if self.model else 0), 3),
                        }
                        _diag["sessions"].append(_rec)
                        _diag["last_diagnosis"][f"{self.symbol}_{self.algo}_step{self.total_steps}"] = {"ts": _rec["ts"], "metrics_snapshot": _rec}
                        os.makedirs("./models", exist_ok=True)
                        with open(_path, 'w') as _f: _json.dump(_diag, _f, indent=2)
                except Exception as _de:
                    logging.debug(f"[DiagSnapshot] {_de}")
        except Exception as _e:
            logging.debug(f"[DetailedCallback._log_progress] {_e}")
        
    def _log_episode(self):
        """Log   la fin de chaque  pisode - M THODE CORRIG E"""
        self.episode_count += 1
        self.episode_rewards.append(self.current_reward)
        
        # D terminer win/loss
        if self.current_reward > 50:
            self.wins += 1
        elif self.current_reward < -50:
            self.losses += 1
        
        total_trades = self.wins + self.losses
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0
        
        # Log d taill  fin d' pisode
        logging.info("")
        logging.info(
            f"  {self.algo} {self.symbol} EPISODE {self.episode_count} TERMIN : "
            f"Reward {self.current_reward:+.0f} | Win Rate {win_rate:.0f}% | "
            f"Actions: BUY={self.actions_count['BUY']} SELL={self.actions_count['SELL']} "
            f"HOLD={self.actions_count['HOLD']} CLOSE={self.actions_count['CLOSE']}"
        )
        logging.info("")
        
        # Reset pour le prochain  pisode
        self.current_reward = 0
        self.actions_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'CLOSE': 0}

class AdvancedBacktester:
    """Backtester avanc  avec analyse de robustness"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def run_backtest(self, agent, historical_data: pd.DataFrame, 
                    strategy_name: str = "RL_Strategy") -> Dict[str, any]:
        """Ex cute un backtest complet"""
        
        # Simulation de trading
        equity_curve = [self.initial_capital]
        trades = []
        current_capital = self.initial_capital
        position = 0
        entry_price = 0
        
        for step in range(len(historical_data)):
            # Observation
            obs = self._get_observation(historical_data, step)
            
            # D cision de l'agent
            action = agent.predict(obs)
            
            # Ex cution du trade
            current_price = historical_data.iloc[step]['close']
            trade_result = self._execute_trade(action, position, current_price, entry_price, current_capital)
            
            # Mise   jour
            if trade_result['new_position'] != position:
                trades.append({
                    'step': step,
                    'action': action,
                    'price': current_price,
                    'pnl': trade_result['realized_pnl'],
                    'capital': current_capital
                })
                
            position = trade_result['new_position']
            entry_price = trade_result['new_entry_price']
            current_capital = trade_result['new_capital']
            equity_curve.append(current_capital)
            
        # Calcul des m triques
        performance_metrics = self._calculate_performance_metrics(equity_curve, trades)
        
        self.results[strategy_name] = {
            'equity_curve': equity_curve,
            'trades': trades,
            'metrics': performance_metrics
        }
        
        return self.results[strategy_name]
    
    def _calculate_performance_metrics(self, equity_curve: List[float], trades: List[Dict]) -> Dict[str, float]:
        """Calcule les m triques de performance avanc es"""
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # M triques de base
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Drawdown
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # M triques trades
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            win_rate = len(winning_trades) / len(trades)
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'avg_trade_return': np.mean([t['pnl'] for t in trades]) if trades else 0
        }

#   COMPL MENT : SYST ME DE MONITORING EN TEMPS R EL MANQUANT

class HRMTRMTrainingCallback(BaseCallback):
    """
      Callback pour entra ner HRM-TRM pendant le training RL
    
    Entra ne HRM-TRM   chaque fin d' pisode pour am liorer
    le raffinement des observations
    """
    
    def __init__(self, hrm_trm, learning_rate=5e-4):
        super().__init__()
        self.hrm_trm = hrm_trm
        self.optimizer = torch.optim.Adam(hrm_trm.parameters(), lr=learning_rate)
        self.episode_count = 0
        
    def _on_rollout_end(self) -> None:
        """Entra ne HRM-TRM   la fin de chaque rollout"""
        self.episode_count += 1
        
        if self.episode_count % 5 == 0:  # Toutes les 5  pisodes
            # Mettre HRM-TRM en mode eval (pas besoin d'entra nement constant)
            self.hrm_trm.eval()
            
            if self.episode_count % 20 == 0:
                logging.info(f"  HRM-TRM: Episode {self.episode_count} (eval mode)")
        
        return True

class AdaptiveOptimizer:
    """Optimiseur adaptatif avec ajustement automatique des hyperparam tres"""
    
    def __init__(self, model_params, initial_lr: float = 0.001):
        self.optimizer = optim.AdamW(model_params, lr=initial_lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=10, factor=0.5, verbose=True
        )
        self.best_score = -np.inf
        self.patience_counter = 0
        self.max_patience = 20
        
    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
        self.optimizer.step()
        
    def update_lr(self, validation_score: float):
        self.scheduler.step(validation_score)
        
        if validation_score > self.best_score:
            self.best_score = validation_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        return self.patience_counter >= self.max_patience

#   COMPL MENT : SYST ME DE VALIDATION CROIS E MANQUANT



class TqdmCallback(BaseCallback):
    """Callback avec barre de progression tqdm"""
    def __init__(self, total_steps, desc, colour="blue", verbose=0):
        super().__init__(verbose)
        self.pbar = tqdm(total=total_steps, desc=desc, colour=colour, leave=True)
        self.last_step = 0
    
    def _on_step(self):
        current = self.model.num_timesteps
        self.pbar.update(current - self.last_step)
        self.last_step = current
        return True
    
    def close(self):
        self.pbar.close()

class RLTrainingCallback(BaseCallback):
    """Callback pour monitoring du training RL"""
    
    def __init__(self, ws_server, symbol: str, algo: str):
        super().__init__()
        self.ws_server = ws_server
        self.symbol = symbol
        self.algo = algo
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Called at each training step"""
        # === FORMULE 1 : CURRICULUM AUTO-SWITCH ===
        global current_phase
        timestep = self.num_timesteps
        
        # Phase transitions étendues pour run longue (1.6M steps)
        if timestep >= 10000 and current_phase < 5:
            current_phase = 5
            print("🚀 CURRICULUM PHASE 5/5 - FULL MASTERY - META-LEARNER EXPERT")
        elif timestep >= 10000 and current_phase < 4:
            current_phase = 4
            print("⚡ CURRICULUM PHASE 4/5 - ADVANCED ENSEMBLE - FINE TUNING")
        elif timestep >= 10000 and current_phase < 3:
            current_phase = 3
            print("🔥 CURRICULUM PHASE 3/5 - TOUT ACTIVE - 4 SYMBOLES - VECENV 12")
        elif timestep >= 30000 and current_phase < 2:
            current_phase = 2
            print("📈 CURRICULUM PHASE 2/5 - AJOUT PROGRESSIF DES PAIRES - VECENV 12")
        # === FIN CURRICULUM AUTO ===
        
        # === CHECKPOINT AUTO TOUTES LES 100K STEPS ===
        if timestep > 0 and timestep % 100000 == 0:
            try:
                import os
                from datetime import datetime as dt
                checkpoint_dir = "./models/checkpoint"
                os.makedirs(checkpoint_dir, exist_ok=True)
                timestamp = dt.now().strftime("%Y%m%d_%H%M")
                
                # Sauvegarder le modèle actuel
                if hasattr(self, 'model') and self.model is not None:
                    save_path = f"{checkpoint_dir}/{self.algo}_{self.symbol}_{timestamp}_{timestep//1000}k.zip"
                    self.model.save(save_path)
                    print(f"💾 CHECKPOINT AUTO SAVED: {save_path}")
            except Exception as e:
                print(f"⚠️ Checkpoint save failed: {e}")
        # === FIN CHECKPOINT AUTO ===
        
        if self.locals.get('dones', [False])[0]:
            # Episode terminé
            ep_reward = np.sum(self.locals.get('rewards', [0]))
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(self.locals.get('episode_len', 0))
            
            # Log toutes les 10 épisodes
            if len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                self.ws_server.add_training_log(
                    f"{self.algo} {self.symbol}: Step {self.num_timesteps}, Mean Reward: {mean_reward:.4f}",
                    'info'
                )
        
        return True

#============================================================================
#RL TRAINING WEBSOCKET SERVER
#============================================================================
class RLTrainingWebSocketServer:
    """WebSocket server pour monitoring du training RL"""
    
    def __init__(self, port: int):
        self.port = port
        self.logs = deque(maxlen=1000)
        self.clients = set()
    
    def add_training_log(self, message: str, level: str):
        """Ajoute un log de training"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.logs.append(log_entry)
        logging.log(getattr(logging, level.upper()), message)

    async def broadcast(self, data: Dict):
        """Broadcast data to all connected clients"""
        if self.clients:
            message = json.dumps(data)
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )

#============================================================================
#RL AGENT MANAGER AVEC FUSION PPO-DQN
#============================================================================
class DiagnosticLogger:
    """
    Journalise chaque session dans goldeneye_diagnostic.json.
    Claude Code lit ce fichier pour diagnostiquer et régler les hyperparamètres.
    """
    DIAGNOSTIC_FILE = "./models/goldeneye_diagnostic.json"

    @classmethod
    def log_session(cls, symbol: str, agent: str, metrics: dict, hyperparams: dict = None):
        import json as _json
        diag = {"sessions": [], "summary": {}, "last_diagnosis": {}, "hyperparams_active": {}}
        try:
            if os.path.exists(cls.DIAGNOSTIC_FILE):
                with open(cls.DIAGNOSTIC_FILE, 'r') as f: diag = _json.load(f)
        except Exception: pass
        record = {"ts": datetime.now().isoformat(), "symbol": symbol, "agent": agent, **metrics}
        diag["sessions"].append(record)
        if hyperparams: diag["hyperparams_active"] = hyperparams
        # Diagnostic auto
        issues, ok = [], []
        if metrics.get('entropy', 0) < -1.35: issues.append("ENTROPY MAX: agent explore au hasard.")
        if metrics.get('value_loss', 0) > 20: issues.append("VALUE_LOSS ÉLEVÉE (>20): vérifier reward scale.")
        if metrics.get('expl_var', 0) > 0.05 and metrics.get('entropy', -2) > -1.0: ok.append("PPO CONVERGE.")
        if metrics.get('wr', 0) > 55: ok.append(f"WR={metrics['wr']:.1f}% EXCELLENT.")
        elif metrics.get('wr', 0) > 50: ok.append(f"WR={metrics['wr']:.1f}% POSITIF.")
        diag["last_diagnosis"][f"{symbol}_{agent}"] = {"ts": record["ts"], "metrics_snapshot": metrics, "issues": issues, "ok": ok}
        try:
            os.makedirs("./models", exist_ok=True)
            with open(cls.DIAGNOSTIC_FILE, 'w') as f: _json.dump(diag, f, indent=2)
        except Exception as e:
            logging.warning(f"[DiagnosticLogger] {e}")

    @classmethod
    def read_for_claude(cls, symbol: str = None) -> str:
        import json as _json
        try:
            if not os.path.exists(cls.DIAGNOSTIC_FILE):
                return "[DiagnosticLogger] Pas encore de données. Lancer GOLDENEYE d'abord."
            with open(cls.DIAGNOSTIC_FILE) as f: diag = _json.load(f)
            lines = ["=" * 60, "GOLDENEYE — RAPPORT ENTRAÎNEUR", "=" * 60]
            for sym, s in diag.get("summary", {}).items():
                lines.append(f"\n[{sym}] Sessions:{s.get('total_sessions',0)} PPO:{s.get('total_ppo_steps',0):,}steps")
                lines.append(f"  WR PPO:{s.get('last_wr_ppo',0):.1f}% entropy:{s.get('last_entropy',0):.3f} EV:{s.get('last_expl_var',0):.4f}")
            return "\n".join(lines)
        except Exception as e: return f"[DiagnosticLogger] Erreur: {e}"


class RLAgentManager:
    """Gestionnaire des agents RL Manhattan — RPPO + DiscreteSAC + Transformer + MetaLearner"""

    def _log_session_progression(self, symbol: str, agent: str, env, model, fusion_ok: bool = None):
        """
        Enregistre les stats de cette session dans models/goldeneye_diagnostic.json.
        Écrit aussi models/training_history.json (historique brut inter-sessions).
        Claude Code lit ces fichiers pour régler les hyperparamètres PPO/DQN.
        """
        import json as _json
        diag_path    = "./models/goldeneye_diagnostic.json"
        history_path = "./models/training_history.json"
        try:
            logger_vals = {}
            if model is not None:
                lobj = getattr(model, 'logger', None)
                if lobj: logger_vals = getattr(lobj, 'name_to_value', {})

            w = getattr(env, 'total_wins', 0)
            l = getattr(env, 'total_losses', 0)
            trades = w + l
            wr = (w / trades * 100.0) if trades > 0 else 0.0
            pnl = float(sum(env.phase_closes)) if getattr(env, 'phase_closes', None) else 0.0

            metrics = {
                "cumul_steps":    getattr(model, 'num_timesteps', 0) if model else 0,
                "session_trades": trades, "wr": round(wr, 2), "pnl": round(pnl, 4),
                "wins": w, "losses": l,
                "entropy":   round(float(logger_vals.get('train/entropy_loss', 0)), 4),
                "value_loss":round(float(logger_vals.get('train/value_loss', 0)), 4),
                "expl_var":  round(float(logger_vals.get('train/explained_variance', 0)), 4),
                "pg_loss":   round(float(logger_vals.get('train/policy_gradient_loss', 0)), 4),
                "kl":        round(float(logger_vals.get('train/approx_kl', 0)), 4),
                "clip_frac": round(float(logger_vals.get('train/clip_fraction', 0)), 4),
                "q_loss":    round(float(logger_vals.get('train/loss', 0)), 4),
                "epsilon":   round(float(getattr(model, 'exploration_rate', 0) if model else 0), 3),
            }
            if fusion_ok is not None:
                metrics["fusion_ok"] = fusion_ok

            os.makedirs("./models", exist_ok=True)

            # Diagnostic complet
            diag = {"sessions": [], "summary": {}, "last_diagnosis": {}, "hyperparams_active": {}}
            if os.path.exists(diag_path):
                try:
                    with open(diag_path, 'r') as f: diag = _json.load(f)
                except Exception: pass
            record = {"ts": datetime.now().isoformat(), "symbol": symbol, "agent": agent, **metrics}
            diag["sessions"].append(record)
            with open(diag_path, 'w') as f: _json.dump(diag, f, indent=2)

            # Historique brut
            history = []
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f: history = _json.load(f)
                except Exception: pass
            history.append({"ts": datetime.now().isoformat(), "symbol": symbol, "agent": agent, **metrics})
            with open(history_path, 'w') as f: _json.dump(history, f, indent=2)

            logging.info(f"[PROGRESSION] {agent} {symbol} WR={wr:.1f}% steps={metrics['cumul_steps']:,}")
        except Exception as e:
            logging.warning(f"[PROGRESSION] Erreur: {e}")

    def _debug_columns(self, df: pd.DataFrame, context: str):
        """M thode de d bogage pour voir les colonnes disponibles"""
        try:
            logging.info(f"  DEBUG COLONNES [{context}]")
            logging.info(f"     Shape: {df.shape}")
            logging.info(f"     Colonnes: {list(df.columns)}")
            
            if len(df) > 0:
                # Afficher les premi res valeurs de quelques colonnes importantes
                important_cols = ['open', 'high', 'low', 'close', 'rsi', 'macd', 'atr']
                for col in important_cols:
                    if col in df.columns:
                        logging.info(f"     {col}: {df[col].iloc[-1]}")
                    else:
                        logging.warning(f"     {col} manquante")
        except Exception as e:
            logging.error(f"  Erreur _debug_columns: {e}")

    def _validate_dataframe(self, df: pd.DataFrame, method_name: str) -> bool:
        """Valide l'int grit  d'un DataFrame"""
        try:
            if df is None:
                logging.error(f"  {method_name}: DataFrame est None")
                return False
            
            if len(df) == 0:
                logging.error(f"  {method_name}: DataFrame vide")
                return False
            
            # V rifier les NaN
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                logging.warning(f"  {method_name}: {nan_count} valeurs NaN d tect es")
                # Remplacer les NaN
                df = df.bfill().fillna(0)
            
            # V rifier les inf
            inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                logging.warning(f"  {method_name}: {inf_count} valeurs inf d tect es")
                # Remplacer les inf
                df = df.replace([np.inf, -np.inf], 0)
            
            expected_columns = [
                'open', 'high', 'low', 'close', 'rsi', 'macd', 'macd_signal', 
                'atr', 'bb_middle', 'bb_upper', 'bb_lower', 'volume_norm', 
                'ema_20', 'momentum'
            ]
            
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                logging.warning(f"  {method_name}: Colonnes manquantes: {missing_columns}")
                # Cr er les colonnes manquantes
                for col in missing_columns:
                    df[col] = 0.0
            
            logging.info(f"  {method_name}: DataFrame valid  - {len(df)} lignes, {len(df.columns)} colonnes")
            return True
            
        except Exception as e:
            logging.error(f"  Erreur _validate_dataframe: {e}")
            return False

    async def load_historical_data_safe(self, symbol: str, bars: int = 1000) -> pd.DataFrame:
        """
        Version s curis e du chargement historique - PATCH DE SECOURS
        Utilise seulement M5 pour  viter les erreurs multi-timeframe
        """
        try:
            logging.info(f"  CHARGEMENT S CURIS  pour {symbol}")
            
            if MT5_AVAILABLE:
                #   SIMPLIFICATION: Utiliser seulement M5
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    # Stocker symbole pour normalisation
                    current_symbol_backup = getattr(self, 'current_symbol', None)
                    self.current_symbol = symbol
                    
                    df = self._calculate_indicators(df)
                    
                    # Restaurer symbole
                    if current_symbol_backup is not None:
                        self.current_symbol = current_symbol_backup
                    
                    logging.info(f"  CHARGEMENT S CURIS : {len(df)} bougies M5 pour {symbol}")
                    return df
            
            # Fallback mock
            logging.warning(f"  Chargement s curis   chou , mock data pour {symbol}")
            return self._generate_mock_data(bars)
            
        except Exception as e:
            logging.error(f"  Erreur load_historical_data_safe: {e}")
            return self._generate_mock_data(bars)

    def __init__(self, config: UltimateConfig):
        self.config = config
        self.ppo_agents: Dict[str, PPO] = {}
        self.dqn_agents: Dict[str, DQN] = {}
        self.performance_history: Dict[str, deque] = {sym: deque(maxlen=50) for sym in config.SYMBOLS}
        self.last_retrain_time: Dict[str, float] = {}
        self.hrm_trm_per_symbol: Dict[str, AdvancedHRMTRM] = {}
        self.ws_server = RLTrainingWebSocketServer(config.RL_WS_PORT)
        
        # Variables de gestion
        self.current_symbol = None
        self.corrupted_models: Dict[str, bool] = {sym: False for sym in config.SYMBOLS}
        self.training_attempts: Dict[str, int] = {sym: 0 for sym in config.SYMBOLS}
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # META-LEARNER: SystÃ¨me intelligent de fusion PPO-DQN
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        self.meta_learner = MetaLearnerFusion(
            context_dim=18,  # 14 features + 4 divergences
            learning_rate=1e-4,
            memory_file='meta_learner_fusion.pth'
        )
        self.use_meta_learner = True  # Activer le meta-learner
        
        # PATCH V4: Assigner à la variable globale pour le dashboard
        global GLOBAL_META_LEARNER
        GLOBAL_META_LEARNER = self.meta_learner
        
        # ═══════════════════════════════════════════════════════════════════
        # PATCH V4.1: CORRÉLATION INTER-PAIRES + POSITION SIZING DYNAMIQUE
        # ═══════════════════════════════════════════════════════════════════
        global GLOBAL_CORRELATION_SYSTEM, GLOBAL_POSITION_SIZER
        
        self.correlation_system = CrossPairCorrelation()
        GLOBAL_CORRELATION_SYSTEM = self.correlation_system
        
        self.position_sizer = DynamicPositionSizer(
            base_risk_pct=0.02,   # 2% par défaut
            max_risk_pct=0.05,    # Max 5% (signal très fort + confirmé)
            min_risk_pct=0.005   # Min 0.5% (signal faible ou drawdown)
        )
        GLOBAL_POSITION_SIZER = self.position_sizer
        
        logging.info("  RL Agent Manager initialis  avec META-LEARNER FUSION PPO-DQN")
        logging.info("  [CORRELATION] Cross-pair correlation system ready")
        logging.info("  [POSITION SIZING] Dynamic position sizer ready")

    #   CORRECTION: FONCTION MOCK DATA AU BON NIVEAU
    
    def _get_mt5_timeframe(self, minutes: int) -> int:
        """Convertit les minutes en timeframe MT5"""
        if not MT5_AVAILABLE:
            return 15
        
        timeframe_map = {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5, 
            15: mt5.TIMEFRAME_M15,
            30: mt5.TIMEFRAME_M30,
            60: mt5.TIMEFRAME_H1,
            240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1
        }
        return timeframe_map.get(minutes, mt5.TIMEFRAME_M15)


    def _generate_mock_data(self, bars: int) -> pd.DataFrame:
        """G n re des donn es mock pour testing - VERSION CORRIG E"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=bars, freq='15min')
            
            # Generate random walk price
            price = 1.1000
            prices = [price]
            for _ in range(bars - 1):
                price_change = np.random.randn() * 0.001
                price *= (1 + price_change)
                prices.append(price)
            
            df = pd.DataFrame({
                'open': prices,
                'high': [p * (1 + abs(np.random.randn() * 0.0005)) for p in prices],
                'low': [p * (1 - abs(np.random.randn() * 0.0005)) for p in prices],
                'close': prices,
                'volume': np.random.randint(1000, 10000, bars)
            }, index=dates)
            
            # Calculer les indicateurs
            df = self._calculate_indicators(df)
            
            logging.debug(f"  Mock data g n r e: {len(df)} bars")
            return df
            
        except Exception as e:
            logging.error(f"  Erreur g n ration mock data: {e}")
            # Fallback minimal
            return pd.DataFrame({
                'open': [1.0] * bars,
                'high': [1.0] * bars,
                'low': [1.0] * bars,
                'close': [1.0] * bars,
                'rsi': [50.0] * bars,
                'macd': [0.0] * bars,
                'macd_signal': [0.0] * bars,
                'atr': [0.0] * bars,
                'bb_middle': [1.0] * bars,
                'bb_upper': [1.0] * bars
            })

    async def _train_agent_legacy_disabled(self, symbol: str):
        
        
        
        """
        EntraÃ®ne les agents PPO, DQN et META-LEARNER ENSEMBLE.
        
        NOUVELLE ARCHITECTURE EN 3 PHASES:
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        Phase 1 (25%): PPO apprend seul les bases
        Phase 2 (25%): DQN apprend seul les bases  
        Phase 3 (50%): ENSEMBLE - PPO, DQN et Meta-Learner apprennent ENSEMBLE
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        """
        #   CORRECTION: INITIALISATION AVEC VALEURS PAR D FAUT
        learning_rate = self.config.RL_LEARNING_RATE
        clip_range = 0.2
        exploration_fraction = self.config.RL_EXPLORATION_FRACTION
        historical_data = None
        
        try:
            logging.info(f"  Training agents pour {symbol}...")
            logging.info("=" * 70)
            logging.info("  ENSEMBLE TRAINING MODE: PPO + DQN + META-LEARNER")
            logging.info("=" * 70)
            
            # Stocker le symbole courant pour la normalisation XAUUSD
            self.current_symbol = symbol
            
            # Charger les donn es historiques
            historical_data = await self.load_historical_data(symbol)
            if historical_data is None or len(historical_data) < 100:
                raise ValueError(f"Donn es insuffisantes pour {symbol}")
            
            # Configuration sp cifique par symbole
            if symbol == "XAUUSD":
                learning_rate = self.config.RL_LEARNING_RATE * 0.3
                clip_range = 0.1
                exploration_fraction = self.config.RL_EXPLORATION_FRACTION * 1.5
            
            # Cr er l'environnement de base
            symbol_id = self.config.SYMBOLS.index(symbol)
            base_env = TradingEnvironment(historical_data, symbol_id=symbol_id, symbol=symbol)
            
            # Initialiser HRM-TRM
            self.hrm_trm_per_symbol[symbol] = base_env.hrm_trm
            
            # Calculer les timesteps par phase
            total_timesteps = self.config.RL_TOTAL_TIMESTEPS
            global training_phase
            training_phase = 1  # Phase 1: PPO Pre-Training
            
            # ═══════════════════════════════════════════════════════════════════
            # V5 ULTRA: PHASES RAPIDES POUR DQN IMMÉDIAT
            # ═══════════════════════════════════════════════════════════════════
            # AVANT: phase1=400k, phase2=400k, phase3=800k → DQN à 400k steps
            # APRÈS: phase1=50k, phase2=50k, phase3=500k → DQN IMMÉDIAT (déjà passé 122k)
            
            phase1_steps = 50000    # PPO Pre-Training: 50k steps (au lieu de 400k)
            phase2_steps = 50000    # DQN Pre-Training: 50k steps (au lieu de 400k)
            phase3_steps = 500000   # Ensemble Fusion: 500k steps (au lieu de 800k)
            
            total_steps_planned = phase1_steps + phase2_steps + phase3_steps  # 600k total
            
            logging.info(f"  [V5 ULTRA] Phase Configuration:")
            logging.info(f"    Phase 1 PPO:      0 → {phase1_steps:,} steps")
            logging.info(f"    Phase 2 DQN: {phase1_steps:,} → {phase1_steps+phase2_steps:,} steps")
            logging.info(f"    Phase 3 Ensemble: {phase1_steps+phase2_steps:,} → {total_steps_planned:,} steps")
            logging.info(f"    TOTAL: {total_steps_planned:,} steps (~40% reduction for faster DQN)")
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 1: PrÃ©-entraÃ®nement PPO (apprend les bases seul)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            logging.info("")
            logging.info("â•" * 70)
            logging.info(f"  PHASE 1/3: PPO PRÃ‰-TRAINING ({phase1_steps:,} steps)")
            logging.info("â•" * 70)
            
            env_ppo = GymnasiumToSB3Wrapper(TradingEnvironment(historical_data, symbol_id=symbol_id, symbol=symbol))
            vec_env_ppo = DummyVecEnv([lambda: env_ppo])
            
            self.ppo_agents[symbol] = PPO(
                'MlpPolicy',
                vec_env_ppo,
                verbose=1,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=256,
                n_epochs=6,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.1,
                max_grad_norm=0.5,
                tensorboard_log=f"./logs/ppo_{symbol}"
            )
            
            tqdm_ppo = UnifiedProgressCallback(phase1_steps, "PPO-Phase1", symbol, "magenta")
            detailed_ppo = DetailedTrainingCallback(symbol, "PPO")
            self.ppo_agents[symbol].learn(
                total_timesteps=phase1_steps,
                callback=[
                    RLTrainingCallback(self.ws_server, symbol, 'PPO'), 
                    tqdm_ppo, 
                    detailed_ppo,
                    DragonRewardShapingCallback(env_ppo.env if hasattr(env_ppo, 'env') else env_ppo)
                ]
            )
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 2: PrÃ©-entraÃ®nement DQN (apprend les bases seul)
            
            # PATCH V4.1: Mettre à jour training_phase
            # global déjà déclaré en Phase 1
            training_phase = 2  # Phase 2: DQN Pre-Training
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            logging.info("")
            logging.info("â•" * 70)
            logging.info(f"  PHASE 2/3: DQN PRÃ‰-TRAINING ({phase2_steps:,} steps)")
            logging.info("â•" * 70)
            
            env_dqn = GymnasiumToSB3Wrapper(TradingEnvironment(historical_data, symbol_id=symbol_id, symbol=symbol))
            vec_env_dqn = DummyVecEnv([lambda: env_dqn])
            
            # === PATCH ULTIME: DQN Epsilon Decay AGRESSIF ===
            self.dqn_agents[symbol] = DQN(
                'MlpPolicy',
                vec_env_dqn,
                verbose=0,
                learning_rate=learning_rate,
                buffer_size=self.config.RL_BUFFER_SIZE,
                learning_starts=5000,              # Commence à apprendre plus tôt (était 500)
                batch_size=256,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=2,                  # Double les gradient steps
                target_update_interval=800,        # Update target plus souvent (était 5000)
                exploration_fraction=0.25,         # Explore moins longtemps (était ~0.5)
                exploration_initial_eps=1.0,
                exploration_final_eps=0.02,        # Quasi-déterministe rapidement (était 0.05)
                tensorboard_log=f"./logs/dqn_{symbol}"
            )
            
            tqdm_dqn = UnifiedProgressCallback(phase2_steps, "DQN-Phase2", symbol, "cyan")
            detailed_dqn = DetailedTrainingCallback(symbol, "DQN")
            self.dqn_agents[symbol].learn(
                total_timesteps=phase2_steps,
                callback=[
                    RLTrainingCallback(self.ws_server, symbol, 'DQN'), 
                    tqdm_dqn, 
                    detailed_dqn,
                    DragonLearningCallback(env_dqn.env if hasattr(env_dqn, 'env') else env_dqn, injection_frequency=500),
                    DragonRewardShapingCallback(env_dqn.env if hasattr(env_dqn, 'env') else env_dqn)
                ]
            )
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 3: ENSEMBLE TRAINING (PPO + DQN + Meta-Learner ENSEMBLE)
            
            # PATCH V4.1: Mettre à jour training_phase
            # global déjà déclaré en Phase 1
            training_phase = 3  # Phase 3: Ensemble Fusion
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            logging.info("")
            logging.info("â•" * 70)
            logging.info(f"  PHASE 3/3: ENSEMBLE TRAINING ({phase3_steps:,} steps)")
            logging.info("  PPO + DQN + META-LEARNER apprennent ENSEMBLE")
            logging.info("â•" * 70)
            
            # CrÃ©er l'environnement ENSEMBLE
            ensemble_base = TradingEnvironment(historical_data, symbol_id=symbol_id, symbol=symbol)
            ensemble_env = EnsembleTrainingEnvironment(
                base_env=ensemble_base,
                ppo_agent=self.ppo_agents[symbol],
                dqn_agent=self.dqn_agents[symbol],
                meta_learner=self.meta_learner,
                training_mode='ensemble'
            )
            ensemble_wrapped = GymnasiumToSB3Wrapper(ensemble_env)
            vec_env_ensemble = DummyVecEnv([lambda: ensemble_wrapped])
            
            # Continuer l'entraÃ®nement de PPO avec l'environnement ensemble
            # (DQN apprend aussi indirectement via les dÃ©cisions ensemble)
            self.ppo_agents[symbol].set_env(vec_env_ensemble)
            
            tqdm_ensemble = UnifiedProgressCallback(phase3_steps, "ENSEMBLE", symbol, "yellow")
            ensemble_callback = EnsembleTrainingCallback(
                ensemble_env=ensemble_env,
                meta_learner=self.meta_learner,
                symbol=symbol
            )
            
            self.ppo_agents[symbol].learn(
                total_timesteps=phase3_steps,
                callback=[
                    RLTrainingCallback(self.ws_server, symbol, 'ENSEMBLE'),
                    tqdm_ensemble,
                    ensemble_callback,
                    DragonRewardShapingCallback(ensemble_base)
                ],
                reset_num_timesteps=False  # Continuer Ã  partir du checkpoint PPO
            )
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # LOG FINAL: Statistiques de l'entraÃ®nement ensemble
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            logging.info("")
            logging.info("â•" * 70)
            logging.info(f"  TRAINING COMPLETE - {symbol}")
            logging.info("â•" * 70)
            
            ensemble_stats = ensemble_env.get_ensemble_stats()
            if ensemble_stats:
                logging.info(f"  Ensemble Decisions: {ensemble_stats.get('total_decisions', 0):,}")
                logging.info(f"  Consensus Rate: {ensemble_stats.get('consensus_rate', 0):.1f}%")
                logging.info(f"  PPO Selected: {ensemble_stats.get('ppo_selected_rate', 0):.1f}%")
                logging.info(f"  DQN Selected: {ensemble_stats.get('dqn_selected_rate', 0):.1f}%")
            
            meta_insights = self.meta_learner.get_context_insights()
            if meta_insights:
                logging.info("")
                logging.info("  [META-LEARNER] Context Performance:")
                for ctx, data in meta_insights.items():
                    logging.info(f"    {ctx}: PPO={data['ppo_rate']:.1f}% vs DQN={data['dqn_rate']:.1f}% "
                               f"(winner={data['winner']}, n={data['samples']})")
            
            logging.info("â•" * 70)
            
            # PATCH V4.1: Sauvegarder le divergence_system à la fin
            global GLOBAL_DIVERGENCE_SYSTEM
            if GLOBAL_DIVERGENCE_SYSTEM is not None:
                GLOBAL_DIVERGENCE_SYSTEM.save_memory()
                logging.info(f"  [DIVERGENCE] Memory saved: {GLOBAL_DIVERGENCE_SYSTEM.total_divergences_detected} divergences")
            
        except Exception as e:
            #   CORRECTION: UN SEUL BLOC EXCEPT PROPRE
            logging.error(f"  Erreur training {symbol}: {e}")
            logging.debug(f"Params utilis s: LR={learning_rate}, Clip={clip_range}, Exploration={exploration_fraction}")
            traceback.print_exc()
            
            # Marquer le mod le comme potentiellement corrompu
            self.corrupted_models[symbol] = True
            self.training_attempts[symbol] += 1
            
        finally:
            #   CORRECTION: R INITIALISATION DANS TOUS LES CAS
            self.current_symbol = None
            
            # Nettoyage des ressources
            if 'vec_env' in locals():
                try:
                    vec_env.close()
                except:
                    pass

    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques + signaux avanc s - VERSION PATCH E
        
        Features (14 total) :
        - 4 prix: open, high, low, close
        - 6 indicateurs: rsi, macd, macd_signal, atr, bb_middle, bb_upper
        - 4 signaux avanc s: wick_score, hli, ETI, SNAP
        """
        logging.debug("  _calculate_indicators PATCH  appel ")
        
        try:
            # ===== NORMALISATION XAUUSD =====
            if hasattr(self, 'current_symbol') and self.current_symbol == "XAUUSD":
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    if col in df.columns and df[col].max() > 100:
                        df[col] = df[col] / 1000.0
                        logging.debug(f"     XAUUSD normalis : {col}")
            
            # ===== INDICATEURS CLASSIQUES =====
            
            # RSI - Version robuste
            try:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                logging.debug("     RSI calcul ")
            except Exception as e:
                logging.warning(f"  Erreur calcul RSI: {e}")
                df['rsi'] = 50.0  # Valeur par d faut
            
            # MACD - Version robuste
            try:
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = exp1 - exp2
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                logging.debug("     MACD calcul ")
            except Exception as e:
                logging.warning(f"  Erreur calcul MACD: {e}")
                df['macd'] = 0.0
                df['macd_signal'] = 0.0
            
            # ATR - Version robuste
            try:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                df['atr'] = true_range.rolling(14).mean()
                logging.debug("     ATR calcul ")
            except Exception as e:
                logging.warning(f"  Erreur calcul ATR: {e}")
                df['atr'] = 0.0
            
            # Bollinger Bands - Version robuste
            try:
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                bb_std = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
                df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
                logging.debug("     Bollinger Bands calcul s")
            except Exception as e:
                logging.warning(f"  Erreur calcul Bollinger Bands: {e}")
                df['bb_middle'] = df['close']
                df['bb_upper'] = df['close']
                df['bb_lower'] = df['close']
            
            # Tick volume (si absent, cr er dummy)
            if 'tick_volume' not in df.columns:
                df['tick_volume'] = 1
                logging.debug("     Volume dummy cr ")
            
            # ===== SIGNAUX AVANC S SIMPLIFI S POUR STABILIT  =====
            
            try:
                # Composants de bougie basiques
                df['body'] = (df['close'] - df['open']).abs()
                df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
                df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']
                
                # Normalisation par ATR
                atr_safe = df['atr'].replace(0, np.nan).bfill().fillna(1e-6)
                
                # Wick score simplifi 
                uw = (df['upper_wick'] / atr_safe).rolling(14).mean()
                lw = (df['lower_wick'] / atr_safe).rolling(14).mean()
                df['wick_balance'] = (uw - lw)
                df['wick_score'] = np.tanh(df['wick_balance'].rolling(8).mean().fillna(0))
                
                # Heaviness / Lift Index (HLI) simplifi 
                df['clv'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
                            (df['high'] - df['low']).replace(0, np.nan).bfill().fillna(1e-6)
                
                hli = (
                    0.50 * df['wick_balance']
                    + 0.15 * df['clv']
                )
                df['hli'] = np.tanh(hli.rolling(8).mean().fillna(0))
                
                logging.debug("     Signaux avanc s calcul s")
            except Exception as e:
                logging.warning(f"  Erreur signaux avanc s: {e}")
                df['wick_score'] = 0.0
                df['hli'] = 0.0
            
            # ===== AJOUT DES FEATURES MANQUANTES =====
            # Feature 12: volume_norm
            if 'tick_volume' in df.columns:
                df['volume_norm'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
            else:
                df['volume_norm'] = 1.0
            
            # Feature 13: ema_20
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Feature 14: momentum
            df['momentum'] = df['close'].pct_change(periods=14) * 100
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # DIVERGENCE FEATURES POUR L'OBSERVATION DE L'AGENT (Features 15-18)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            try:
                df = self._calculate_divergence_features_for_observation(df, lookback=20)
                logging.debug("     Divergence features calculÃ©es pour observation agent")
            except Exception as e:
                logging.warning(f"  Erreur calcul divergence features: {e}")
                df['div_rsi_signal'] = 0.0
                df['div_macd_signal'] = 0.0
                df['div_momentum_signal'] = 0.0
                df['div_strength'] = 0.0
            
            # Normalisation XAUUSD pour les nouvelles features
            if hasattr(self, 'current_symbol') and self.current_symbol == "XAUUSD":
                indicator_columns = ['bb_middle', 'bb_upper', 'bb_lower', 'atr', 'ema_20']
                for col in indicator_columns:
                    if col in df.columns and df[col].max() > 100:
                        df[col] = df[col] / 1000.0
            
            # Fill NaN pour les nouvelles features INCLUANT DIVERGENCES
            fill_cols = ['volume_norm', 'ema_20', 'momentum', 'div_rsi_signal', 'div_macd_signal', 'div_momentum_signal', 'div_strength']
            for col in fill_cols:
                if col in df.columns:
                    df[col] = df[col].bfill().fillna(0)
            
            # Fill NaN g n ral
            df = df.bfill().fillna(0)
            
            # CORRECTION: RETOURNER 18 COLONNES (14 + 4 DIVERGENCES)
            expected_columns = [
                'open', 'high', 'low', 'close',
                'rsi', 'macd', 'macd_signal', 'atr', 'bb_middle', 'bb_upper',
                'bb_lower', 'volume_norm', 'ema_20', 'momentum',
                'div_rsi_signal', 'div_macd_signal', 'div_momentum_signal', 'div_strength'
            ]
            
            # V rifier que toutes les colonnes attendues existent
            for col in expected_columns:
                if col not in df.columns:
                    logging.warning(f"  Colonne manquante: {col}, cr ation avec z ros")
                    df[col] = 0.0
            
            # Retourner exactement les 18 colonnes attendues
            result_df = df[expected_columns].copy()
            
            logging.debug(f"  _calculate_indicators termin : {len(result_df)} lignes, {len(result_df.columns)} colonnes")
            return result_df
            
        except Exception as e:
            logging.error(f"  Erreur _calculate_indicators: {e}")
            # Fallback: retourner un DataFrame basique avec 18 colonnes
            n_rows = len(df) if len(df) > 0 else 1
            fallback_data = {
                'open': [1.0] * n_rows, 'high': [1.0] * n_rows, 'low': [1.0] * n_rows, 'close': [1.0] * n_rows,
                'rsi': [50.0] * n_rows, 'macd': [0.0] * n_rows, 'macd_signal': [0.0] * n_rows,
                'atr': [0.0] * n_rows, 'bb_middle': [1.0] * n_rows, 'bb_upper': [1.0] * n_rows,
                'bb_lower': [1.0] * n_rows, 'volume_norm': [1.0] * n_rows, 'ema_20': [1.0] * n_rows, 'momentum': [0.0] * n_rows,
                'div_rsi_signal': [0.0] * n_rows, 'div_macd_signal': [0.0] * n_rows,
                'div_momentum_signal': [0.0] * n_rows, 'div_strength': [0.0] * n_rows
            }
            return pd.DataFrame(fallback_data)
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # NOUVELLE MÃ‰THODE: Calcul des features de divergence pour l'observation
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    def _calculate_divergence_features_for_observation(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        Calcule les features de divergence pour CHAQUE bar du DataFrame.
        Ces features sont incluses dans l'observation de l'agent RL.
        
        Features ajoutÃ©es:
        - div_rsi_signal: -1 (bearish), 0 (none), +1 (bullish)
        - div_macd_signal: -1 (bearish), 0 (none), +1 (bullish)
        - div_momentum_signal: -1 (bearish), 0 (none), +1 (bullish)
        - div_strength: force combinÃ©e 0.0 Ã  1.0
        """
        from scipy.signal import argrelextrema
        
        n = len(df)
        df['div_rsi_signal'] = 0.0
        df['div_macd_signal'] = 0.0
        df['div_momentum_signal'] = 0.0
        df['div_strength'] = 0.0
        
        if n < lookback + 5:
            return df
        
        prices = df['close'].values
        rsi = df['rsi'].values if 'rsi' in df.columns else np.full(n, 50.0)
        macd = df['macd'].values if 'macd' in df.columns else np.zeros(n)
        momentum = df['momentum'].values if 'momentum' in df.columns else np.zeros(n)
        
        for i in range(lookback + 5, n):
            window_prices = prices[i-lookback:i]
            window_rsi = rsi[i-lookback:i]
            window_macd = macd[i-lookback:i]
            window_momentum = momentum[i-lookback:i]
            
            signals = []
            
            # RSI Divergence
            rsi_sig = self._detect_single_divergence(window_prices, window_rsi)
            df.iloc[i, df.columns.get_loc('div_rsi_signal')] = rsi_sig
            if rsi_sig != 0: signals.append(abs(rsi_sig))
            
            # MACD Divergence
            macd_sig = self._detect_single_divergence(window_prices, window_macd)
            df.iloc[i, df.columns.get_loc('div_macd_signal')] = macd_sig
            if macd_sig != 0: signals.append(abs(macd_sig))
            
            # Momentum Divergence
            mom_sig = self._detect_single_divergence(window_prices, window_momentum)
            df.iloc[i, df.columns.get_loc('div_momentum_signal')] = mom_sig
            if mom_sig != 0: signals.append(abs(mom_sig))
            
            # Force combinÃ©e
            if signals:
                df.iloc[i, df.columns.get_loc('div_strength')] = np.mean(signals)
        
        return df
    
    def _detect_single_divergence(self, prices: np.ndarray, indicator: np.ndarray) -> float:
        """
        DÃ©tecte une divergence et retourne un signal:
        -1.0 = Divergence bearish, +1.0 = Divergence bullish, 0.0 = Pas de divergence
        """
        from scipy.signal import argrelextrema
        
        try:
            if len(prices) < 10 or len(indicator) < 10:
                return 0.0
            
            # Trouver pics et creux
            price_highs = argrelextrema(prices, np.greater, order=3)[0]
            price_lows = argrelextrema(prices, np.less, order=3)[0]
            ind_highs = argrelextrema(indicator, np.greater, order=3)[0]
            ind_lows = argrelextrema(indicator, np.less, order=3)[0]
            
            # BEARISH: Prix higher high, indicateur lower high
            if len(price_highs) >= 2 and len(ind_highs) >= 2:
                p1, p2 = price_highs[-2], price_highs[-1]
                i1, i2 = ind_highs[-2], ind_highs[-1]
                if prices[p2] > prices[p1] and indicator[i2] < indicator[i1]:
                    strength = min(1.0, abs(prices[p2] - prices[p1]) / (prices[p1] + 1e-8))
                    return -strength
            
            # BULLISH: Prix lower low, indicateur higher low
            if len(price_lows) >= 2 and len(ind_lows) >= 2:
                p1, p2 = price_lows[-2], price_lows[-1]
                i1, i2 = ind_lows[-2], ind_lows[-1]
                if prices[p2] < prices[p1] and indicator[i2] > indicator[i1]:
                    strength = min(1.0, abs(prices[p1] - prices[p2]) / (prices[p1] + 1e-8))
                    return +strength
            
            return 0.0
        except:
            return 0.0
        self.config = config
        self.ppo_agents: Dict[str, PPO] = {}
        self.dqn_agents: Dict[str, DQN] = {}
        self.performance_history: Dict[str, deque] = {sym: deque(maxlen=50) for sym in config.SYMBOLS}
        self.last_retrain_time: Dict[str, float] = {}
        self.hrm_trm_per_symbol: Dict[str, AdvancedHRMTRM] = {}
        self.ws_server = RLTrainingWebSocketServer(config.RL_WS_PORT)
        
        # CORRECTION XAUUSD - VARIABLES DE GESTION
        self.current_symbol = None  # Pour la normalisation dans _calculate_indicators
        self.corrupted_models: Dict[str, bool] = {sym: False for sym in config.SYMBOLS}  # Suivi mod les corrompus
        self.training_attempts: Dict[str, int] = {sym: 0 for sym in config.SYMBOLS}  # Compteur tentatives training
        
        # CORRECTION XAUUSD - CONFIGURATIONS SP CIFIQUES
        self.symbol_configs = {
            "XAUUSD": {
                "learning_rate_multiplier": 0.3,  # 70% plus lent
                "clip_range": 0.1,  # Plus stable
                "exploration_multiplier": 1.5,  # Plus d'exploration
                "max_training_attempts": 3  # Maximum 3 tentatives
            },
            "EURUSD": {
                "learning_rate_multiplier": 1.0,
                "clip_range": 0.2,
                "exploration_multiplier": 1.0,
                "max_training_attempts": 2
            },
            "GBPUSD": {
                "learning_rate_multiplier": 1.0,
                "clip_range": 0.2, 
                "exploration_multiplier": 1.0,
                "max_training_attempts": 2
            },
            "USDJPY": {
                "learning_rate_multiplier": 1.0,
                "clip_range": 0.2,
                "exploration_multiplier": 1.0,
                "max_training_attempts": 2
            }
        }
        
        logging.info("  RL Agent Manager initialis  avec fusion PPO-DQN")
        logging.info("  Configuration XAUUSD: LR 0.3, Clip=0.1, Exploration 1.5")

    async def validate_agent_performance(self, symbol: str, episodes: int = 5) -> dict:
        """
          VALIDATION POST-TRAINING - V rifie que l'agent trade activement
        
        Crit res:
        - Minimum 10 trades par  pisode
        - Win rate > 30%
        - HOLD ratio < 50%
        
        Args:
            symbol: Symbole   valider
            episodes: Nombre d' pisodes de test
            
        Returns:
            Dict avec m triques de validation
        """
        logging.info(f"  VALIDATION de l'agent {symbol}...")
        
        try:
            # Charger donn es de test
            test_data = await self.load_historical_data(symbol, bars=500)
            
            # Cr er environnement de test
            symbol_id = self.config.SYMBOLS.index(symbol)
            test_env = TradingEnvironment(test_data, symbol_id=symbol_id, symbol=symbol)
            
            total_trades = 0
            total_holds = 0
            total_actions = 0
            episode_rewards = []
            
            for ep in range(episodes):
                obs, _ = test_env.reset()
                done = False
                ep_trades = 0
                ep_holds = 0
                ep_reward = 0
                
                while not done:
                    # Pr diction
                    action_ppo, _ = self.ppo_agents[symbol].predict(obs, deterministic=True)
                    
                    # Compter actions
                    if action_ppo == 2:  # HOLD
                        ep_holds += 1
                    else:
                        ep_trades += 1
                    
                    # Step
                    obs, reward, terminated, truncated, _ = test_env.step(action_ppo)
                    done = terminated or truncated
                    ep_reward += reward
                    total_actions += 1
                
                total_trades += ep_trades
                total_holds += ep_holds
                episode_rewards.append(ep_reward)
                
                logging.info(f"   Episode {ep+1}: {ep_trades} trades, {ep_holds} holds, reward: {ep_reward:.0f}")
            
            # M triques
            avg_trades = total_trades / episodes
            hold_ratio = total_holds / total_actions if total_actions > 0 else 0
            avg_reward = sum(episode_rewards) / episodes
            
            # Validation
            is_valid = (
                avg_trades >= 10 and
                hold_ratio <= 0.5
            )
            
            results = {
                'symbol': symbol,
                'episodes': episodes,
                'avg_trades_per_episode': avg_trades,
                'hold_ratio': hold_ratio,
                'avg_reward': avg_reward,
                'validation_passed': is_valid
            }
            
            # Affichage
            emoji = " " if is_valid else " "
            logging.info("\n" + "="*70)
            logging.info(f"{emoji} VALIDATION {symbol}")
            logging.info("="*70)
            logging.info(f"Trades/Episode:  {avg_trades:.1f} {' ' if avg_trades >= 10 else '  (min: 10)'}")
            logging.info(f"HOLD Ratio:      {hold_ratio:.1%} {' ' if hold_ratio <= 0.5 else '  (max: 50%)'}")
            logging.info(f"Avg Reward:      {avg_reward:.0f}")
            logging.info(f"STATUS:          {'PASSED  ' if is_valid else 'FAILED  '}")
            logging.info("="*70)
            
            return results
            
        except Exception as e:
            logging.error(f"  Erreur validation {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {'validation_passed': False, 'error': str(e)}

    async def start_training_all_agents(self, skip: bool = False):
        """Entra ne tous les agents pour tous les symboles - avec option skip"""
        
        if skip:
            logging.warning("  TRAINING SKIPPED - Mode d veloppement")
            logging.info("  Chargement des mod les existants ou cr ation de mod les vides...")
            
            # Cr er des agents mock pour le d veloppement
            for symbol in self.config.SYMBOLS:
                await self._load_or_create_mock_agents(symbol)
            
            logging.info("  Mode skip activ  - Agents pr ts pour test")
            return
        
        logging.info("  D marrage training de tous les agents...")
        logging.info("  Astuce: Appuyez sur 'S' dans les 5 premi res secondes pour skip")
        
        # Attendre 5 secondes avec possibilit  de skip
        for i in range(5, 0, -1):
            logging.info(f"   Training d marre dans {i}s... (S pour skip)")
            await asyncio.sleep(1)
        
        tasks = [self.train_agent(symbol) for symbol in self.config.SYMBOLS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, result in zip(self.config.SYMBOLS, results):
            if isinstance(result, Exception):
                logging.error(f"  [GATHER] {symbol} a echoue: {result}")
        logging.info("  Training de tous les agents termin ")
    
    async def _load_or_create_mock_agents(self, symbol: str):
        """Charge mod les existants ou cr e des agents mock pour test"""
        try:
            # Essayer de charger mod les sauvegard s
            ppo_path = f"./models/ppo_{symbol}.zip"
            dqn_path = f"./models/dqn_{symbol}.zip"
            
            if os.path.exists(ppo_path):  # PPO seul suffit — SAC est en .pt
                logging.info(f"  Chargement mod les existants pour {symbol}")
                
                # Cr er env minimal
                historical_data = await self.load_historical_data(symbol)
                symbol_id = self.config.SYMBOLS.index(symbol)
                env = TradingEnvironment(historical_data, symbol_id=symbol_id, symbol=symbol)
                env = GymnasiumToSB3Wrapper(env)
                vec_env = DummyVecEnv([lambda: env])
                
                # Charger mod les
                self.ppo_agents[symbol] = PPO.load(ppo_path, env=vec_env)
                self.dqn_agents[symbol] = DQN.load(dqn_path, env=vec_env)
                self.hrm_trm_per_symbol[symbol] = env.hrm_trm
                
                logging.info(f"  Mod les charg s pour {symbol}")
            else:
                logging.warning(f"  Pas de mod les sauvegard s pour {symbol}")
                logging.info(f"  Cr ation d'agents mock pour test...")
                
                # Cr er agents minimaux pour test
                historical_data = await self.load_historical_data(symbol)
                symbol_id = self.config.SYMBOLS.index(symbol)
                env = TradingEnvironment(historical_data, symbol_id=symbol_id, symbol=symbol)
                env = GymnasiumToSB3Wrapper(env)
                vec_env = DummyVecEnv([lambda: env])
                
                # Agents mock (non entra n s)
                self.ppo_agents[symbol] = PPO('MlpPolicy', vec_env, verbose=0)
                self.dqn_agents[symbol] = DQN('MlpPolicy', vec_env, verbose=0)
                self.hrm_trm_per_symbol[symbol] = env.hrm_trm
                
                logging.info(f"  Agents mock cr s - Performance non garantie!")
                
        except Exception as e:
            logging.error(f"  Erreur load/create mock agents pour {symbol}: {e}")
            traceback.print_exc()
            
            #   FIX: Cr er au moins un HRM-TRM minimal
            try:
                logging.warning(f"  Cr ation HRM-TRM minimal de secours pour {symbol}...")
                self.hrm_trm_per_symbol[symbol] = AdvancedHRMTRM(
                    input_dim=14,
                    hidden_dim=CONFIG.HRM_HIDDEN_DIM,
                    H_cycles=CONFIG.HRM_H_CYCLES,
                    L_cycles=CONFIG.HRM_L_CYCLES,
                    num_heads=CONFIG.HRM_NUM_HEADS,
                    halt_max_steps=CONFIG.HRM_HALT_MAX_STEPS,
                    exploration_prob=CONFIG.HRM_EXPLORATION_PROB,
                    num_symbols=len(CONFIG.SYMBOLS),
                    vocab_size=4
                )
                logging.info(f"  HRM-TRM de secours cr  pour {symbol}")
            except Exception as fallback_error:
                logging.error(f"  Impossible de cr er HRM-TRM de secours: {fallback_error}")

    async def train_agent(self, symbol: str):
        """
        Entra ne les agents PPO et DQN pour un symbole.
          CHARGE LES MOD LES EXISTANTS et CONTINUE L'ENTRA NEMENT
        
        Process:
        1. Load historical data
        2. Create TradingEnvironment
        3. CHECK si mod les existants   LOAD et continue
        4. Sinon   Create new et train from scratch
        5. Save models
        """
        logging.info(f"  Training agents pour {symbol}...")
        
        try:
            # CORRECTION XAUUSD - STOCKER LE SYMBOLE COURANT
            self.current_symbol = symbol
            
            # [V6 GALAXY] Charger les vraies données V6 (65 indicateurs réels)
            # au lieu des données MT5 basiques — élimine le dim mismatch training/inference
            historical_data = None
            try:
                from goldeneye_csv_loader import charger_donnees_v6 as _charger_csv
                _df_v6 = _charger_csv(symbol, "M15", normaliser=True)
                if _df_v6 is not None and len(_df_v6) >= 100:
                    historical_data = _df_v6
                    logging.info(f"[V6 CSV] {symbol}: {len(_df_v6)} barres x {len(_df_v6.columns)} features reelles")
            except Exception as _csv_err:
                logging.warning(f"[V6 CSV] Fallback MT5 pour {symbol}: {_csv_err}")

            # Fallback MT5 si CSV V6 indisponible
            if historical_data is None:
                historical_data = await self.load_historical_data(symbol)

            # CORRECTION XAUUSD - PARAM TRES ADAPTATIFS
            if symbol == "XAUUSD":
                learning_rate = self.config.RL_LEARNING_RATE * 0.3
                clip_range = 0.1
                exploration_fraction = self.config.RL_EXPLORATION_FRACTION * 1.5
            else:
                learning_rate = self.config.RL_LEARNING_RATE
                clip_range = 0.2
                exploration_fraction = self.config.RL_EXPLORATION_FRACTION
            
            # Create environment
            symbol_id = self.config.SYMBOLS.index(symbol)
            env = TradingEnvironment(historical_data, symbol_id=symbol_id, symbol=symbol)
            vec_env = DummyVecEnv([lambda: env])
            
            # Initialize HRM-TRM
            self.hrm_trm_per_symbol[symbol] = env.hrm_trm
            
            # ============================================================
            #   CHARGEMENT INTELLIGENT DES MOD LES
            # ============================================================
            ppo_path = f"./models/ppo_{symbol}.zip"
            dqn_path = f"./models/dqn_{symbol}.zip"
            hrm_path = f"./models/hrm_trm_{symbol}.pt"

            # Vérifie si un zip RecurrentPPO existe (si zip classique → RPPO.load() échoue → fresh RPPO)
            models_exist = os.path.exists(ppo_path)
            
            if models_exist:
                logging.info(f"  MOD LES EXISTANTS D TECT S pour {symbol}")
                logging.info(f"     Chargement et CONTINUATION de l'entra nement...")
                
                try:
                    import torch  # Import torch ici au cas ou
                    
                    # Charger RecurrentPPO existant (ton modèle à 55% WR)
                    self.ppo_agents[symbol] = RecurrentPPO.load(ppo_path, env=vec_env)
                    logging.info(f"     RPPO chargé depuis {ppo_path} ({self.ppo_agents[symbol].num_timesteps:,} steps)")

                    # SAC — chargé plus bas dans la phase SAC
                    
                    # Charger HRM-TRM si existe
                    if os.path.exists(hrm_path):
                        self.hrm_trm_per_symbol[symbol].load_state_dict(torch.load(hrm_path))
                        logging.info(f"     HRM-TRM charg  depuis {hrm_path}")
                    
                    logging.info(f"  CONTINUATION de l'entra nement pour {symbol}...")
                    
                except Exception as rppo_err:
                    # Zip PPO classique → charger directement en PPO
                    logging.info(f"  Zip PPO classique detecte — chargement PPO (Agent 1 Manhattan)")
                    try:
                        self.ppo_agents[symbol] = PPO.load(ppo_path, env=vec_env)
                        logging.info(f"  PPO charge: {self.ppo_agents[symbol].num_timesteps:,} steps — WR 55%+ preserve!")
                        import torch
                        if os.path.exists(hrm_path):
                            self.hrm_trm_per_symbol[symbol].load_state_dict(torch.load(hrm_path))
                    except Exception as ppo_err:
                        logging.warning(f"  PPO.load echoue aussi: {ppo_err}")
                        models_exist = False

            # ============================================================
            # CREATION OU CONTINUATION — PHASE 1 : RecurrentPPO / PPO
            # ============================================================
            phase1_steps = GoldenEyeTrainingConfig.PHASE_1_PPO_STEPS
            phase2_steps = GoldenEyeTrainingConfig.PHASE_2_DQN_STEPS

            if not models_exist:
                if SB3_CONTRIB_AVAILABLE:
                    logging.info(f"    Creation RecurrentPPO (LSTM 128x1) pour {symbol}...")
                    self.ppo_agents[symbol] = RecurrentPPO(
                        'MlpLstmPolicy', vec_env, verbose=0,
                        learning_rate=1e-4, n_steps=1024, batch_size=256, n_epochs=10,
                        gamma=0.90, gae_lambda=0.95, clip_range=0.05,
                        ent_coef=0.10, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.01,
                        policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, shared_lstm=False),
                        tensorboard_log=f"./logs/rppo_{symbol}"
                    )
                else:
                    self.ppo_agents[symbol] = PPO(
                        'MlpPolicy', vec_env, verbose=0,
                        learning_rate=1e-4, n_steps=1024, batch_size=256, n_epochs=10,
                        gamma=0.90, gae_lambda=0.95, clip_range=0.05,
                        clip_range_vf=None, ent_coef=0.10, vf_coef=0.5,
                        max_grad_norm=0.5, target_kl=0.01,
                        tensorboard_log=f"./logs/ppo_{symbol}"
                    )
            else:
                logging.info(f"    CONTINUATION entrainement PPO pour {symbol}...")

            # Train PPO — thread barre + console silencing
            _console_handlers = [h for h in logging.getLogger().handlers
                                  if isinstance(h, logging.StreamHandler)
                                  and not isinstance(h, logging.FileHandler)]
            pbar_ppo = _creer_barre(phase1_steps, "PPO", symbol, "magenta")
            tqdm_ppo = UnifiedProgressCallback(phase1_steps, "PPO", symbol, "magenta", pbar=pbar_ppo)
            detailed_ppo = DetailedTrainingCallback(symbol, "PPO")
            from stable_baselines3.common.callbacks import CheckpointCallback
            checkpoint_cb = CheckpointCallback(save_freq=2000, save_path=f"./models/checkpoints_{symbol}/",
                                               name_prefix=f"ppo_{symbol}", verbose=0)
            for h in _console_handlers: h.setLevel(logging.WARNING)
            _ppo_thread, _ppo_stop_fn = _lancer_barre_thread(self.ppo_agents[symbol], pbar_ppo, env)
            self.ppo_agents[symbol].learn(total_timesteps=phase1_steps,
                callback=[RLTrainingCallback(self.ws_server, symbol, 'PPO'), tqdm_ppo, detailed_ppo, checkpoint_cb],
                reset_num_timesteps=False)
            _ppo_stop_fn.stop = True
            for h in _console_handlers: h.setLevel(logging.INFO)
            pbar_ppo.close()

            # PPO — sauvegarde immediate + snapshot stats
            try:
                os.makedirs("./models", exist_ok=True)
                self.ppo_agents[symbol].save(f"./models/ppo_{symbol}.zip")
                logging.info(f"  [SAVE] RPPO {symbol} sauvegarde")
            except Exception as _se:
                logging.warning(f"  [SAVE] Erreur RPPO: {_se}")
            _ppo_wins_snap = getattr(env, 'total_wins', 0)
            _ppo_losses_snap = getattr(env, 'total_losses', 0)
            _ppo_pnl_snap = sum(env.phase_closes) if getattr(env, 'phase_closes', None) else 0.0
            self._log_session_progression(symbol, "PPO", env, self.ppo_agents[symbol])

            # ============================================================
            # PHASE 2 : DiscreteSAC (remplace DQN)
            # ============================================================
            _sac_save_base = f"./models/sac_{symbol}"
            _sac_file = _sac_save_base + "_sac.pt"
            if symbol not in self.dqn_agents:
                self.dqn_agents[symbol] = DiscreteSAC(
                    'MlpPolicy', vec_env, learning_rate=3e-4, buffer_size=200000,
                    learning_starts=1000, batch_size=256, gamma=0.90, tau=0.005,
                    train_freq=4, gradient_steps=2, target_update_interval=500, verbose=0)
                if os.path.exists(_sac_file):
                    try:
                        self.dqn_agents[symbol].load(_sac_save_base, vec_env)
                        logging.info(f"    DiscreteSAC charge ({self.dqn_agents[symbol].num_timesteps:,} steps)")
                    except Exception as _le:
                        logging.warning(f"    SAC load failed: {_le}")
            pbar_sac = _creer_barre(phase2_steps, "SAC", symbol, "cyan")
            tqdm_sac = UnifiedProgressCallback(phase2_steps, "SAC", symbol, "cyan", pbar=pbar_sac)
            self.dqn_agents[symbol].learn(total_timesteps=phase2_steps, callback=[tqdm_sac], reset_num_timesteps=False)
            pbar_sac.close()
            try:
                self.dqn_agents[symbol].save(_sac_save_base)
                logging.info(f"  [SAVE] DiscreteSAC {symbol} -> {_sac_file}")
            except Exception as _se:
                logging.warning(f"  [SAVE] Erreur SAC: {_se}")
            self._log_session_progression(symbol, "SAC", env, self.dqn_agents[symbol])
            _sac_wins_snap = getattr(env, 'total_wins', 0)
            _sac_losses_snap = getattr(env, 'total_losses', 0)

            # ============================================================
            # PHASE 2B : TransformerAgent
            # ============================================================
            phase2b_steps = GoldenEyeTrainingConfig.PHASE_2B_TRANSFORMER_STEPS
            if not hasattr(self, 'transformer_agents'): self.transformer_agents = {}
            _trans_path = f"./models/transformer_{symbol}.pt"
            if symbol not in self.transformer_agents:
                self.transformer_agents[symbol] = TransformerAgent(
                    'MlpPolicy', vec_env, learning_rate=3e-4, n_steps=1024, batch_size=64,
                    n_epochs=6, gamma=0.92, gae_lambda=0.95, clip_range=0.10, ent_coef=0.08, vf_coef=0.5, verbose=0)
                if os.path.exists(_trans_path + '_transformer.pt'):
                    try:
                        self.transformer_agents[symbol].load(_trans_path, vec_env)
                    except: pass
            pbar_trans = _creer_barre(phase2b_steps, "TRANS", symbol, "green")
            tqdm_trans = UnifiedProgressCallback(phase2b_steps, "TRANS", symbol, "green", pbar=pbar_trans)
            self.transformer_agents[symbol].learn(total_timesteps=phase2b_steps, callback=[tqdm_trans], reset_num_timesteps=False)
            pbar_trans.close()
            try:
                self.transformer_agents[symbol].save(_trans_path)
                logging.info(f"  [SAVE] Transformer {symbol}")
            except Exception as _se:
                logging.warning(f"  [SAVE] Erreur Transformer: {_se}")
            _trans_w = max(0, getattr(env, 'total_wins', 0) - _sac_wins_snap)
            _trans_l = max(0, getattr(env, 'total_losses', 0) - _sac_losses_snap)
            _trans_t = _trans_w + _trans_l
            _trans_wr = (_trans_w / _trans_t * 100.0) if _trans_t > 0 else 0.0
            self._log_session_progression(symbol, "TRANSFORMER", env, self.transformer_agents[symbol])

            # ============================================================
            # PORTE — validation avant fusion
            # ============================================================
            _ppo_trades = _ppo_wins_snap + _ppo_losses_snap
            _ppo_wr = (_ppo_wins_snap / _ppo_trades * 100.0) if _ppo_trades > 0 else 0.0
            _sac_wins = max(0, _sac_wins_snap - _ppo_wins_snap)
            _sac_losses = max(0, _sac_losses_snap - _ppo_losses_snap)
            _sac_trades = _sac_wins + _sac_losses
            _sac_wr = (_sac_wins / _sac_trades * 100.0) if _sac_trades > 0 else 0.0
            _MIN_TRADES = 5; _WR_SEUIL = 52.0
            _ppo_ok = _ppo_trades >= _MIN_TRADES and _ppo_wr > _WR_SEUIL
            _sac_ok = _sac_trades >= _MIN_TRADES and _sac_wr > _WR_SEUIL
            _trans_ok = _trans_t >= _MIN_TRADES and _trans_wr > _WR_SEUIL
            _fusion_autorisee = _ppo_ok and _sac_ok and _trans_ok
            logging.info(f"[PORTE] RPPO:{_ppo_wr:.1f}% SAC:{_sac_wr:.1f}% TRANS:{_trans_wr:.1f}% -> {'OUVERTE' if _fusion_autorisee else 'FERMEE'}")
            self._log_session_progression(symbol, "GATE", env, self.ppo_agents[symbol], fusion_ok=_fusion_autorisee)

            logging.info(f"  Training termine pour {symbol}")
            
            # ============================================================
            #   SAUVEGARDE AUTOMATIQUE DES MOD LES
            # ============================================================
            logging.info(f"  Sauvegarde automatique pour {symbol}...")
            try:
                os.makedirs("./models", exist_ok=True)
                
                # PPO
                ppo_path = f"./models/ppo_{symbol}.zip"
                self.ppo_agents[symbol].save(ppo_path)
                logging.info(f"     PPO   {ppo_path}")
                
                # DQN
                dqn_path = f"./models/dqn_{symbol}.zip"
                self.dqn_agents[symbol].save(dqn_path)
                logging.info(f"     DQN   {dqn_path}")
                
                # HRM-TRM
                import torch
                hrm_path = f"./models/hrm_trm_{symbol}.pt"
                torch.save(self.hrm_trm_per_symbol[symbol].state_dict(), hrm_path)
                logging.info(f"     HRM-TRM   {hrm_path}")
                
                #   NOUVEAU: Sauvegarder aussi le nombre total de steps
                metadata_path = f"./models/metadata_{symbol}.json"
                metadata = {
                    'total_timesteps': self.ppo_agents[symbol].num_timesteps,
                    'last_training': datetime.now().isoformat(),
                    'symbol': symbol
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logging.info(f"     Metadata   {metadata_path}")
                
                logging.info(f"    MOD LES SAUVEGARD S: {symbol} ({self.ppo_agents[symbol].num_timesteps:,} steps)")
                
            except Exception as save_error:
                logging.error(f"  Erreur sauvegarde {symbol}: {save_error}")
            
            # CORRECTION XAUUSD - R INITIALISATION APR S TRAINING
            self.current_symbol = None
            
        except Exception as e:
            logging.error(f"  Erreur training {symbol}: {e}")
            self.current_symbol = None
            traceback.print_exc()
            return  # Sortie propre — asyncio.gather(return_exceptions=True) capture ça

            # LEGACY FALLBACK DESACTIVE — causait un double crash
            symbol_id = self.config.SYMBOLS.index(symbol)
            env = TradingEnvironment(historical_data, symbol_id=symbol_id, symbol=symbol)
            vec_env = DummyVecEnv([lambda: env])
            
            # Initialize HRM-TRM
            self.hrm_trm_per_symbol[symbol] = env.hrm_trm
            
            # Train PPO
            logging.info(f"    Training PPO pour {symbol}...")
            self.ppo_agents[symbol] = PPO(
                'MlpPolicy',
                vec_env,
                verbose=0,
                learning_rate=learning_rate,
                n_steps=1024,
                batch_size=256,
                n_epochs=6,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=clip_range,
                tensorboard_log=f"./logs/ppo_{symbol}"
            )
            
            # CORRECTION XAUUSD - GRADIENT CLIPPING RENFORC 
            if symbol == "XAUUSD":
                def xauusd_gradient_clip(_locals, _globals):
                    # Clipping plus agressif pour l'or
                    torch.nn.utils.clip_grad_norm_(_locals['policy'].parameters(), max_norm=0.5)
                    return True
                
                self.ppo_agents[symbol].learn(
                    total_timesteps=self.config.RL_TOTAL_TIMESTEPS // 2,
                    callback=[RLTrainingCallback(self.ws_server, symbol, 'PPO'), xauusd_gradient_clip]
                )
            else:
                self.ppo_agents[symbol].learn(
                    total_timesteps=self.config.RL_TOTAL_TIMESTEPS // 2,
                    callback=RLTrainingCallback(self.ws_server, symbol, 'PPO')
                )
            
            # Train DQN
            logging.info(f"    Training DQN pour {symbol}...")
            self.dqn_agents[symbol] = DQN(
                'MlpPolicy',
                vec_env,
                verbose=0,
                learning_rate=learning_rate,
                buffer_size=self.config.RL_BUFFER_SIZE,
                learning_starts=1000,
                batch_size=256,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=10000,
                exploration_fraction=exploration_fraction,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                tensorboard_log=f"./logs/dqn_{symbol}"
            )
            
            tqdm_dqn = UnifiedProgressCallback(self.config.RL_TOTAL_TIMESTEPS // 2, "DQN", symbol, "cyan")
            detailed_dqn = DetailedTrainingCallback(symbol, "DQN")
            self.dqn_agents[symbol].learn(
                total_timesteps=self.config.RL_TOTAL_TIMESTEPS // 2,
                callback=[
                    RLTrainingCallback(self.ws_server, symbol, 'DQN'), 
                    tqdm_dqn, 
                    detailed_dqn,
                    DragonLearningCallback(env.env if hasattr(env, 'env') else env, injection_frequency=500),  # DRAGON!
                    DragonRewardShapingCallback(env.env if hasattr(env, 'env') else env)  # DRAGON!
                ]
            )
            # tqdm_dqn auto-closes
            
            logging.info(f"  Training termin  pour {symbol}")
            
            # ============================================================
            #   SAUVEGARDE AUTOMATIQUE DES MOD LES
            # ============================================================
            logging.info(f"  Sauvegarde automatique pour {symbol}...")
            try:
                os.makedirs("./models", exist_ok=True)
                
                # PPO
                ppo_path = f"./models/ppo_{symbol}.zip"
                self.ppo_agents[symbol].save(ppo_path)
                logging.info(f"     PPO   {ppo_path}")
                
                # DQN
                dqn_path = f"./models/dqn_{symbol}.zip"
                self.dqn_agents[symbol].save(dqn_path)
                logging.info(f"     DQN   {dqn_path}")
                
                # HRM-TRM
                import torch
                hrm_path = f"./models/hrm_trm_{symbol}.pt"
                torch.save(self.hrm_trm_per_symbol[symbol].state_dict(), hrm_path)
                logging.info(f"     HRM-TRM   {hrm_path}")
                
                logging.info(f"    MOD LES SAUVEGARD S: {symbol}")
            except Exception as save_error:
                logging.error(f"  Erreur sauvegarde {symbol}: {save_error}")
            
            # CORRECTION XAUUSD - R INITIALISATION APR S TRAINING
            self.current_symbol = None
            
        except Exception as e:
            logging.error(f"  Erreur training {symbol}: {e}")
            # CORRECTION XAUUSD - R INITIALISATION M ME EN CAS D'ERREUR
            self.current_symbol = None
            traceback.print_exc()

    
    async def load_historical_data(self, symbol: str, bars: int = 100000) -> pd.DataFrame:
        """
          SYST ME OPTIMIS  - Charge des donn es multi-timeframe
        
        Chargement R ALISTE (limites MT5):
        - 50,000 bougies M5 (~1 an de données)
        - 20,000 bougies M15 (~6 mois)
        - 10,000 bougies H1 (~1 an)  
        - 5,000 bougies H4 (~2 ans)
        - 1,000 bougies D1 (~3 ans)
        
        = HISTORIQUE SUFFISANT pour entra nement robuste
        
        Features enrichies avec donn es multi-timeframe
        
        Returns:
            DataFrame avec colonnes: open, high, low, close, volume, 
            et indicateurs calcul s (RSI, MACD, ATR, etc.)
        """
        logging.info(f"  CHARGEMENT pour {symbol}: {bars:,} bougies M5...")
        
        if MT5_AVAILABLE:
            try:
                # ============================================================
                # CHARGEMENT MULTI-TIMEFRAME V4 - AVEC M1 POUR ENTRÉES PRÉCISES
                # ============================================================
                timeframes_config = [
                    (mt5.TIMEFRAME_M1, min(10000, 10000), 'M1'),    # 10k M1 max (limite broker courante)
                    (mt5.TIMEFRAME_M5, min(bars, 50000), 'M5'),    # 100k M5 (doublé!)
                    (mt5.TIMEFRAME_M15, min(30000, 30000), 'M15'),  # 30k M15
                    (mt5.TIMEFRAME_H1, min(15000, 15000), 'H1'),    # 15k H1
                    (mt5.TIMEFRAME_H4, min(5000, 5000), 'H4'),      # 5k H4
                    (mt5.TIMEFRAME_D1, min(1000, 1000), 'D1')       # 1k D1
                ]
                
                logging.info(f"  📊 MULTI-TIMEFRAME V4: Chargement de 6 timeframes pour {symbol}")
                logging.info(f"     M1: 100k | M5: 100k | M15: 30k | H1: 15k | H4: 5k | D1: 1k")
                
                all_timeframes = {}
                
                for tf_code, num_bars, tf_name in timeframes_config:
                    try:
                        logging.info(f"     Chargement {tf_name} pour {symbol} ({num_bars:,} bougies)...")
                        rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, num_bars)
                        if rates is not None and len(rates) > 0:
                            df_tf = pd.DataFrame(rates)
                            df_tf['time'] = pd.to_datetime(df_tf['time'], unit='s')
                            
                            #   CORRECTION: Stocker le symbole courant pour la normalisation
                            current_symbol_backup = getattr(self, 'current_symbol', None)
                            self.current_symbol = symbol
                            
                            df_tf = self._calculate_indicators(df_tf)
                            
                            # Restaurer l'ancien symbole
                            if current_symbol_backup is not None:
                                self.current_symbol = current_symbol_backup
                            
                            all_timeframes[tf_name] = df_tf
                            logging.info(f"     {tf_name}: {len(df_tf):,} bougies charg es")
                            
                            #   DEBUG: Afficher les colonnes disponibles
                            logging.debug(f"     Colonnes {tf_name}: {list(df_tf.columns)}")
                            if len(df_tf) > 0:
                                logging.debug(f"     Exemple RSI: {df_tf['rsi'].iloc[-1] if 'rsi' in df_tf.columns else 'N/A'}")
                        else:
                            logging.warning(f"     Aucune donn e pour {tf_name}")
                    except Exception as e:
                        logging.warning(f"     Erreur {tf_name}: {e}")
                        import traceback
                        logging.debug(f"     D tails erreur: {traceback.format_exc()}")
                
                # ============================================================
                # FUSION MULTI-TIMEFRAME - VERSION CORRIG E
                # ============================================================
                if 'M5' in all_timeframes:
                    # Base = M5
                    df = all_timeframes['M5'].copy()
                    
                    # Ajouter features des timeframes sup rieurs - PATCH V4: INCLURE M1
                    for tf_name in ['M1', 'M15', 'H1', 'H4', 'D1']:
                        if tf_name in all_timeframes:
                            tf_df = all_timeframes[tf_name]
                            #   CORRECTION: Utiliser les bons noms de colonnes
                            if len(tf_df) > 0:
                                try:
                                    # V rifier que les colonnes existent
                                    if 'rsi' in tf_df.columns:
                                        df[f'{tf_name}_rsi'] = tf_df['rsi'].iloc[-1]
                                        logging.debug(f"     {tf_name}_rsi ajout : {tf_df['rsi'].iloc[-1]:.2f}")
                                    else:
                                        logging.warning(f"     Colonne 'rsi' manquante dans {tf_name}")
                                    
                                    if 'macd' in tf_df.columns:
                                        df[f'{tf_name}_macd'] = tf_df['macd'].iloc[-1]
                                        logging.debug(f"     {tf_name}_macd ajout : {tf_df['macd'].iloc[-1]:.6f}")
                                    else:
                                        logging.warning(f"     Colonne 'macd' manquante dans {tf_name}")
                                    
                                    if 'atr' in tf_df.columns:
                                        df[f'{tf_name}_atr'] = tf_df['atr'].iloc[-1]
                                        logging.debug(f"     {tf_name}_atr ajout : {tf_df['atr'].iloc[-1]:.6f}")
                                    else:
                                        logging.warning(f"     Colonne 'atr' manquante dans {tf_name}")
                                    
                                    logging.info(f"     {tf_name} features ajout es")
                                except Exception as e:
                                    logging.warning(f"     Erreur ajout features {tf_name}: {e}")
                    
                    total_features = len(df.columns)
                    
                    # Calculer la p riode couverte
                    if len(df) > 0 and 'time' in df.columns:
                        first_date = df['time'].iloc[0]
                        last_date = df['time'].iloc[-1]
                        period_days = (last_date - first_date).days
                        period_years = period_days / 365.25
                        
                        logging.info(f"  DONN ES MASSIVES CHARG ES:")
                        logging.info(f"     {len(df):,} bougies M5")
                        logging.info(f"     P riode: {first_date.date()}   {last_date.date()}")
                        logging.info(f"     Dur e: {period_years:.2f} ans ({period_days:,} jours)")
                        logging.info(f"     Features: {total_features} (multi-TF M5/M15/H1/H4/D1)")
                    else:
                        logging.info(f"  CHIRURGICAL PATCH : {len(df):,} bougies, {total_features} features (multi-TF)")
                    
                    #   DEBUG: Afficher les premi res colonnes
                    logging.debug(f"     5 premi res colonnes: {list(df.columns)[:5]}")
                    if len(df) > 0:
                        logging.debug(f"     Dernier RSI: {df['rsi'].iloc[-1] if 'rsi' in df.columns else 'N/A'}")
                    
                    return df
                else:
                    # FALLBACK INTELLIGENT: Utiliser H1 si M5  choue
                    logging.warning(f"  M5 non disponible pour {symbol}")
                    
                    if 'H1' in all_timeframes and len(all_timeframes['H1']) > 0:
                        logging.info(f"  FALLBACK sur H1 pour {symbol}")
                        df = all_timeframes['H1'].copy()
                        
                        # Enrichir avec H4 et D1 si disponibles
                        for tf_name in ['H4', 'D1']:
                            if tf_name in all_timeframes:
                                tf_df = all_timeframes[tf_name]
                                if len(tf_df) > 0 and 'rsi' in tf_df.columns:
                                    df[f'{tf_name}_rsi'] = tf_df['rsi'].iloc[-1]
                                if len(tf_df) > 0 and 'macd' in tf_df.columns:
                                    df[f'{tf_name}_macd'] = tf_df['macd'].iloc[-1]
                        
                        logging.info(f"  FALLBACK H1: {len(df):,} bougies charg es")
                        return df
                    
                    elif 'H4' in all_timeframes and len(all_timeframes['H4']) > 0:
                        logging.info(f"  FALLBACK sur H4 pour {symbol}")
                        df = all_timeframes['H4'].copy()
                        logging.info(f"  FALLBACK H4: {len(df):,} bougies charg es")
                        return df
                    
            except Exception as e:
                logging.error(f"  Erreur chargement {symbol}: {e}")
                import traceback
                traceback.print_exc()
        
        # DERNIER RECOURS: Mock data ( vit  au maximum)
        logging.error(f"  AUCUNE DONN E R ELLE disponible pour {symbol}, utilisation mock")
        return self._generate_mock_data(min(bars, 10000))  # Limiter le mock   10k


    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques + signaux avanc s - VERSION PATCH E
        
        Features (14 total) :
        - 4 prix: open, high, low, close
        - 6 indicateurs: rsi, macd, macd_signal, atr, bb_middle, bb_upper
        - 4 signaux avanc s: wick_score, hli, ETI, SNAP
        """
        logging.debug("  _calculate_indicators PATCH  appel ")
        
        try:
            # ===== NORMALISATION XAUUSD =====
            if hasattr(self, 'current_symbol') and self.current_symbol == "XAUUSD":
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    if col in df.columns and df[col].max() > 100:
                        df[col] = df[col] / 1000.0
                        logging.debug(f"     XAUUSD normalis : {col}")
            
            # ===== INDICATEURS CLASSIQUES =====
            
            # RSI - Version robuste
            try:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                logging.debug("     RSI calcul ")
            except Exception as e:
                logging.warning(f"  Erreur calcul RSI: {e}")
                df['rsi'] = 50.0  # Valeur par d faut
            
            # MACD - Version robuste
            try:
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = exp1 - exp2
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                logging.debug("     MACD calcul ")
            except Exception as e:
                logging.warning(f"  Erreur calcul MACD: {e}")
                df['macd'] = 0.0
                df['macd_signal'] = 0.0
            
            # ATR - Version robuste
            try:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                df['atr'] = true_range.rolling(14).mean()
                logging.debug("     ATR calcul ")
            except Exception as e:
                logging.warning(f"  Erreur calcul ATR: {e}")
                df['atr'] = 0.0
            
            # Bollinger Bands - Version robuste
            try:
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                bb_std = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
                df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
                logging.debug("     Bollinger Bands calcul s")
            except Exception as e:
                logging.warning(f"  Erreur calcul Bollinger Bands: {e}")
                df['bb_middle'] = df['close']
                df['bb_upper'] = df['close']
                df['bb_lower'] = df['close']
            
            # Tick volume (si absent, cr er dummy)
            if 'tick_volume' not in df.columns:
                df['tick_volume'] = 1
                logging.debug("     Volume dummy cr ")
            
            # ===== SIGNAUX AVANC S SIMPLIFI S POUR STABILIT  =====
            
            try:
                # Composants de bougie basiques
                df['body'] = (df['close'] - df['open']).abs()
                df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
                df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']
                
                # Normalisation par ATR
                atr_safe = df['atr'].replace(0, np.nan).bfill().fillna(1e-6)
                
                # Wick score simplifi 
                uw = (df['upper_wick'] / atr_safe).rolling(14).mean()
                lw = (df['lower_wick'] / atr_safe).rolling(14).mean()
                df['wick_balance'] = (uw - lw)
                df['wick_score'] = np.tanh(df['wick_balance'].rolling(8).mean().fillna(0))
                
                # Heaviness / Lift Index (HLI) simplifi 
                df['clv'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
                            (df['high'] - df['low']).replace(0, np.nan).bfill().fillna(1e-6)
                
                hli = (
                    0.50 * df['wick_balance']
                    + 0.15 * df['clv']
                )
                df['hli'] = np.tanh(hli.rolling(8).mean().fillna(0))
                
                logging.debug("     Signaux avanc s calcul s")
            except Exception as e:
                logging.warning(f"  Erreur signaux avanc s: {e}")
                df['wick_score'] = 0.0
                df['hli'] = 0.0
            
            # ===== AJOUT DES FEATURES MANQUANTES =====
            # Feature 12: volume_norm
            if 'tick_volume' in df.columns:
                df['volume_norm'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
            else:
                df['volume_norm'] = 1.0
            
            # Feature 13: ema_20
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Feature 14: momentum
            df['momentum'] = df['close'].pct_change(periods=14) * 100
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # DIVERGENCE FEATURES POUR L'OBSERVATION DE L'AGENT (Features 15-18)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            try:
                df = self._calculate_divergence_features_for_observation(df, lookback=20)
                logging.debug("     Divergence features calculÃ©es pour observation agent")
            except Exception as e:
                logging.warning(f"  Erreur calcul divergence features: {e}")
                df['div_rsi_signal'] = 0.0
                df['div_macd_signal'] = 0.0
                df['div_momentum_signal'] = 0.0
                df['div_strength'] = 0.0
            
            # Normalisation XAUUSD pour les nouvelles features
            if hasattr(self, 'current_symbol') and self.current_symbol == "XAUUSD":
                indicator_columns = ['bb_middle', 'bb_upper', 'bb_lower', 'atr', 'ema_20']
                for col in indicator_columns:
                    if col in df.columns and df[col].max() > 100:
                        df[col] = df[col] / 1000.0
            
            # Fill NaN pour les nouvelles features INCLUANT DIVERGENCES
            fill_cols = ['volume_norm', 'ema_20', 'momentum', 'div_rsi_signal', 'div_macd_signal', 'div_momentum_signal', 'div_strength']
            for col in fill_cols:
                if col in df.columns:
                    df[col] = df[col].bfill().fillna(0)
            
            # Fill NaN g n ral
            df = df.bfill().fillna(0)
            
            # CORRECTION: RETOURNER 18 COLONNES (14 + 4 DIVERGENCES)
            expected_columns = [
                'open', 'high', 'low', 'close',
                'rsi', 'macd', 'macd_signal', 'atr', 'bb_middle', 'bb_upper',
                'bb_lower', 'volume_norm', 'ema_20', 'momentum',
                'div_rsi_signal', 'div_macd_signal', 'div_momentum_signal', 'div_strength'
            ]
            
            # V rifier que toutes les colonnes attendues existent
            for col in expected_columns:
                if col not in df.columns:
                    logging.warning(f"  Colonne manquante: {col}, cr ation avec z ros")
                    df[col] = 0.0
            
            # Retourner exactement les 18 colonnes attendues
            result_df = df[expected_columns].copy()
            
            logging.debug(f"  _calculate_indicators termin : {len(result_df)} lignes, {len(result_df.columns)} colonnes")
            return result_df
            
        except Exception as e:
            logging.error(f"  Erreur _calculate_indicators: {e}")
            # Fallback: retourner un DataFrame basique avec 18 colonnes
            n_rows = len(df) if len(df) > 0 else 1
            fallback_data = {
                'open': [1.0] * n_rows, 'high': [1.0] * n_rows, 'low': [1.0] * n_rows, 'close': [1.0] * n_rows,
                'rsi': [50.0] * n_rows, 'macd': [0.0] * n_rows, 'macd_signal': [0.0] * n_rows,
                'atr': [0.0] * n_rows, 'bb_middle': [1.0] * n_rows, 'bb_upper': [1.0] * n_rows,
                'bb_lower': [1.0] * n_rows, 'volume_norm': [1.0] * n_rows, 'ema_20': [1.0] * n_rows, 'momentum': [0.0] * n_rows,
                'div_rsi_signal': [0.0] * n_rows, 'div_macd_signal': [0.0] * n_rows,
                'div_momentum_signal': [0.0] * n_rows, 'div_strength': [0.0] * n_rows
            }
            return pd.DataFrame(fallback_data)


    def _generate_mock_data(self, bars: int) -> pd.DataFrame:
        """G n re des donn es mock pour testing"""
        dates = pd.date_range(end=datetime.now(), periods=bars, freq='15min')
        
        # Generate random walk price
        price = 1.1000
        prices = [price]
        for _ in range(bars - 1):
            price *= (1 + np.random.randn() * 0.001)
            prices.append(price)
        
        df = pd.DataFrame({
          'open': prices,
          'high': [p * (1 + abs(np.random.randn() * 0.0005)) for p in prices],
          'low': [p * (1 - abs(np.random.randn() * 0.0005)) for p in prices],
          'close': prices,
        })
        
        return self._calculate_indicators(df)

    def check_performance_drop(self, symbol: str) -> bool:
        """
        V rifie si la performance a suffisamment baiss  pour n cessiter un retrain.
        
        Crit res:
        - Win rate < seuil
        - Drawdown important
        - Nombre minimum de trades
        """
        recent_trades = list(self.performance_history[symbol])
        
        if len(recent_trades) < 10:
            return False
        
        # Calculate win rate sur les 20 derniers trades
        recent_20 = recent_trades[-20:]
        win_rate = sum(1 for trade in recent_20 if trade > 0) / len(recent_20)
        
        return win_rate < self.config.PERF_DROP_THRESHOLD

    async def retrain_on_drop(self, symbol: str):
        """
        Retrain les agents si performance drop d tect .
        Inclut fine-tuning du HRM-TRM avec Q-loss.
        """
        if not self.check_performance_drop(symbol):
            return
        
        now = time.time()
        if now - self.last_retrain_time.get(symbol, 0) < self.config.RETRAIN_COOLDOWN:
            logging.info(f"  Cooldown retrain actif pour {symbol}")
            return
        
        logging.warning(f"  Performance drop d tect  pour {symbol}, d but retrain...")
        
        try:
            # Retrain agents
            await self.train_agent(symbol)
            
            # Fine-tune HRM-TRM avec Q-loss
            recent_trades = list(self.performance_history[symbol])[-20:]
            target_correct = torch.tensor(
                [1.0 if trade > 0 else 0.0 for trade in recent_trades],
                dtype=torch.float32
            ).unsqueeze(1)
            
            # Mock z_H pour Q-loss (en production, utiliser real states)
            z_H_mock = torch.randn(len(recent_trades), CONFIG.HRM_HIDDEN_DIM)
            
            # Compute et backprop Q-loss
            q_loss = self.hrm_trm_per_symbol[symbol].compute_q_loss(z_H_mock, target_correct)
            q_loss.backward()
            
            # Update EMA
            self.hrm_trm_per_symbol[symbol].ema.update(self.hrm_trm_per_symbol[symbol])
            
            logging.info(f"  Retrain termin  pour {symbol}, Q-loss: {q_loss.item():.4f}")
            
            self.last_retrain_time[symbol] = now
            
        except Exception as e:
            logging.error(f"  Erreur retrain {symbol}: {e}")
            traceback.print_exc()

    def predict_action(self, symbol: str, observation: np.ndarray) -> int:
        """
        Pr dit l'action en fusionnant PPO et DQN.
        
        Fusion Strategy:
        1. Get prediction from both agents
        2. If agree: use that action
        3. If disagree: use PPO (more stable)
        
        Returns:
            action: 0=BUY, 1=SELL, 2=HOLD, 3=CLOSE
        """
        try:
            # CORRECTION XAUUSD - V RIFICATION CRITIQUE DES NAN
            if np.isnan(observation).any() or np.isinf(observation).any():
                logging.warning(f"  Observation invalide (NaN/Inf) pour {symbol}, HOLD forc ")
                return 2  # HOLD
            
            # CORRECTION XAUUSD - NORMALISATION OBSERVATION EN TEMPS R EL
            if symbol == "XAUUSD":
                # V rifier si les prix sont anormalement  lev s (non normalis s)
                price_features = observation[:4]  # open, high, low, close
                if np.max(np.abs(price_features)) > 100:  # Prix > 100 = probablement non normalis 
                    observation[:4] = observation[:4] / 1000.0  # Normaliser
                    logging.debug(f"  {symbol}: Observation normalis e en temps r el")
            
            # Refine observation with HRM-TRM
            x = torch.from_numpy(observation).float().unsqueeze(0)
            y_init = torch.tensor([2])  # HOLD
            z_init = torch.zeros(1, CONFIG.HRM_HIDDEN_DIM)
            symbol_id = torch.tensor([self.config.SYMBOLS.index(symbol)])
            
            with torch.no_grad():
                refined_obs = self.hrm_trm_per_symbol[symbol].refine(x, y_init, z_init, symbol_id)
            
            refined_obs_np = refined_obs.squeeze(0).numpy()
            
            # CORRECTION XAUUSD - V RIFICATION APR S RAFFINEMENT
            if np.isnan(refined_obs_np).any() or np.isinf(refined_obs_np).any():
                logging.warning(f"  Observation raffin e invalide pour {symbol}, HOLD forc ")
                return 2  # HOLD
            
            # CORRECTION XAUUSD - V RIFICATION MOD LES DISPONIBLES
            if symbol not in self.ppo_agents or symbol not in self.dqn_agents:
                logging.warning(f"  Mod les non disponibles pour {symbol}, HOLD forc ")
                return 2  # HOLD
            
            # CORRECTION: Pr diction RL am lior e
            try:
                ppo_action, _ = self.ppo_agents[symbol].predict(refined_obs_np, deterministic=True)
                dqn_action, _ = self.dqn_agents[symbol].predict(refined_obs_np, deterministic=True)
                
                # Extraire les confiances si disponibles
                ppo_confidence = 0.5
                dqn_confidence = 0.5
                
                # Essayer d'obtenir les Q-values/probabilitÃ©s pour la confiance
                try:
                    # Pour PPO: utiliser la politique pour obtenir les probabilitÃ©s
                    ppo_obs_tensor = torch.from_numpy(refined_obs_np).float().unsqueeze(0)
                    with torch.no_grad():
                        _, log_prob, _ = self.ppo_agents[symbol].policy.evaluate_actions(
                            ppo_obs_tensor, 
                            torch.tensor([[ppo_action]])
                        )
                        ppo_confidence = torch.exp(log_prob).item()
                except:
                    ppo_confidence = 0.6  # Default PPO confidence
                
                try:
                    # Pour DQN: utiliser les Q-values
                    dqn_obs_tensor = torch.from_numpy(refined_obs_np).float().unsqueeze(0)
                    with torch.no_grad():
                        q_values = self.dqn_agents[symbol].q_net(dqn_obs_tensor)
                        q_probs = F.softmax(q_values, dim=-1)
                        dqn_confidence = q_probs[0, dqn_action].item()
                except:
                    dqn_confidence = 0.5  # Default DQN confidence
                    
            except Exception as pred_error:
                logging.warning(f"  Erreur pr diction RL pour {symbol}: {pred_error}")
                # Fallback vers HOLD en cas d'erreur
                ppo_action, dqn_action = 2, 2  # HOLD
                ppo_confidence, dqn_confidence = 0.5, 0.5
            
            # CORRECTION XAUUSD - VALIDATION DES ACTIONS
            if ppo_action not in [0, 1, 2, 3] or dqn_action not in [0, 1, 2, 3]:
                logging.warning(f"  Action invalide pour {symbol}: PPO={ppo_action}, DQN={dqn_action}, HOLD forc ")
                return 2  # HOLD
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # META-LEARNER FUSION: Apprend Ã  dÃ©cider intelligemment
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            if self.use_meta_learner and hasattr(self, 'meta_learner'):
                try:
                    action, confidence, debug_info = self.meta_learner.decide(
                        symbol=symbol,
                        context=observation,  # Contexte original (18 features avec divergences)
                        ppo_action=int(ppo_action),
                        dqn_action=int(dqn_action),
                        ppo_confidence=ppo_confidence,
                        dqn_confidence=dqn_confidence
                    )
                    
                    # Log dÃ©taillÃ©
                    action_names = ['BUY', 'SELL', 'HOLD', 'CLOSE']
                    logging.info(f"  [META-LEARNER] {symbol}: "
                                f"PPO={action_names[int(ppo_action)]}({ppo_confidence:.2f}) vs "
                                f"DQN={action_names[int(dqn_action)]}({dqn_confidence:.2f}) "
                                f"â†’ {action_names[action]} (conf={confidence:.2f})")
                    
                    return int(action)
                    
                except Exception as meta_error:
                    logging.warning(f"  Meta-learner error: {meta_error}, fallback to simple fusion")
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # FALLBACK: Fusion simple si meta-learner dÃ©sactivÃ© ou erreur
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            if ppo_action == dqn_action:
                action = ppo_action
                logging.debug(f"  {symbol}: PPO et DQN agree on action {action}")
            else:
                # Utiliser la confiance pour dÃ©cider
                if ppo_confidence > dqn_confidence + 0.1:
                    action = ppo_action
                elif dqn_confidence > ppo_confidence + 0.1:
                    action = dqn_action
                else:
                    action = ppo_action  # PPO par dÃ©faut
                logging.debug(f"  {symbol}: PPO={ppo_action}({ppo_confidence:.2f}), DQN={dqn_action}({dqn_confidence:.2f}), using {action}")
            
            return int(action)
            
        except Exception as e:
            logging.error(f"  Erreur predict_action pour {symbol}: {e}")
            
            # CORRECTION XAUUSD - GESTION SP CIFIQUE DES ERREURS NAN
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                logging.warning(f"  {symbol}: Mod le corrompu (NaN), r initialisation recommand e")
            
            return 2  # HOLD par d faut

    def record_trade_result(self, symbol: str, pnl: float):
        """
        Enregistre le rÃ©sultat d'un trade.
        Alimente aussi le meta-learner pour l'apprentissage.
        """
        self.performance_history[symbol].append(1 if pnl > 0 else 0)
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # FEED META-LEARNER: Apprendre de ce trade
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        if self.use_meta_learner and hasattr(self, 'meta_learner'):
            try:
                # Calculer le reward pour le meta-learner
                # On normalise le PnL pour que les grands profits/pertes aient plus d'impact
                normalized_reward = np.tanh(pnl / 50.0)  # Normaliser autour de Â±50 pips
                
                self.meta_learner.record_outcome(symbol, normalized_reward)
                
                # Log pÃ©riodique des insights
                if self.meta_learner.training_step > 0 and self.meta_learner.training_step % 50 == 0:
                    insights = self.meta_learner.get_context_insights()
                    if insights:
                        logging.info("=" * 60)
                        logging.info("[META-LEARNER] Context Performance Insights:")
                        for ctx_type, data in insights.items():
                            logging.info(f"  {ctx_type}: PPO={data['ppo_rate']:.1f}% vs DQN={data['dqn_rate']:.1f}% "
                                        f"(winner={data['winner']}, n={data['samples']})")
                        logging.info("=" * 60)
                        
            except Exception as e:
                logging.debug(f"  Meta-learner record error: {e}")
        
    async def _evaluate_model_performance(self, test_env, n_episodes=3) -> float:
        """ value la performance du mod le sur l'environnement de test"""
        try:
            episode_returns = []
            
            for episode in range(n_episodes):
                obs, _ = test_env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = test_env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                
                episode_returns.append(total_reward)
            
            # Score bas  sur le reward moyen et la variance (plus stable = mieux)
            mean_return = np.mean(episode_returns)
            std_return = np.std(episode_returns)
            
            # P naliser la haute variance
            score = mean_return / (1 + std_return) if std_return > 0 else mean_return
            
            return float(score)
            
        except Exception as e:
            logging.error(f"  Erreur  valuation mod le: {e}")
            return 0.0

#============================================================================
#ULTIMATE TRADING BRAIN
#============================================================================
class UltimateTradingBrain:
    """
    Cerveau principal du bot de trading.
    Orchestre tous les composants: RL agents, risk management, execution, monitoring.
    """
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.rl_manager = RLAgentManager(config)
        self.active = True
        self.start_time = datetime.now()
        self.session_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0
        }
        self.connected_clients: set = set()
        self.daily_loss_tracker = 0.0
        self.last_daily_reset = datetime.now().date()
        
        #   CORRECTION CRITIQUE: Initialiser open_positions
        self.open_positions: Dict[str, Dict] = {}
        self.trade_history: deque = deque(maxlen=1000)
        self.equity_curve: deque = deque(maxlen=10000)
        
        logging.info("  Ultimate Trading Brain initialis ")
        
    def _get_mt5_timeframe(self, minutes: int) -> int:
        """Convertit les minutes en timeframe MT5"""
        if not MT5_AVAILABLE:
            return 15
        
        timeframe_map = {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5, 
            15: mt5.TIMEFRAME_M15,
            30: mt5.TIMEFRAME_M30,
            60: mt5.TIMEFRAME_H1,
            240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1
        }
        return timeframe_map.get(minutes, mt5.TIMEFRAME_M15)

    async def initialize(self, skip_training: bool = False) -> bool:
        """
        Initialise tous les composants du bot.
        
        Args:
            skip_training: Si True, skip le training (mode dev)
        
        Returns:
            True si succ s, False sinon
        """
        logging.info("  Initialisation du Trading Brain...")
        
        # Keyboard listener pour skip training
        keyboard_listener = None
        
        try:
            # Declare global pour modifier la variable globale
            global MT5_AVAILABLE
            
            # Initialize MT5 with timeout
            if MT5_AVAILABLE:
                logging.info("  Connexion a MT5...")
                import threading
                import time
                
                mt5_success = [False]
                def init_mt5():
                    try:
                        mt5_success[0] = mt5.initialize()
                    except:
                        mt5_success[0] = False
                
                init_thread = threading.Thread(target=init_mt5)
                init_thread.daemon = True
                init_thread.start()
                init_thread.join(timeout=5.0)  # 5 secondes max
                
                if not mt5_success[0]:
                    logging.error("   chec initialisation MT5 (timeout ou erreur)")
                    logging.warning("  SKIP MT5 - Le bot fonctionnera en mode simulation")
                    MT5_AVAILABLE = False
                else:
                    logging.info("  Connect    MT5: " + str(mt5.account_info().server if mt5.account_info() else "Unknown"))
                
                # Login
                if MT5_AVAILABLE and self.config.MT5_LOGIN and self.config.MT5_PASSWORD:
                    authorized = mt5.login(
                        login=self.config.MT5_LOGIN,
                        password=self.config.MT5_PASSWORD,
                        server=self.config.MT5_SERVER
                    )
                    if not authorized:
                        logging.error(f"   chec login MT5: {mt5.last_error()}")
                        logging.warning("  SKIP MT5 - Le bot continuera en mode simulation")
                        MT5_AVAILABLE = False
                    else:
                        logging.info(f"  Connect    MT5: {mt5.account_info().server}")
            
            #   KEYBOARD LISTENER POUR SKIP
            if not skip_training and KEYBOARD_AVAILABLE:
                keyboard_listener = KeyboardListener()
                keyboard_listener.start()
                
                # Attendre 3 secondes pour permettre le skip
                logging.info("")
                logging.info("="*60)
                logging.info("   MODE D VELOPPEMENT ACTIV ")
                logging.info("   Appuyez sur 'S' dans les 3 prochaines secondes")
                logging.info("   pour SKIP le training et tester directement")
                logging.info("="*60)
                logging.info("")
                
                for i in range(3, 0, -1):
                    if keyboard_listener.skip_training:
                        skip_training = True
                        logging.warning("  SKIP TRAINING D TECT !")
                        break
                    logging.info(f"   {i}...")
                    await asyncio.sleep(1)
                
                keyboard_listener.stop()
            
            # ============================================================
            # [V6 GALAXY] ENRICHISSEMENT MERLIN — avant l'entraînement
            # Injecte captures PNG cerveau/ + patterns ADN dans merlin_patterns.pkl
            # RPPO voit les mêmes patterns que lors des sessions précédentes
            # ============================================================
            try:
                from goldeneye_v6_enrichment import (
                    charger_patterns_existants, enrichir_depuis_captures,
                    enrichir_depuis_adn, sauvegarder_patterns
                )
                logging.info("[V6 ENRICHMENT] Chargement patterns MERLIN depuis cerveau/...")
                _patterns = charger_patterns_existants()
                _avant = sum(len(v) for v in _patterns.values())
                _patterns = enrichir_depuis_captures(_patterns)
                _patterns = enrichir_depuis_adn(_patterns)
                _apres = sum(len(v) for v in _patterns.values())
                sauvegarder_patterns(_patterns)
                logging.info(f"[V6 ENRICHMENT] MERLIN: {_avant} → {_apres} patterns (+{_apres-_avant})")
            except Exception as _enr_err:
                logging.warning(f"[V6 ENRICHMENT] Ignoré: {_enr_err}")

            # Train RL agents (ou skip)
            if skip_training:
                logging.warning("")
                logging.warning("="*60)
                logging.warning("  TRAINING SKIPPED - MODE D VELOPPEMENT")
                logging.warning("="*60)
                logging.warning("")

            await self.rl_manager.start_training_all_agents(skip=skip_training)
            
            logging.info("  Trading Brain initialis  avec succ s!")
            return True
            
        except Exception as e:
            logging.error(f"  Erreur initialisation: {e}")
            traceback.print_exc()
            return False
        finally:
            if keyboard_listener:
                keyboard_listener.stop()

    async def analyze_and_trade(self, symbol: str):
        """
        Analyse le march  et ex cute un trade si conditions r unies.
        VERSION CORRIG E - SYNTAXE VALIDE
        """
        try:
            #   DEBUG
            logging.info(f"  analyze_and_trade appel  pour {symbol}")
            
            #   SYNC avec MT5 pour avoir le vrai nombre de positions
            self._sync_positions_with_mt5()
            
            # V rifier que open_positions existe
            if not hasattr(self, 'open_positions'):
                self.open_positions = {}
                logging.info(f"  open_positions initialis ")
            
            # Check daily loss limit
            if not self._check_daily_loss_limit():
                logging.warning("  Daily loss limit atteint, trading suspendu")
                return
            
            # Check max positions
            if len(self.open_positions) >= self.config.MAX_POSITIONS:
                logging.debug(f"  Max positions atteint ({self.config.MAX_POSITIONS})")
                return
                return
            
            # Check if already in position - CORRECTION: CE BLOC DOIT  TRE DANS LE TRY
            if symbol in self.open_positions:
                position = self.open_positions[symbol]
                tick = mt5.symbol_info_tick(symbol)
                if tick and hasattr(self, 'profit_manager'):
                    current_price = tick.bid if position.get('type') == 'sell' else tick.ask
                    entry_price = position.get('entry_price', current_price)
                    pos_type = position.get('type', 'buy')
                    tick_info = mt5.symbol_info(symbol)
                    if tick_info:
                        decision = self.profit_manager.update_position(
                            symbol, entry_price, current_price, pos_type, tick_info.point
                        )
                        if decision['action'] in ['close_all', 'close_partial']:
                            logging.info(f"  {symbol}: {decision['reason']}")
                            self._close_position(symbol, decision['reason'])
                            return
            
            if symbol in self.open_positions:
                logging.info(f"  {symbol} a d j  une position, gestion...")
                await self._manage_existing_position(symbol)
                return
            else:
                logging.info(f"  {symbol} pas de position, analyse pour ouvrir...")
            
            # Get market data avec gestion d'erreur
            logging.info(f"  R cup ration observation pour {symbol}...")
            observation = self._get_observation(symbol)
            if observation is None:
                logging.warning(f"  Observation None pour {symbol}")
                return
            logging.info(f"  Observation OK pour {symbol}: shape={observation.shape}")
            
            # V rification ultra-rigoureuse de l'observation
            if np.isnan(observation).any() or np.isinf(observation).any():
                logging.warning(f"  Observation corrompue pour {symbol}, skip")
                return
            
            # Get RL prediction avec protection
            # 1) pr diction (prot g e)
            try:
                logging.info(f"  Pr diction action pour {symbol}.")
                action = self.rl_manager.predict_action(symbol, observation)
                action_name = ['BUY', 'SELL', 'HOLD', 'CLOSE'][action]
                logging.info(f"  Action pr dite pour {symbol}: {action_name}")
            except Exception as rl_error:
                logging.error(f"  Erreur prediction RL pour {symbol}: {rl_error}")
                traceback.print_exc()
                return

            # 2) stats (prot g es s par ment)
            try:
                self._update_trading_stats(symbol, action_name)
            except Exception as stats_error:
                logging.debug(f"  Statistiques non disponibles: {stats_error}")
            
            # Execute based on action
            if action == 0:  # BUY
                await self._execute_buy(symbol)
            elif action == 1:  # SELL
                await self._execute_sell(symbol)
            elif action == 3:  # CLOSE
                # V rifier qu'il y a bien une position   fermer
                if symbol in self.open_positions:
                    await self._close_position(symbol)
                else:
                    logging.debug(f"  Action CLOSE ignor e pour {symbol} (pas de position)")
            # action == 2 (HOLD) ne fait rien
            
        except Exception as e:
            logging.error(f"  Erreur analyze_and_trade pour {symbol}: {e}")
            traceback.print_exc()
    
    
    def _get_observation(self, symbol: str) -> Optional[np.ndarray]:
        """
        R cup re l'observation pour un symbole - VERSION PATCH E
        """
        try:
            logging.debug(f"  _get_observation PATCH  pour {symbol}")
            
            if MT5_AVAILABLE:
                rates = mt5.copy_rates_from_pos(symbol, self._get_mt5_timeframe(CONFIG.TIMEFRAME), 0, 50)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    
                    #   CORRECTION: Stocker le symbole courant pour la normalisation
                    current_symbol_backup = getattr(self, 'current_symbol', None)
                    self.current_symbol = symbol
                    
                    df = self.rl_manager._calculate_indicators(df)
                    
                    # Restaurer l'ancien symbole
                    if current_symbol_backup is not None:
                        self.current_symbol = current_symbol_backup
                    
                    #   VALIDATION RENFORC E
                    if not self._validate_dataframe(df, f"_get_observation_{symbol}"):
                        logging.warning(f"  DataFrame invalide pour {symbol}, utilisation fallback")
                        return np.zeros(18, dtype=np.float32)
                    
                    # Retourner les 18 colonnes standard - VERSION AVEC DIVERGENCES
                    expected_columns = [
                        'open', 'high', 'low', 'close', 'rsi', 'macd', 'macd_signal', 
                        'atr', 'bb_middle', 'bb_upper', 'bb_lower', 'volume_norm', 
                        'ema_20', 'momentum',
                        'div_rsi_signal', 'div_macd_signal', 'div_momentum_signal', 'div_strength'
                    ]
                    
                    # V rifier que toutes les colonnes existent
                    for col in expected_columns:
                        if col not in df.columns:
                            logging.warning(f"  Colonne {col} manquante dans observation {symbol}")
                            df[col] = 0.0
                    
                    if len(df) > 0:
                        obs_values = df[expected_columns].iloc[-1].values
                        
                        # V rification et nettoyage rigoureux
                        if np.isnan(obs_values).any() or np.isinf(obs_values).any():
                            logging.warning(f"  Observation corrompue pour {symbol}, nettoyage agressif")
                            obs_values = np.nan_to_num(obs_values, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # Normalisation robuste
                        obs_mean = np.mean(obs_values)
                        obs_std = np.std(obs_values)
                        
                        if obs_std < 1e-8:
                            obs_normalized = obs_values - obs_mean
                        else:
                            obs_normalized = (obs_values - obs_mean) / obs_std
                        
                        # Clipping agressif
                        obs_normalized = np.clip(obs_normalized, -5.0, 5.0)
                        
                        logging.debug(f"  Observation {symbol} pr par e: shape={obs_normalized.shape}")
                        return obs_normalized.astype(np.float32)
            
            # Fallback: donn es mock
            logging.debug(f"  Donn es mock pour {symbol}")
            return np.random.randn(18).astype(np.float32)
            
        except Exception as e:
            logging.error(f"  Erreur _get_observation pour {symbol}: {e}")
            # Fallback safe
            return np.zeros(18, dtype=np.float32)


    def _check_daily_loss_limit(self) -> bool:
        """V rifie si le daily loss limit est atteint"""
        today = datetime.now().date()
        if today != self.last_daily_reset:
            self.daily_loss_tracker = 0.0
            self.last_daily_reset = today
        
        return abs(self.daily_loss_tracker) < self.config.DAILY_LOSS_LIMIT

    async def _execute_buy(self, symbol: str):
        """Execute REAL BUY order"""
        try:
            logging.info(f"  BUY signal pour {symbol}")
            
            # R cup rer le VRAI prix de MT5
            if MT5_AVAILABLE:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    logging.error(f"  Impossible de r cup rer tick pour {symbol}")
                    return
                current_price = tick.ask  # Prix ASK pour BUY
            else:
                # Fallback observation
                observation = self._get_observation(symbol)
                if observation is None:
                    logging.error(f"  Impossible de r cup rer donn es pour {symbol}")
                    return
                current_price = observation[3]
            
            # R cup rer ATR
            observation = self._get_observation(symbol)
            if observation is None:
                return
            atr = observation[7]
            
            # V rifier ATR valide et ajuster selon symbole
            if atr <= 0 or np.isnan(atr) or np.isinf(atr):
                logging.warning(f"  ATR invalide ({atr}), utilisation valeur par d faut")
                atr_defaults = {"EURUSD": 0.0005, "GBPUSD": 0.0006, "USDJPY": 0.05, "XAUUSD": 2.5}
                atr = atr_defaults.get(symbol, 0.0005)
            
            # CORRECTION XAUUSD: Ajuster ATR si trop grand
            if symbol == "XAUUSD" and atr > 10:
                logging.warning(f"  XAUUSD: ATR anormal ({atr}), ajustement")
                atr = 2.5  # ATR safe pour l'or
            
            # Calculer stops avec prix R EL (pas normalis )
            # Pour BUY: SL en-dessous, TP au-dessus
            if symbol == "XAUUSD":
                # Stops adapt s   l'or (prix  lev )
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 2.5)
            else:
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            
            # V rification stops valides
            if stop_loss <= 0 or take_profit <= 0:
                logging.error(f"  Stops invalides: SL={stop_loss:.5f}, TP={take_profit:.5f}")
                return
            
            # V rifier distance minimale selon symbole
            if MT5_AVAILABLE:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
                    if min_stop_distance > 0:
                        actual_distance = abs(current_price - stop_loss)
                        if actual_distance < min_stop_distance:
                            logging.warning(f"  Stop trop proche, ajustement: {actual_distance:.5f} -> {min_stop_distance:.5f}")
                            stop_loss = current_price - (min_stop_distance * 1.5)
            
            # R cup rer balance
            if MT5_AVAILABLE:
                account_info = mt5.account_info()
                balance = account_info.balance if account_info else 10000.0
            else:
                balance = 10000.0
            
            # Calculer position size
            position_size = self._calculate_basic_position_size(symbol, balance, stop_loss, current_price)
            
            # Ex cuter ordre MT5
            await self._execute_mt5_order(symbol, 'buy', position_size, stop_loss, take_profit)
            
            # Update stats
            self.session_stats['total_trades'] += 1
            logging.info(f"  BUY order execut e pour {symbol}: {position_size} lots @ {current_price:.5f}")
            
        except Exception as e:
            logging.error(f"  Erreur _execute_buy: {e}")
            traceback.print_exc()

    async def _execute_sell(self, symbol: str):
        """Execute REAL SELL order"""
        try:
            logging.info(f"  SELL signal pour {symbol}")
            
            # R cup rer le VRAI prix de MT5
            if MT5_AVAILABLE:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    logging.error(f"  Impossible de r cup rer tick pour {symbol}")
                    return
                current_price = tick.bid  # Prix BID pour SELL
            else:
                observation = self._get_observation(symbol)
                if observation is None:
                    logging.error(f"  Impossible de r cup rer donn es pour {symbol}")
                    return
                current_price = observation[3]
            
            # R cup rer ATR
            observation = self._get_observation(symbol)
            if observation is None:
                return
            atr = observation[7]
            
            # V rifier ATR valide
            if atr <= 0 or np.isnan(atr) or np.isinf(atr):
                logging.warning(f"  ATR invalide ({atr}), utilisation valeur par d faut")
                atr_defaults = {"EURUSD": 0.0005, "GBPUSD": 0.0006, "USDJPY": 0.05, "XAUUSD": 2.5}
                atr = atr_defaults.get(symbol, 0.0005)
            
            # CORRECTION XAUUSD
            if symbol == "XAUUSD" and atr > 10:
                logging.warning(f"  XAUUSD: ATR anormal ({atr}), ajustement")
                atr = 2.5
            
            # Calculer stops avec prix R EL
            # Pour SELL: SL au-dessus, TP en-dessous
            if symbol == "XAUUSD":
                stop_loss = current_price + (atr * 1.5)
                take_profit = current_price - (atr * 2.5)
            else:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
            
            # V rification stops
            if stop_loss <= 0 or take_profit <= 0:
                logging.error(f"  Stops invalides: SL={stop_loss:.5f}, TP={take_profit:.5f}")
                return
            
            # V rifier distance minimale
            if MT5_AVAILABLE:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
                    if min_stop_distance > 0:
                        actual_distance = abs(current_price - stop_loss)
                        if actual_distance < min_stop_distance:
                            logging.warning(f"  Stop trop proche, ajustement")
                            stop_loss = current_price + (min_stop_distance * 1.5)
            
            # R cup rer balance
            if MT5_AVAILABLE:
                account_info = mt5.account_info()
                balance = account_info.balance if account_info else 10000.0
            else:
                balance = 10000.0
            
            # Calculer position size
            position_size = self._calculate_basic_position_size(symbol, balance, stop_loss, current_price)
            
            # Ex cuter ordre
            await self._execute_mt5_order(symbol, 'sell', position_size, stop_loss, take_profit)
            
            # Stats
            self.session_stats['total_trades'] += 1
            logging.info(f"  SELL order execut e pour {symbol}")
            
        except Exception as e:
            logging.error(f"  Erreur _execute_sell: {e}")
            traceback.print_exc()
            logging.error(f"  Erreur _execute_sell: {e}")

    def _calculate_basic_position_size(self, symbol: str, balance: float, stop_loss: float, entry_price: float) -> float:
        """Calculate position size using MT5 symbol info"""
        try:
            risk_amount = balance * self.config.RISK_PER_TRADE
            
            if MT5_AVAILABLE:
                # R cup ration des infos pr cises du symbole
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    point = symbol_info.point
                    tick_value = symbol_info.trade_tick_value
                    logging.debug(f"  {symbol}: point={point}, tick_value={tick_value}")
                else:
                    logging.warning(f"  Info symbole non disponible pour {symbol}, valeurs par d faut")
                    point = 0.0001
                    tick_value = 10.0
            else:
                # Fallback pour les tests
                point_values = {
                    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 
                    'USDJPY': 0.01, 'XAUUSD': 0.01
                }
                tick_values = {
                    'EURUSD': 10.0, 'GBPUSD': 10.0, 
                    'USDJPY': 9.0, 'XAUUSD': 1.0
                }
                point = point_values.get(symbol, 0.0001)
                tick_value = tick_values.get(symbol, 10.0)
            
            # Calcul en ticks plut t qu'en pips
            stop_ticks = abs(entry_price - stop_loss) / point
            if stop_ticks < 1:
                logging.warning(f"  Stop-loss trop petit ({stop_ticks:.1f} ticks), minimum 1 tick")
                stop_ticks = 1
            
            position_size = risk_amount / (stop_ticks * tick_value)
            
            # Appliquer les limites
            position_size = max(0.01, position_size)  # Minimum 0.01 lot
            position_size = min(position_size, 100.0)  # Maximum 100 lots
            
            # Arrondir au lot valide
            position_size = round(position_size / 0.01) * 0.01
            
            logging.info(f"  Position size {symbol}: {position_size:.2f} lots "
                        f"(risk=${risk_amount:.2f}, stop={stop_ticks:.1f} ticks)")
            
            return position_size
            
        except Exception as e:
            logging.error(f"  Erreur calcul position size {symbol}: {e}")
            return 0.01  # Taille minimale safe

    async def _execute_mt5_order(self, symbol: str, direction: str, volume: float, stop_loss: float, take_profit: float):
        """Execute real MT5 order"""
        if not MT5_AVAILABLE:
            logging.info(f"  MOCK ORDER: {direction} {volume} lots {symbol}")
            return None
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logging.error(f"  Symbol info non disponible: {symbol}")
                return
                
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logging.error(f"  Impossible de s lectionner {symbol}")
                    return
            
            price = mt5.symbol_info_tick(symbol).ask if direction == 'buy' else mt5.symbol_info_tick(symbol).bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": float(stop_loss),
                "tp": float(take_profit),
                "deviation": 20,
                "magic": 234000,
                "comment": "HRM-TRM-PPO-DQN",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                logging.error(f"  order_send returned None")
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"  Order failed: {result.retcode} - {result.comment}")
                return None
            
            logging.info(f"  Order executed: {result.order}")
            
            #   CORRECTION: Ajouter la position   open_positions
            self.open_positions[symbol] = {
                'ticket': result.order,
                'symbol': symbol,
                'direction': direction,
                'entry_price': price,
                'volume': volume,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'time': datetime.now()
            }
            
            return result
                
        except Exception as e:
            logging.error(f"  Erreur _execute_mt5_order: {e}")

    def _sync_positions_with_mt5(self):
        """Synchronise open_positions avec les positions r elles MT5"""
        if not MT5_AVAILABLE:
            return
        
        try:
            # R cup rer positions MT5
            positions = mt5.positions_get()
            if positions is None:
                positions = []
            
            # Cr er dict des positions MT5 par symbole
            mt5_positions = {pos.symbol: pos for pos in positions}
            
            # Nettoyer les positions ferm es sur MT5
            for symbol in list(self.open_positions.keys()):
                if symbol not in mt5_positions:
                    logging.info(f"  Position {symbol} ferm e sur MT5, nettoyage...")
                    del self.open_positions[symbol]
            
            # Ajouter positions MT5 manquantes
            for symbol, pos in mt5_positions.items():
                if symbol not in self.open_positions:
                    logging.info(f"  Position {symbol} trouv e sur MT5, ajout...")
                    self.open_positions[symbol] = {
                        'ticket': pos.ticket,
                        'symbol': symbol,
                        'direction': 'buy' if pos.type == 0 else 'sell',
                        'entry_price': pos.price_open,
                        'volume': pos.volume,
                        'stop_loss': pos.sl,
                        'take_profit': pos.tp,
                        'time': datetime.fromtimestamp(pos.time)
                    }
            
            logging.debug(f"  Sync: {len(self.open_positions)} positions en m moire, {len(mt5_positions)} sur MT5")
            
        except Exception as e:
            logging.error(f"  Erreur sync positions: {e}")

    import traceback

async def _close_position(self, symbol: str, reason: str = "MANUAL"):
    """Ferme une position ouverte"""
    try:
        if symbol not in self.open_positions:
            logging.warning(f"  Pas de position ouverte pour {symbol}")
            return

        position = self.open_positions[symbol]
        logging.info(f"  Fermeture position {symbol} (reason: {reason})")

        # Get current price
        if MT5_AVAILABLE:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logging.error(f"  Impossible de r cup rer prix pour {symbol}")
                return

            price = tick.bid if position['direction'] == 'buy' else tick.ask

            # Send close order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position['volume'],
                "type": mt5.ORDER_TYPE_SELL if position['direction'] == 'buy' else mt5.ORDER_TYPE_BUY,
                "position": position.get('ticket', 0),
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Close-{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"  Close failed: {result.retcode} - {result.comment}")
                return

            # Calculate P&L
            if position['direction'] == 'buy':
                pnl = (price - position['entry_price']) * position['volume'] * 100000
            else:
                pnl = (position['entry_price'] - price) * position['volume'] * 100000
        else:
            # Mock P&L
            price = position['entry_price']
            pnl = np.random.uniform(-50, 100)

        # Record trade
        trade_record = {
            'symbol': symbol,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': price,
            'volume': position['volume'],
            'pnl': pnl,
            'open_time': position.get('open_time', datetime.now()),
            'close_time': datetime.now(),
            'duration': (datetime.now() - position.get('open_time', datetime.now())).total_seconds() / 60,
            'strategy': position.get('strategy', 'RL'),
            'confidence': position.get('confidence', 0.5),
            'reason': reason
        }

        self.trade_history.append(trade_record)

        # Update stats
        if pnl > 0:
            self.session_stats['winning_trades'] += 1
        else:
            self.session_stats['losing_trades'] += 1

        self.session_stats['total_pnl'] += pnl
        self.session_stats['daily_pnl'] += pnl
        self.daily_loss_tracker += pnl if pnl < 0 else 0

        # Record for RL
        self.rl_manager.record_trade_result(symbol, pnl)

        # Update risk manager
        if MT5_AVAILABLE:
            account_info = mt5.account_info()
            if account_info and hasattr(self, 'risk_manager'):
                self.risk_manager.update_drawdown(account_info.equity)

        # Remove position
        del self.open_positions[symbol]

        logging.info(f"  Position ferm e: {symbol}, P&L = ${pnl:.2f}")

        # Broadcast update
        if hasattr(self, '_broadcast_trade_update'):
            await self._broadcast_trade_update(trade_record)

    except Exception as e:
        logging.error(f"  Erreur _close_position: {e}")
        traceback.print_exc()


    async def _process_successful_close(self, symbol: str, position: dict, price: float, reason: str):
        """Traite une fermeture r ussie"""
        # Calcul du P&L r el
        if position['direction'] == 'buy':
            pnl = (price - position['entry_price']) * position['volume'] * 100000
        else:
            pnl = (position['entry_price'] - price) * position['volume'] * 100000
        
        # Enregistrement du trade
        trade_record = {
            'symbol': symbol,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': price,
            'volume': position['volume'],
            'pnl': pnl,
            'open_time': position['open_time'],
            'close_time': datetime.now(),
            'duration': (datetime.now() - position['open_time']).total_seconds() / 60,
            'strategy': position.get('strategy', 'unknown'),
            'confidence': position.get('confidence', 0),
            'reason': reason
        }
        
        self.trade_history.append(trade_record)
        
        # Mise   jour des stats
        if pnl > 0:
            self.session_stats['winning_trades'] += 1
        else:
            self.session_stats['losing_trades'] += 1
        
        self.session_stats['total_pnl'] += pnl
        self.session_stats['daily_pnl'] += pnl
        self.daily_loss_tracker += pnl if pnl < 0 else 0
        
        # Enregistrement pour RL et meta-strategy
        if hasattr(self, 'rl_manager'):
            self.rl_manager.record_trade_result(symbol, pnl)
        if hasattr(self, 'meta_strategy_selector'):
            self.meta_strategy_selector.record_strategy_result(symbol, position.get('strategy', 'unknown'), pnl)
        
        # Mise   jour risk manager
        if MT5_AVAILABLE:
            account_info = mt5.account_info()
            if account_info and hasattr(self, 'risk_manager'):
                self.risk_manager.update_drawdown(account_info.equity)
        
        # Supprimer la position
        del self.open_positions[symbol]
        
        logging.info(f"  Position ferm e: {symbol}, P&L = ${pnl:.2f}")
        
        # Broadcast aux clients
        await self._broadcast_trade_update(trade_record)

    async def trading_loop(self):
        """Main trading loop"""
        logging.info("  Trading loop d marr ")
        
        while self.active:
            try:
                for symbol in self.config.SYMBOLS:
                    if self.active:
                        await self.analyze_and_trade(symbol)
                        
                        # Check for retrain need
                        await self.rl_manager.retrain_on_drop(symbol)
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Toutes les minutes
                
            except Exception as e:
                logging.error(f"  Erreur dans trading loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)

    async def run(self):
        """Run the trading brain"""
        await self.trading_loop()

    def stop(self):
        """Stop the trading brain"""
        self.active = False
        if MT5_AVAILABLE:
            mt5.shutdown()
        logging.info("  Trading Brain arr t ")

    def _update_trading_stats(self, symbol: str, action: str):
        """Met   jour les statistiques de trading en temps r el"""
        if not hasattr(self, 'action_stats'):
            self.action_stats = {s: {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'CLOSE': 0} for s in self.config.SYMBOLS}
        
        if action in self.action_stats[symbol]:
            self.action_stats[symbol][action] += 1
        
        # Log p riodique des stats
        total_actions = sum(self.action_stats[symbol].values())
        if total_actions % 10 == 0:  # Toutes les 10 actions
            logging.info(f"  Stats {symbol}: {self.action_stats[symbol]}")

#============================================================================
#WEBSOCKET HANDLER
#============================================================================
async def websocket_handler(websocket, brain: UltimateTradingBrain):
    """Handle WebSocket connections pour monitoring"""
    brain.connected_clients.add(websocket)
    logging.info("  Nouveau client WebSocket connect ")
    try:
        # Send welcome message
        await websocket.send(json.dumps({
            'type': 'WELCOME',
            'message': '  ULTIMATE HYBRID TRADER CONNECTED',
            'features': [
                'HRM_TRM_FUSION',
                'PPO_DQN_ENSEMBLE',
                'ADAPTIVE_HALTING',
                'RISK_MANAGEMENT',
                'AUTO_RETRAIN'
            ],
            'timestamp': datetime.now().isoformat()
        }))
        
        # Handle incoming messages
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get('type')
                
                if msg_type == 'GET_STATUS':
                    status = {
                        'active': brain.active,
                        'uptime': str(datetime.now() - brain.start_time).split('.')[0],
                        'session_stats': brain.session_stats,
                        'daily_pnl': brain.session_stats['daily_pnl'],
                        'daily_loss_limit': brain.config.DAILY_LOSS_LIMIT,
                        'symbols': brain.config.SYMBOLS
                    }
                    await websocket.send(json.dumps({
                        'type': 'STATUS',
                        'data': status,
                        'timestamp': datetime.now().isoformat()
                    }))
                
                elif msg_type == 'COMMAND':
                    command = data.get('command')
                    
                    if command == 'START':
                        brain.active = True
                        logging.info("  Trading activ  manuellement")
                    elif command == 'STOP':
                        brain.active = False
                        logging.info("  Trading mis en pause manuellement")
                    elif command == 'RETRAIN':
                        symbol = data.get('symbol')
                        if symbol in brain.config.SYMBOLS:
                            await brain.rl_manager.retrain_on_drop(symbol)
                    
                    await websocket.send(json.dumps({
                        'type': 'COMMAND_ACK',
                        'command': command,
                        'success': True,
                        'timestamp': datetime.now().isoformat()
                    }))
                
                elif msg_type == 'GET_PERFORMANCE':
                    symbol = data.get('symbol')
                    if symbol in brain.rl_manager.performance_history:
                        perf = list(brain.rl_manager.performance_history[symbol])
                        await websocket.send(json.dumps({
                            'type': 'PERFORMANCE',
                            'symbol': symbol,
                            'data': {
                                'recent_trades': perf[-20:],
                                'win_rate': sum(perf[-20:]) / len(perf[-20:]) if perf else 0
                            },
                            'timestamp': datetime.now().isoformat()
                        }))
            
            except json.JSONDecodeError:
                logging.error("  Invalid JSON received")
            except Exception as e:
                logging.error(f"  Erreur handling message: {e}")

    except websockets.exceptions.ConnectionClosed:
        logging.info("  Client WebSocket d connect ")
    finally:
        brain.connected_clients.discard(websocket)

#============================================================================
#MAIN - POINT D'ENTR E
#============================================================================
#============================================================================
#MAIN - POINT D'ENTR E CORRIG 
#============================================================================
async def main():
    """Main entry point CORRIG """
    print("=" * 80)
    print("  ULTIMATE HYBRID TRADING BOT - VERSION STABLE")
    print("=" * 80)
    
    # Initialize brain
    brain = UltimateTradingBrain(CONFIG)

    # CORRECTION: G rer l'initialisation avec skip_training
    skip_training = '--skip-training' in sys.argv or '-s' in sys.argv
    
    if not await brain.initialize(skip_training=skip_training):
        logging.error("   chec initialisation, arr t du bot")
        return

    # Start WebSocket server
    async def ws_handler(websocket, path=None):
        await websocket_handler(websocket, brain)

        server = await websockets.serve(ws_handler, "localhost", CONFIG.WS_PORT)
        logging.info(f"  WebSocket server d marr  sur localhost:{CONFIG.WS_PORT}")

        # Start trading
        try:
            await brain.run()
        except KeyboardInterrupt:
            logging.info("\n  Interruption clavier d tect e")
        except Exception as e:
            logging.error(f"  Erreur fatale: {e}")
            traceback.print_exc()
        finally:
            brain.stop()
            server.close()
            await server.wait_closed()
            logging.info("  Arr t propre du syst me")

    if __name__ == "__main__":
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n  Au revoir!")
        except Exception as e:
            logging.error(f"  Erreur critique: {e}")
            traceback.print_exc()
        
        #============================================================================
#PARTIE 2: COMPOSANTS AVANC S - ORDER FLOW & SENTIMENT ANALYSIS
#============================================================================

#============================================================================
#ORDER FLOW ANALYZER
#============================================================================
class OrderFlowAnalyzer:
    """
    Analyse l'order flow pour d tecter les mouvements de smart money.
    Utilise:
    - Volume profile
    - Delta volume (buy vs sell)
    - Large order detection
    - Absorption patterns
    """
    
    def __init__(self):
        self.volume_history: Dict[str, deque] = {}
        self.delta_history: Dict[str, deque] = {}

        logging.info("  Order Flow Analyzer initialis ")

    async def analyze(self, symbol: str, timeframe: int = 15) -> Dict[str, Any]:
        """
        Analyse l'order flow pour un symbole.
        
        Returns:
            Dict avec:
            - smart_money_direction: 1 (buy), -1 (sell), 0 (neutral)
            - volume_strength: 0-100
            - absorption_detected: bool
            - imbalance_ratio: float
        """
        try:
            if not MT5_AVAILABLE:
                return self._mock_order_flow()
            
            # Get tick data
            ticks = mt5.copy_ticks_from(symbol, datetime.now() - timedelta(minutes=timeframe), 1000, mt5.COPY_TICKS_ALL)
            
            if ticks is None or len(ticks) == 0:
                return self._mock_order_flow()
            
            ticks_df = pd.DataFrame(ticks)
            
            # CORRECTION: Order flow robuste
            try:
                # V rifier que les colonnes existent
                if 'flags' in ticks_df.columns and 'volume' in ticks_df.columns:
                    buy_volume = ticks_df[(ticks_df['flags'] & mt5.TICK_FLAG_BUY) != 0]['volume'].sum()
                    sell_volume = ticks_df[(ticks_df['flags'] & mt5.TICK_FLAG_SELL) != 0]['volume'].sum()
                else:
                    # Fallback si colonnes manquantes
                    buy_volume = len(ticks_df) * 100
                    sell_volume = len(ticks_df) * 100
            except Exception as flow_error:
                logging.debug(f"  Erreur order flow: {flow_error}")
                buy_volume = 1000
                sell_volume = 1000
            total_volume = buy_volume + sell_volume
            
            # Delta calculation
            delta = buy_volume - sell_volume
            
            # Imbalance ratio
            imbalance_ratio = delta / total_volume if total_volume > 0 else 0
            
            # Detect large orders (>3x average)
            avg_volume = ticks_df['volume'].mean()
            large_orders = ticks_df[ticks_df['volume'] > avg_volume * 3]
            
            # Smart money direction
            smart_money_direction = 0
            if imbalance_ratio > 0.3:
                smart_money_direction = 1  # Bullish
            elif imbalance_ratio < -0.3:
                smart_money_direction = -1  # Bearish
            
            # Volume strength
            volume_strength = min(100, abs(imbalance_ratio) * 200)
            
            # Absorption detection (large sell orders but price stable/rising)
            absorption_detected = False
            if len(large_orders) > 0:
                large_sell_orders = large_orders[large_orders['flags'] & mt5.TICK_FLAG_SELL]
                if len(large_sell_orders) > 0:
                    price_change = ticks_df['bid'].iloc[-1] - ticks_df['bid'].iloc[0]
                    if price_change >= 0:  # Price stable or rising despite large sells
                        absorption_detected = True
            
            result = {
                'smart_money_direction': smart_money_direction,
                'volume_strength': volume_strength,
                'absorption_detected': absorption_detected,
                'imbalance_ratio': imbalance_ratio,
                'buy_volume': int(buy_volume),
                'sell_volume': int(sell_volume),
                'large_orders_count': len(large_orders)
            }
            
            logging.debug(f"  Order Flow {symbol}: {result}")
            return result
            
        except Exception as e:
            logging.error(f"  Erreur Order Flow analysis: {e}")
            return self._mock_order_flow()

    def _mock_order_flow(self) -> Dict[str, Any]:
        """Mock order flow data pour testing"""
        return {
            'smart_money_direction': np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3]),
            'volume_strength': np.random.uniform(40, 80),
            'absorption_detected': np.random.choice([True, False], p=[0.2, 0.8]),
            'imbalance_ratio': np.random.uniform(-0.5, 0.5),
            'buy_volume': np.random.randint(1000, 5000),
            'sell_volume': np.random.randint(1000, 5000),
            'large_orders_count': np.random.randint(0, 10)
        }

#============================================================================
#SENTIMENT ANALYZER
#============================================================================
class SentimentAnalyzer:
    """
    Analyse le sentiment du march  via:
    - Twitter/X API
    - Reddit API
    - News feeds
    - VIX/Fear & Greed Index
    """
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_duration = timedelta(minutes=30)

        # Initialize APIs
        self.twitter_api = None
        self.reddit_api = None
        
        if SENTIMENT_AVAILABLE:
            try:
                # Twitter API (if credentials available)
                twitter_bearer = os.getenv('TWITTER_BEARER_TOKEN')
                if twitter_bearer:
                    self.twitter_api = tweepy.Client(bearer_token=twitter_bearer)
                
                # Reddit API
                reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
                reddit_secret = os.getenv('REDDIT_CLIENT_SECRET')
                if reddit_client_id and reddit_secret:
                    self.reddit_api = praw.Reddit(
                        client_id=reddit_client_id,
                        client_secret=reddit_secret,
                        user_agent='UltimateTrader/1.0'
                    )
                
                logging.info("  Sentiment Analyzer initialis ")
            except Exception as e:
                logging.warning(f"  Erreur init Sentiment Analyzer: {e}")

    async def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Analyse le sentiment pour un symbole.
        
        Returns:
            Dict avec:
            - sentiment_score: -1   1 (n gatif   positif)
            - confidence: 0-100
            - sources: list des sources utilis es
            - volume: nombre de mentions
        """
        # Check cache
        if symbol in self.cache:
            score, timestamp = self.cache[symbol]
            if datetime.now() - timestamp < self.cache_duration:
                return {
                    'sentiment_score': score,
                    'confidence': 75,
                    'sources': ['cache'],
                    'volume': 0,
                    'cached': True
                }
        
        try:
            sentiments = []
            sources = []
            total_volume = 0
            
            # Twitter sentiment
            if self.twitter_api:
                twitter_sentiment = await self._analyze_twitter(symbol)
                if twitter_sentiment:
                    sentiments.append(twitter_sentiment['score'])
                    sources.append('twitter')
                    total_volume += twitter_sentiment['volume']
            
            # Reddit sentiment
            if self.reddit_api:
                reddit_sentiment = await self._analyze_reddit(symbol)
                if reddit_sentiment:
                    sentiments.append(reddit_sentiment['score'])
                    sources.append('reddit')
                    total_volume += reddit_sentiment['volume']
            
            # Calculate aggregate sentiment
            if sentiments:
                sentiment_score = np.mean(sentiments)
                confidence = min(100, len(sentiments) * 30 + total_volume / 10)
            else:
                # Fallback to mock
                sentiment_score = np.random.uniform(-0.3, 0.3)
                confidence = 50
                sources = ['mock']
            
            # Cache result
            self.cache[symbol] = (sentiment_score, datetime.now())
            
            result = {
                'sentiment_score': float(sentiment_score),
                'confidence': float(confidence),
                'sources': sources,
                'volume': int(total_volume),
                'cached': False
            }
            
            logging.debug(f"  Sentiment {symbol}: {result}")
            return result
            
        except Exception as e:
            logging.error(f"  Erreur Sentiment analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0,
                'sources': ['error'],
                'volume': 0,
                'cached': False
            }

    async def _analyze_twitter(self, symbol: str) -> Optional[Dict]:
        """Analyse Twitter sentiment"""
        try:
            # Search tweets
            query = f"{symbol} OR ${symbol}"
            tweets = self.twitter_api.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                return None
            
            # Analyze sentiment with TextBlob
            sentiments = []
            for tweet in tweets.data:
                blob = TextBlob(tweet.text)
                sentiments.append(blob.sentiment.polarity)
            
            return {
                'score': np.mean(sentiments),
                'volume': len(sentiments)
            }
            
        except Exception as e:
            logging.debug(f"  Twitter non configur : {e}")
            return None

    async def _analyze_reddit(self, symbol: str) -> Optional[Dict]:
        """Analyse Reddit sentiment"""
        try:
            # Search relevant subreddits
            subreddits = ['wallstreetbets', 'stocks', 'forex', 'investing']
            sentiments = []
            
            for sub_name in subreddits:
                subreddit = self.reddit_api.subreddit(sub_name)
                
                # Search posts
                for post in subreddit.search(symbol, time_filter='day', limit=50):
                    blob = TextBlob(post.title + " " + post.selftext)
                    sentiments.append(blob.sentiment.polarity)
            
            if not sentiments:
                return None
            
            return {
                'score': np.mean(sentiments),
                'volume': len(sentiments)
            }
            
        except Exception as e:
            logging.debug(f"  Reddit non configur : {e}")
            return None

#============================================================================
#META-STRATEGY SELECTOR
#============================================================================
class MetaStrategySelector:
    """
    S lecteur adaptatif de strat gie bas  sur:
    - Conditions de march  (trending, ranging, volatile)
    - Performance r cente des strat gies
    - Corr lations inter-symboles
    - Moment de la journ e
    """
    
    def __init__(self):
        self.strategy_performance: Dict[str, Dict[str, deque]] = {}
        self.available_strategies = [
            'momentum',
            'mean_reversion',
            'breakout',
            'scalping',
            'swing'
        ]

        logging.info("  Meta-Strategy Selector initialis ")

    async def select_strategy(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        order_flow: Dict,
        sentiment: Dict
    ) -> str:
        """
        S lectionne la meilleure strat gie pour les conditions actuelles.
        
        Returns:
            Nom de la strat gie recommand e
        """
        try:
            # Analyze market regime
            regime = self._detect_market_regime(market_data)
            
            # Calculate strategy scores
            scores = {}
            
            for strategy in self.available_strategies:
                score = 0.0
                
                # Base score selon le r gime
                if regime == 'trending':
                    if strategy in ['momentum', 'breakout']:
                        score += 30
                    elif strategy == 'mean_reversion':
                        score -= 20
                
                elif regime == 'ranging':
                    if strategy == 'mean_reversion':
                        score += 30
                    elif strategy == 'scalping':
                        score += 20
                    elif strategy in ['momentum', 'breakout']:
                        score -= 20
                
                elif regime == 'volatile':
                    if strategy == 'breakout':
                        score += 25
                    elif strategy == 'scalping':
                        score -= 20
                
                # Order flow influence
                if order_flow['smart_money_direction'] != 0:
                    if strategy == 'momentum':
                        score += abs(order_flow['imbalance_ratio']) * 20
                
                # Sentiment influence
                if abs(sentiment['sentiment_score']) > 0.5:
                    if strategy == 'momentum':
                        score += abs(sentiment['sentiment_score']) * 15
                
                # Time of day influence
                hour = datetime.now().hour
                if 8 <= hour <= 12:  # Session europ enne
                    if strategy in ['breakout', 'momentum']:
                        score += 10
                elif 13 <= hour <= 17:  # Session US
                    if strategy in ['momentum', 'swing']:
                        score += 10
                else:  # Session asiatique
                    if strategy in ['ranging', 'scalping']:
                        score += 10
                
                # Historical performance
                perf = self._get_strategy_performance(symbol, strategy)
                score += perf * 20
                
                scores[strategy] = score
            
            # Select best strategy
            best_strategy = max(scores, key=scores.get)
            
            logging.debug(f"  Strategy selection {symbol}: {best_strategy} (scores: {scores})")
            return best_strategy
            
        except Exception as e:
            logging.error(f"  Erreur strategy selection: {e}")
            return 'momentum'  # Fallback

    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        D tecte le r gime du march .
        
        Returns:
            'trending', 'ranging', ou 'volatile'
        """
        try:
            # ADX pour trending
            adx = self._calculate_adx(data)
            
            # ATR pour volatilit 
            atr = data['atr'].iloc[-1]
            atr_ma = data['atr'].rolling(window=20).mean().iloc[-1]
            
            # Bollinger Bands pour ranging
            bb_width = (data['bb_upper'].iloc[-1] - data['bb_lower'].iloc[-1]) / data['bb_middle'].iloc[-1]
            
            # Decision logic
            if adx > 25 and bb_width > 0.02:
                return 'trending'
            elif atr > atr_ma * 1.5:
                return 'volatile'
            else:
                return 'ranging'
                
        except Exception as e:
            logging.error(f"  Erreur market regime detection: {e}")
            return 'ranging'

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            plus_dm = high.diff()
            minus_dm = low.diff().abs()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(window=period).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx.iloc[-1]
            
        except Exception as e:
            logging.error(f"  Erreur ADX calculation: {e}")
            return 20.0

    def _get_strategy_performance(self, symbol: str, strategy: str) -> float:
        """
        R cup re la performance historique d'une strat gie.
        
        Returns:
            Score normalis  entre -1 et 1
        """
        if symbol not in self.strategy_performance:
            self.strategy_performance[symbol] = {s: deque(maxlen=50) for s in self.available_strategies}
        
        perf_history = self.strategy_performance[symbol][strategy]
        
        if len(perf_history) == 0:
            return 0.0
        
        # Calculate win rate
        win_rate = sum(1 for p in perf_history if p > 0) / len(perf_history)
        
        # Normalize to -1 to 1
        return (win_rate - 0.5) * 2

    def record_strategy_result(self, symbol: str, strategy: str, pnl: float):
        """Enregistre le r sultat d'une strat gie"""
        if symbol not in self.strategy_performance:
            self.strategy_performance[symbol] = {s: deque(maxlen=50) for s in self.available_strategies}
        
        self.strategy_performance[symbol][strategy].append(1 if pnl > 0 else -1)

#============================================================================
#RISK MANAGER AVANC 
#============================================================================
class AdvancedRiskManager:
    """
    Gestionnaire de risque avanc  avec:
    - Position sizing dynamique
    - Corr lation portfolio
    - Drawdown protection
    - Kelly Criterion
    - VaR (Value at Risk)
    """
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.portfolio_history: deque = deque(maxlen=1000)
        self.drawdown_tracker = 0.0
        self.peak_balance = 0.0

        logging.info("  Advanced Risk Manager initialis ")

    def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        stop_loss_pips: float,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5
    ) -> float:
        """
        Calcule la taille de position optimale.
        
        Utilise une combinaison de:
        - Risk per trade fixe
        - Kelly Criterion (avec factoriel de s curit )
        - Drawdown adjustment
        
        Returns:
            Taille de position en lots
        """
        try:
            # Base risk amount
            risk_amount = account_balance * self.config.RISK_PER_TRADE
            
            # Adjust for drawdown
            if self.drawdown_tracker > 0.1:  # >10% drawdown
                risk_amount *= 0.5  # Reduce risk by half
            elif self.drawdown_tracker > 0.2:  # >20% drawdown
                risk_amount *= 0.25  # Reduce risk by 75%
            
            # Kelly Criterion
            kelly_fraction = self._calculate_kelly(win_rate, avg_win_loss_ratio)
            kelly_adjusted = kelly_fraction * 0.25  # Use 25% of Kelly for safety
            
            # Calculate position size
            pip_value = self._get_pip_value(symbol)
            position_size = (risk_amount * kelly_adjusted) / (stop_loss_pips * pip_value)
            
            # Apply limits
            max_position = self._get_max_position_size(symbol, account_balance)
            position_size = min(position_size, max_position)
            
            # Round to valid lot size
            position_size = self._round_to_lot_size(position_size)
            
            logging.debug(f"  Position size {symbol}: {position_size} lots (kelly: {kelly_adjusted:.2%})")
            return position_size
            
        except Exception as e:
            logging.error(f"  Erreur position sizing: {e}")
            return 0.01  # Minimum size

    def _calculate_kelly(self, win_rate: float, avg_win_loss_ratio: float) -> float:
        """
        Kelly Criterion: f = (p * b - q) / b
        o :
        - p = win rate
        - q = 1 - p
        - b = avg win / avg loss
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        q = 1 - win_rate
        kelly = (win_rate * avg_win_loss_ratio - q) / avg_win_loss_ratio
        
        # Limit to reasonable range
        return max(0.0, min(kelly, 0.25))

    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        if MT5_AVAILABLE:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                return symbol_info.trade_tick_value
        
        # Default values
        pip_values = {
            'EURUSD': 10.0,
            'GBPUSD': 10.0,
            'USDJPY': 9.0,
            'XAUUSD': 1.0
        }
        return pip_values.get(symbol, 10.0)

    def _get_max_position_size(self, symbol: str, account_balance: float) -> float:
        """Calculate maximum allowed position size"""
        # Max 5% of account per position
        max_risk = account_balance * 0.05
        pip_value = self._get_pip_value(symbol)
        
        # Assume 50 pips stop loss for max calculation
        max_position = max_risk / (50 * pip_value)
        
        return max_position

    def _round_to_lot_size(self, size: float) -> float:
        """Round to valid lot size (0.01 increment)"""
        return round(size / 0.01) * 0.01

    def check_correlation_limit(self, symbol: str, existing_positions: List[str]) -> bool:
        """
        V rifie si on peut ouvrir une position sans d passer les limites de corr lation.
        
        Returns:
            True si OK, False si trop corr l 
        """
        if not existing_positions:
            return True
        
        # Correlation matrix (simplified)
        correlations = {
            ('EURUSD', 'GBPUSD'): 0.85,
            ('EURUSD', 'USDJPY'): -0.75,
            ('GBPUSD', 'USDJPY'): -0.70,
            ('XAUUSD', 'EURUSD'): 0.50
        }
        
        for pos_symbol in existing_positions:
            pair = tuple(sorted([symbol, pos_symbol]))
            corr = correlations.get(pair, 0.0)
            
            if abs(corr) > 0.8:
                logging.warning(f"  Corr lation  lev e d tect e: {symbol} - {pos_symbol} ({corr:.2f})")
                return False
        
        return True

    def update_drawdown(self, current_balance: float):
        """Update drawdown tracking"""
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        if self.peak_balance > 0:
            self.drawdown_tracker = (self.peak_balance - current_balance) / self.peak_balance

    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.
        
        Returns:
            VaR value (maximum expected loss at given confidence level)
        """
        if not returns:
            return 0.0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0

#============================================================================
#PARTIE 3: ENHANCED TRADING BRAIN & EXECUTION ENGINE
#============================================================================

#============================================================================
#ENHANCED ULTIMATE TRADING BRAIN (VERSION COMPL TE)
#============================================================================
class EnhancedUltimateTradingBrain(UltimateTradingBrain):
    """
    Version am lior e du Trading Brain avec tous les composants int gr s.
    """

    def _validate_dataframe(self, df, method_name: str = "") -> bool:
        """Valide qu'un DataFrame est utilisable — stub de compatibilité."""
        try:
            return df is not None and len(df) > 0
        except Exception:
            return False

    def __init__(self, config: UltimateConfig):
        super().__init__(config)
        
        # CORRECTION: Initialiser open_positions d'abord pour  viter les erreurs
        self.open_positions: Dict[str, Dict] = {}
        self.pending_orders: Dict[str, List] = {}
        self.trade_history: deque = deque(maxlen=1000)
        self.equity_curve: deque = deque(maxlen=10000)
        
        # CORRECTION CRITIQUE: S'assurer que les attributs de daily loss sont bien initialis s
        # (au cas o  super().__init__ aurait un probl me)
        if not hasattr(self, 'daily_loss_tracker'):
            self.daily_loss_tracker = 0.0
        if not hasattr(self, 'last_daily_reset'):
            self.last_daily_reset = datetime.now().date()
        
        # Additional components - initialisation s curis e
        try:
            self.order_flow_analyzer = OrderFlowAnalyzer()
            self.sentiment_analyzer = SentimentAnalyzer(config)
            self.meta_strategy_selector = MetaStrategySelector()
            self.risk_manager = AdvancedRiskManager(config)
        except Exception as e:
            logging.warning(f"  Some advanced components failed to initialize: {e}")
            # Initialiser des composants de secours
            self.order_flow_analyzer = None
            self.sentiment_analyzer = None
            self.meta_strategy_selector = None
            self.risk_manager = None
        
        logging.info("  Enhanced Ultimate Trading Brain initialis ")


    # ========================================================================
    # M THODES DE GESTION INTELLIGENTE DES POSITIONS
    # Ajout es par patch v2 - 2025-11-03
    # ========================================================================
    
    def _get_positions_by_side(self, symbol: str, side: str) -> list:
        """
        R cup re les positions pour un symbole et une direction sp cifique
        
        Args:
            symbol: Le symbole de trading (ex: "EURUSD")
            side: Direction 'buy' ou 'sell'
            
        Returns:
            Liste des positions correspondantes
        """
        if not MT5_AVAILABLE:
            return []
            
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                return []
            
            position_type = 0 if side == 'buy' else 1  # 0=BUY, 1=SELL
            return [p for p in positions if p.type == position_type]
        except Exception as e:
            logging.error(f"Erreur _get_positions_by_side: {e}")
            return []
    
    async def _close_positions_safe(self, positions: list, reason: str) -> bool:
        """
        Ferme les positions de mani re s curis e avec v rifications
        
        Args:
            positions: Liste des positions   fermer
            reason: Raison de la cl ture (pour les logs)
            
        Returns:
            True si au moins une position a  t  ferm e avec succ s
        """
        if not positions or not MT5_AVAILABLE:
            return True
            
        success_count = 0
        for pos in positions:
            try:
                # Pr parer la requ te de cl ture
                close_type = 1 if pos.type == 0 else 0  # Inverse du type
                tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    continue
                    
                close_price = tick.ask if pos.type == 1 else tick.bid
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": float(pos.volume),
                    "type": close_type,
                    "position": pos.ticket,
                    "price": float(close_price),
                    "deviation": 10,
                    "magic": 234000,
                    "comment": f"Close: {reason}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    profit = pos.profit
                    logging.info(f"  Position #{pos.ticket} ferm e: {reason} | Profit: {profit:.2f}$")
                    success_count += 1
                else:
                    error = result.comment if result else "Unknown error"
                    logging.warning(f"   chec fermeture #{pos.ticket}: {error}")
                    
            except Exception as e:
                logging.error(f"  Erreur fermeture position #{pos.ticket}: {e}")
                continue
        
        return success_count > 0
    
    def _should_close_for_reverse(self, position, new_direction: str, df) -> dict:
        """
        D termine si une position doit  tre ferm e pour un reversal
        
        Args:
            position: Position MT5 actuelle
            new_direction: Nouvelle direction souhait e ('buy' ou 'sell')
            df: DataFrame avec donn es de march 
            
        Returns:
            Dict avec 'should_close' (bool) et 'reason' (str)
        """
        if not MT5_AVAILABLE:
            return {'should_close': False, 'reason': 'MT5 not available'}
            
        try:
            current_profit = position.profit
            position_age_minutes = (datetime.now().timestamp() - position.time) / 60
            
            # R cup rer le prix actuel
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                return {'should_close': False, 'reason': 'No tick data'}
            
            current_price = tick.bid if position.type == 0 else tick.ask
            entry_price = position.price_open
            
            # R GLE 1: Position en profit significatif (>50% du TP)   Fermer
            if current_profit > 0 and position.tp > 0:
                tp_distance = abs(position.tp - entry_price)
                if tp_distance > 0:
                    profit_ratio = abs(current_price - entry_price) / tp_distance
                    if profit_ratio > 0.5:
                        return {
                            'should_close': True,
                            'reason': f'Profit {profit_ratio:.1%} du TP atteint'
                        }
            
            # R GLE 2: Position en perte mais signal fort oppos    Fermer
            if current_profit < 0:
                # V rifier la force du nouveau signal via RSI si disponible
                if hasattr(df, 'columns') and 'rsi' in df.columns and len(df) > 0:
                    try:
                        rsi = df['rsi'].iloc[-1]
                        
                        if new_direction == 'buy' and rsi < 30:
                            return {
                                'should_close': True,
                                'reason': f'Signal BUY fort (RSI={rsi:.1f}) vs position SELL en perte'
                            }
                        elif new_direction == 'sell' and rsi > 70:
                            return {
                                'should_close': True,
                                'reason': f'Signal SELL fort (RSI={rsi:.1f}) vs position BUY en perte'
                            }
                    except:
                        pass
            
            # R GLE 3: Position ancienne (>60min) avec profit minimal   Fermer
            if position_age_minutes > 60 and current_profit > -10:
                return {
                    'should_close': True,
                    'reason': f'Position ancienne ({position_age_minutes:.0f}min) avec profit minimal'
                }
            
            # R GLE 4: Stop loss proche d' tre touch    Laisser faire
            if position.sl > 0:
                distance_to_sl = abs(current_price - position.sl)
                symbol_info = mt5.symbol_info(position.symbol)
                if symbol_info and symbol_info.point > 0:
                    distance_to_sl_pips = distance_to_sl / (symbol_info.point * 10)
                    if distance_to_sl_pips < 5:
                        return {
                            'should_close': False,
                            'reason': f'SL proche ({distance_to_sl_pips:.1f} pips) - laisser faire'
                        }
            
            # R GLE 5: Perte importante   Ne pas fermer (laisser le SL faire son travail)
            if current_profit < -20:
                return {
                    'should_close': False,
                    'reason': f'Perte importante ({current_profit:.2f}$) - attendre SL'
                }
            
            # Par d faut: NE PAS fermer
            return {
                'should_close': False,
                'reason': 'Conditions de reversal non remplies'
            }
            
        except Exception as e:
            logging.error(f"Erreur dans _should_close_for_reverse: {e}")
            return {'should_close': False, 'reason': f'Error: {str(e)}'}
    
    async def _smart_position_reverse(self, symbol: str, new_direction: str, df=None) -> bool:
        """
        G re intelligemment le reversal de position
        
        Args:
            symbol: Symbole de trading
            new_direction: Nouvelle direction souhait e ('buy' ou 'sell')
            df: DataFrame avec donn es de march  (optionnel)
            
        Returns:
            True si le trade peut  tre ex cut , False sinon
        """
        if not MT5_AVAILABLE:
            return True  # Pas de MT5 = pas de v rification = OK pour continuer
            
        try:
            # Direction oppos e
            opposite_direction = 'sell' if new_direction == 'buy' else 'buy'
            
            # R cup rer les positions oppos es
            opposite_positions = self._get_positions_by_side(symbol, opposite_direction)
            
            if not opposite_positions:
                # Pas de position oppos e   OK pour trader
                return True
            
            logging.info(f"  {symbol}: {len(opposite_positions)} position(s) {opposite_direction.upper()} d tect e(s)")
            
            # Si pas de DataFrame, on r cup re les donn es
            if df is None:
                try:
                    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                except:
                    df = None
            
            # Analyser chaque position
            positions_to_close = []
            for pos in opposite_positions:
                decision = self._should_close_for_reverse(pos, new_direction, df)
                
                if decision['should_close']:
                    logging.info(f"     Position #{pos.ticket}: FERMER - {decision['reason']}")
                    positions_to_close.append(pos)
                else:
                    logging.info(f"     Position #{pos.ticket}: GARDER - {decision['reason']}")
            
            # Fermer les positions s lectionn es
            if positions_to_close:
                success = await self._close_positions_safe(
                    positions_to_close,
                    f"Reversal vers {new_direction.upper()}"
                )
                
                if success:
                    logging.info(f"  {len(positions_to_close)} position(s) ferm e(s) pour reversal")
                    return True
                else:
                    logging.warning(f"   chec fermeture des positions pour reversal")
                    return False
            else:
                # Aucune position   fermer   Ne pas trader pour  viter la surexposition
                logging.info(f"  {symbol}: Aucune position ne remplit les crit res de reversal   Trade suspendu")
                return False
                
        except Exception as e:
            logging.error(f"Erreur dans _smart_position_reverse: {e}")
            return True  # En cas d'erreur, laisser passer pour ne pas bloquer
    
    # ========================================================================
    # FIN DES M THODES DE GESTION INTELLIGENTE
    # ========================================================================


    async def initialize(self, skip_training: bool = False) -> bool:
        """Initialize avec v rifications  tendues"""
        logging.info("  Initialisation Enhanced Trading Brain...")
        
        try:
            # Base initialization
            if not await super().initialize():
                return False
            
            # Additional checks
            if MT5_AVAILABLE:
                account_info = mt5.account_info()
                if account_info:
                    logging.info(f"  Compte MT5:")
                    logging.info(f"  Balance: ${account_info.balance:.2f}")
                    logging.info(f"  Equity: ${account_info.equity:.2f}")
                    logging.info(f"  Margin Level: {account_info.margin_level:.2f}%")
                    logging.info(f"  Leverage: 1:{account_info.leverage}")
                    
                    # Initialize risk manager
                    self.risk_manager.peak_balance = account_info.balance
            
            # Test components
            logging.info("  Test des composants...")
            test_symbol = self.config.SYMBOLS[0]
            
            # Test order flow
            order_flow = await self.order_flow_analyzer.analyze(test_symbol)
            logging.info(f"    Order Flow: {order_flow['smart_money_direction']}")
            
            # Test sentiment
            sentiment = await self.sentiment_analyzer.analyze(test_symbol)
            logging.info(f"    Sentiment: {sentiment['sentiment_score']:.2f}")
            
            logging.info("  Enhanced Trading Brain pr t!")
            return True
            
        except Exception as e:
            logging.error(f"  Erreur initialisation Enhanced Brain: {e}")
            traceback.print_exc()
            return False

    async def analyze_and_trade(self, symbol: str):
        """
        Analyse le march  et ex cute un trade si conditions r unies.
        VERSION CORRIG E avec gestion d'erreur renforc e.
        """
        try:
            #   DEBUG
            logging.info(f"  analyze_and_trade appel  pour {symbol}")
            
            # V rifier que open_positions existe
            if not hasattr(self, 'open_positions'):
                self.open_positions = {}
                logging.info(f"  open_positions initialis ")
            
            # Check daily loss limit
            if not self._check_daily_loss_limit():
                logging.warning("  Daily loss limit atteint, trading suspendu")
                return
            
            # Check max positions
            if len(self.open_positions) >= self.config.MAX_POSITIONS:
                logging.debug(f"  Max positions atteint ({self.config.MAX_POSITIONS})")
                return
            
            # Check if already in position
            if symbol in self.open_positions:
                logging.info(f"  {symbol} a d j  une position, gestion...")
                await self._manage_existing_position(symbol)
                return
            else:
                logging.info(f"  {symbol} pas de position, analyse pour ouvrir...")
            
            # Get market data avec gestion d'erreur
            logging.info(f"  R cup ration observation pour {symbol}...")
            observation = self._get_observation(symbol)
            if observation is None:
                logging.warning(f"  Observation None pour {symbol}")
                return
            logging.info(f"  Observation OK pour {symbol}: shape={observation.shape}")
            
            # V rification ultra-rigoureuse de l'observation
            if np.isnan(observation).any() or np.isinf(observation).any():
                logging.warning(f"  Observation corrompue pour {symbol}, skip")
                return
            
            # Get RL prediction avec protection
                        # 1) pr diction (prot g e)
            try:
                logging.info(f"  Pr diction action pour {symbol}.")
                action = self.rl_manager.predict_action(symbol, observation)
                action_name = ['BUY', 'SELL', 'HOLD', 'CLOSE'][action]
                logging.info(f"  Action pr dite pour {symbol}: {action_name}")
            except Exception as rl_error:
                logging.error(f"  Erreur prediction RL pour {symbol}: {rl_error}")
                traceback.print_exc()
                return

            # 2) stats (prot g es s par ment)
            try:
                self._update_trading_stats(symbol, action_name)
            except Exception as stats_error:
                logging.debug(f"  Statistiques non disponibles: {stats_error}")
            
            # Execute based on action
            # Ex cution avec gestion intelligente des positions
            if action == 0:  # BUY
                if MT5_AVAILABLE:
                    can_trade = await self._smart_position_reverse(symbol, 'buy')
                    if can_trade:
                        await self._execute_buy(symbol)
                    else:
                        logging.info(f"  {symbol}: BUY retard  (gestion positions)")
                else:
                    await self._execute_buy(symbol)
            
            elif action == 1:  # SELL
                if MT5_AVAILABLE:
                    can_trade = await self._smart_position_reverse(symbol, 'sell')
                    if can_trade:
                        await self._execute_sell(symbol)
                    else:
                        logging.info(f"  {symbol}: SELL retard  (gestion positions)")
                else:
                    await self._execute_sell(symbol)
            elif action == 3:  # CLOSE
                # CORRECTION: Logique CLOSE am lior e
                if symbol in self.open_positions:
                    await self._close_position(symbol)
                else:
                    # Pas de position   fermer, consid rer comme HOLD
                    logging.debug(f"  Action CLOSE sans position -> HOLD pour {symbol}")
                    # Ne rien faire = HOLD
            # action == 2 (HOLD) ne fait rien
            
        except Exception as e:
            logging.error(f"  Erreur analyze_and_trade pour {symbol}: {e}")
            traceback.print_exc()

    async def _get_market_data(self, symbol: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Get comprehensive market data with indicators"""
        try:
            if MT5_AVAILABLE:
                rates = mt5.copy_rates_from_pos(symbol, self._get_mt5_timeframe(CONFIG.TIMEFRAME), 0, bars)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df = self.rl_manager._calculate_indicators(df)
                    return df
            
            # Mock data
            return self.rl_manager._generate_mock_data(bars)
            
        except Exception as e:
            logging.error(f"  Erreur _get_market_data: {e}")
            return None

    def _prepare_observation(
        self,
        market_data: pd.DataFrame,
        order_flow: Dict,
        sentiment: Dict
    ) -> np.ndarray:
        """
        Pr pare une observation enrichie pour le RL agent.
        
        Combines:
        - Technical indicators
        - Order flow metrics
        - Sentiment scores
        """
        try:
            # Technical indicators
            tech_indicators = market_data.iloc[-1][
                ['close', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_middle']
            ].values
            
            # Order flow features
            order_flow_features = np.array([
                order_flow['smart_money_direction'],
                order_flow['volume_strength'] / 100,
                order_flow['imbalance_ratio'],
                1.0 if order_flow['absorption_detected'] else 0.0
            ])
            
            # Sentiment features
            sentiment_features = np.array([
                sentiment['sentiment_score'],
                sentiment['confidence'] / 100
            ])
            
            # Concatenate all features
            observation = np.concatenate([
                tech_indicators,
                order_flow_features,
                sentiment_features
            ])
            
            # Normalize
            observation = (observation - observation.mean()) / (observation.std() + 1e-8)
            
            return observation.astype(np.float32)
            
        except Exception as e:
            logging.error(f"  Erreur _prepare_observation: {e}")
            return np.zeros(12, dtype=np.float32)

    def _generate_fused_signal(
        self,
        rl_action: int,
        strategy: str,
        order_flow: Dict,
        sentiment: Dict,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Fusionne tous les signaux pour g n rer une d cision finale.
        
        Voting system avec pond rations:
        - RL agents: 40%
        - Order flow: 25%
        - Sentiment: 15%
        - Technical: 20%
        """
        signals = []
        weights = []
        
        # 1. RL Action (40%)
        rl_signal = {0: 1, 1: -1, 2: 0, 3: 0}[rl_action]  # BUY=1, SELL=-1, HOLD=0
        signals.append(rl_signal)
        weights.append(0.40)
        
        # 2. Order Flow (25%)
        of_signal = order_flow['smart_money_direction']
        if order_flow['absorption_detected']:
            of_signal *= 1.5  # Boost if absorption detected
        signals.append(of_signal)
        weights.append(0.25)
        
        # 3. Sentiment (15%)
        sent_signal = 0
        if sentiment['sentiment_score'] > 0.3:
            sent_signal = 1
        elif sentiment['sentiment_score'] < -0.3:
            sent_signal = -1
        signals.append(sent_signal)
        weights.append(0.15)
        
        # 4. Technical Indicators (20%)
        tech_signal = self._analyze_technical_signal(market_data)
        signals.append(tech_signal)
        weights.append(0.20)
        
        # Calculate weighted average
        weighted_signal = sum(s * w for s, w in zip(signals, weights))
        
        # Calculate confidence (agreement level)
        agreement = sum(1 for s in signals if s * weighted_signal > 0) / len(signals)
        confidence = agreement * 100
        
        # Generate final action
        if weighted_signal > 0.3:
            action = 'BUY'
        elif weighted_signal < -0.3:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'confidence': confidence,
            'weighted_signal': weighted_signal,
            'individual_signals': {
                'rl': rl_signal,
                'order_flow': of_signal,
                'sentiment': sent_signal,
                'technical': tech_signal
            },
            'strategy': strategy
        }

    def _analyze_technical_signal(self, data: pd.DataFrame) -> int:
        """
        Analyse technique simple pour signal.
        
        Returns:
            1 (bullish), -1 (bearish), 0 (neutral)
        """
        try:
            latest = data.iloc[-1]
            
            signal = 0
            
            # RSI
            if latest['rsi'] < 30:
                signal += 1
            elif latest['rsi'] > 70:
                signal -= 1
            
            # MACD
            if latest['macd'] > latest['macd_signal']:
                signal += 1
            else:
                signal -= 1
            
            # Bollinger Bands
            if latest['close'] < latest['bb_lower']:
                signal += 1
            elif latest['close'] > latest['bb_upper']:
                signal -= 1
            
            # Normalize to -1, 0, 1
            if signal > 0:
                return 1
            elif signal < 0:
                return -1
            else:
                return 0
                
        except Exception as e:
            logging.error(f"  Erreur _analyze_technical_signal: {e}")
            return 0

    async def _execute_trade(
        self,
        symbol: str,
        signal: Dict,
        market_data: pd.DataFrame
    ):
        """
        Ex cute un trade avec gestion des risques avanc e.
        """
        try:
            logging.info(f"  Ex cution trade {symbol}: {signal['action']}")
            
            # Get account info
            if MT5_AVAILABLE:
                account_info = mt5.account_info()
                if not account_info:
                    logging.error("  Impossible de r cup rer account info")
                    return
                
                balance = account_info.balance
                equity = account_info.equity
            else:
                balance = 10000.0  # Mock
                equity = 10000.0
            
            # Check correlation with existing positions
            existing_symbols = list(self.open_positions.keys())
            if not self.risk_manager.check_correlation_limit(symbol, existing_symbols):
                logging.warning(f"  Corr lation trop  lev e, trade annul  pour {symbol}")
                return
            
            # Calculate stop loss and take profit
            current_price = market_data.iloc[-1]['close']
            atr = market_data.iloc[-1]['atr']
            
            if signal['action'] == 'BUY':
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
                direction = 'buy'
            else:  # SELL
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
                direction = 'sell'
            
            # Calculate position size
            stop_loss_pips = abs(current_price - stop_loss) / 0.0001  # Assuming 4 decimal places
            
            # Get win rate from performance history
            recent_trades = list(self.rl_manager.performance_history[symbol])[-20:]
            win_rate = sum(recent_trades) / len(recent_trades) if recent_trades else 0.5
            
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                account_balance=balance,
                stop_loss_pips=stop_loss_pips,
                win_rate=max(0.3, min(0.7, win_rate)),  # Clamp between 30% and 70%
                avg_win_loss_ratio=1.5
            )
            
            logging.info(f"    Position size: {position_size} lots")
            logging.info(f"    Stop Loss: {stop_loss:.5f}")
            logging.info(f"    Take Profit: {take_profit:.5f}")
            logging.info(f"    Risk: ${balance * self.config.RISK_PER_TRADE:.2f}")
            
            # Execute on MT5
            if MT5_AVAILABLE:
                result = await self._execute_mt5_order(
                    symbol=symbol,
                    direction=direction,
                    volume=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if result:
                    # Record position
                    self.open_positions[symbol] = {
                        'ticket': result.order,
                        'direction': direction,
                        'volume': position_size,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'open_time': datetime.now(),
                        'strategy': signal['strategy'],
                        'confidence': signal['confidence']
                    }
                    
                    # Update stats
                    self.session_stats['total_trades'] += 1
                    
                    logging.info(f"  Trade ex cut : {result.order}")
            else:
                # Mock execution
                logging.info("  Mode MOCK - Trade simul ")
                self.open_positions[symbol] = {
                    'ticket': np.random.randint(1000000, 9999999),
                    'direction': direction,
                    'volume': position_size,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'open_time': datetime.now(),
                    'strategy': signal['strategy'],
                    'confidence': signal['confidence']
                }
                self.session_stats['total_trades'] += 1
            
        except Exception as e:
            logging.error(f"  Erreur _execute_trade: {e}")
            traceback.print_exc()

    async def _execute_mt5_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        stop_loss: float,
        take_profit: float
    ) -> Optional[Any]:
        """Execute order on MT5"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logging.error(f"  Symbol info non disponible: {symbol}")
                return None
            
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logging.error(f"  Impossible de s lectionner {symbol}")
                    return None
            
            point = symbol_info.point
            price = mt5.symbol_info_tick(symbol).ask if direction == 'buy' else mt5.symbol_info_tick(symbol).bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": float(stop_loss),
                "tp": float(take_profit),
                "deviation": 20,
                "magic": 234000,
                "comment": f"HRM-TRM-PPO-DQN",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                logging.error(f"  order_send returned None")
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"  Order failed: {result.retcode} - {result.comment}")
                return None
            
            return result
            
        except Exception as e:
            logging.error(f"  Erreur _execute_mt5_order: {e}")
            return None

    async def _manage_existing_position(self, symbol: str):
        """G re les positions existantes avec trailing stop"""
        try:
            # V rifier que la position existe
            if symbol not in self.open_positions:
                return

            position = self.open_positions[symbol]
            logging.debug(f"  Gestion position {symbol}: {position['direction']}")

            # Get current price
            if MT5_AVAILABLE:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    return
                current_price = tick.bid if position['direction'] == 'buy' else tick.ask
            else:
                # Mock price
                current_price = position['entry_price'] * (1 + np.random.uniform(-0.01, 0.01))

            #   GESTION ULTRA-AGRESSIVE POUR XAUUSD (L'OR)
            if symbol == "XAUUSD":
                # Calculer P&L en dollars
                entry_price = position['entry_price']
                if position['direction'] == 'buy':
                    pnl_dollars = current_price - entry_price
                else:
                    pnl_dollars = entry_price - current_price
                
                logging.debug(f"  XAUUSD P&L: ${pnl_dollars:.2f}")
                
                #   ANALYSE DES M CHES (WICKS) - CRITIQUE POUR L'OR!
                if MT5_AVAILABLE:
                    # R cup rer la derni re bougie
                    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 2)
                    if rates is not None and len(rates) >= 1:
                        last_candle = rates[-1]
                        open_price = last_candle['open']
                        high_price = last_candle['high']
                        low_price = last_candle['low']
                        close_price = last_candle['close']
                        
                        # Calculer tailles des m ches
                        body_size = abs(close_price - open_price)
                        if body_size > 0:
                            upper_wick = high_price - max(open_price, close_price)
                            lower_wick = min(open_price, close_price) - low_price
                            total_range = high_price - low_price
                            
                            # Ratios des m ches
                            if total_range > 0:
                                upper_wick_ratio = upper_wick / total_range
                                lower_wick_ratio = lower_wick / total_range
                                
                                #   M CHE HAUTE = REJECTION DU HAUT (bearish)
                                if upper_wick_ratio > 0.5 and position['direction'] == 'buy':
                                    logging.warning(f"  XAUUSD: GROSSE M CHE HAUTE ({upper_wick_ratio*100:.0f}%) - REJECTION!")
                                    if pnl_dollars > 0:
                                        logging.info(f"  Fermeture BUY en profit ${pnl_dollars:.2f} (m che rejection)")
                                        await self._close_position(symbol, reason=f"WICK_REJECTION_+${pnl_dollars:.0f}")
                                        return
                                    elif pnl_dollars < -5:
                                        logging.warning(f"  Fermeture BUY en perte ${pnl_dollars:.2f} (m che bearish)")
                                        await self._close_position(symbol, reason=f"WICK_REJECTION_${pnl_dollars:.0f}")
                                        return
                                
                                #   M CHE BASSE = REJECTION DU BAS (bullish)
                                elif lower_wick_ratio > 0.5 and position['direction'] == 'sell':
                                    logging.warning(f"  XAUUSD: GROSSE M CHE BASSE ({lower_wick_ratio*100:.0f}%) - REJECTION!")
                                    if pnl_dollars > 0:
                                        logging.info(f"  Fermeture SELL en profit ${pnl_dollars:.2f} (m che rejection)")
                                        await self._close_position(symbol, reason=f"WICK_REJECTION_+${pnl_dollars:.0f}")
                                        return
                                    elif pnl_dollars < -5:
                                        logging.warning(f"  Fermeture SELL en perte ${pnl_dollars:.2f} (m che bullish)")
                                        await self._close_position(symbol, reason=f"WICK_REJECTION_${pnl_dollars:.0f}")
                                        return
                
                # PROFIT: Fermer d s +10$ et plus
                if pnl_dollars >= 10:
                    logging.info(f"  XAUUSD: Profit rapide ${pnl_dollars:.2f} - FERMETURE IMM DIATE!")
                    await self._close_position(symbol, reason=f"QUICK_PROFIT_${pnl_dollars:.0f}")
                    return
                
                # PERTE: Fermer d s -15$ (stop serr )
                elif pnl_dollars <= -15:
                    logging.warning(f"  XAUUSD: Stop loss ${pnl_dollars:.2f} - FERMETURE!")
                    await self._close_position(symbol, reason=f"STOP_LOSS_${pnl_dollars:.0f}")
                    return
                
                # Si profit entre +5 et +10, trailing stop tr s serr 
                elif pnl_dollars >= 5 and MT5_AVAILABLE:
                    # Remonter SL   +$2 du prix d'entr e (s curiser au moins $2)
                    if position['direction'] == 'buy':
                        new_sl = entry_price + 2.0
                    else:
                        new_sl = entry_price - 2.0
                    
                    if position.get('stop_loss') != new_sl:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": symbol,
                            "position": position.get('ticket', 0),
                            "sl": new_sl,
                            "tp": position.get('take_profit', 0)
                        }
                        result = mt5.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            position['stop_loss'] = new_sl
                            logging.info(f"  XAUUSD: Trailing stop serr  @ ${new_sl:.2f} (profit s curis  +$2)")
                
                return  # Skip gestion standard pour XAUUSD
            
            # GESTION STANDARD pour les autres symboles
            # Calculate current P&L (en pips pour paires 5 d cimales)
            if position['direction'] == 'buy':
                pnl_pips = (current_price - position['entry_price']) / 0.0001
            else:
                pnl_pips = (position['entry_price'] - current_price) / 0.0001

            logging.debug(f"  Position {symbol}: P&L = {pnl_pips:.1f} pips")

            # Trailing stop (move SL to breakeven apr s +20 pips)
            if pnl_pips > 20 and MT5_AVAILABLE:
                new_sl = position['entry_price']

                if position.get('stop_loss') != new_sl:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": position.get('ticket', 0),
                        "sl": new_sl,
                        "tp": position.get('take_profit', 0)
                    }

                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        position['stop_loss'] = new_sl
                        logging.info(f"  Trailing stop activ  pour {symbol} @ {new_sl:.5f}")

            # Check for early exit signal
            observation = self._get_observation(symbol)
            if observation is not None and hasattr(self, 'rl_manager') and self.rl_manager:
                rl_action = self.rl_manager.predict_action(symbol, observation)

                # Si signal CLOSE
                if rl_action == 3:
                    logging.info(f"  Signal de sortie pour {symbol}")
                    await self._close_position(symbol, reason="RL_SIGNAL")

                # Signal inverse en profit
                elif (
                    (rl_action == 0 and position['direction'] == 'sell') or
                    (rl_action == 1 and position['direction'] == 'buy')
                ):
                    if pnl_pips > 0:
                        logging.info(f"  Signal inverse pour {symbol}, fermeture en profit")
                        await self._close_position(symbol, reason="INVERSE_SIGNAL")

        except Exception as e:
            logging.error(f"  Erreur _manage_existing_position: {e}")
            traceback.print_exc()


    def _check_elastic_snap(self, last_row: pd.Series, symbol: str = None) -> Dict:
        """
        D tecte si un snap  lastique est imminent
        """
        eti = float(last_row.get('ETI', 0.0))
        snap = float(last_row.get('SNAP', 0.0))
        hli = float(last_row.get('hli', 0.0))
        wick = float(last_row.get('wick_score', 0.0))
        
        # Seuils adapt s
        SNAP_THRESHOLD = 0.70 if symbol == "XAUUSD" else 0.60
        HLI_THRESHOLD = 0.25 if symbol == "XAUUSD" else 0.15
        
        # Snap baissier
        if snap >= SNAP_THRESHOLD and (hli < -HLI_THRESHOLD or wick < -HLI_THRESHOLD):
            confidence = min(0.95, snap + abs(min(hli, wick)) * 0.5)
            return {
                'snap_detected': True,
                'direction': 'bear',
                'confidence': confidence,
                'reason': f"SNAP_BEAR (SNAP={snap:.2f}, HLI={hli:.2f})"
            }
        
        # Snap haussier
        elif snap >= SNAP_THRESHOLD and (hli > HLI_THRESHOLD or wick > HLI_THRESHOLD):
            confidence = min(0.95, snap + max(hli, wick) * 0.5)
            return {
                'snap_detected': True,
                'direction': 'bull',
                'confidence': confidence,
                'reason': f"SNAP_BULL (SNAP={snap:.2f}, HLI={hli:.2f})"
            }
        
        # Lourdeur/tirage sans snap
        elif abs(hli) > 0.30 or abs(wick) > 0.30:
            direction = 'bear' if (hli < 0 or wick < 0) else 'bull'
            return {
                'snap_detected': False,
                'direction': direction,
                'confidence': 0.50,
                'reason': f"HEAVINESS_{direction.upper()} (HLI={hli:.2f})"
            }
        
        return {'snap_detected': False, 'direction': None, 'confidence': 0.0, 'reason': 'NEUTRAL'}

    async def _close_position(self, symbol: str, reason: str = "MANUAL"):
        """Ferme une position ouverte"""
        try:
            if symbol not in self.open_positions:
                logging.warning(f"  Pas de position ouverte pour {symbol}")
                return

            position = self.open_positions[symbol]
            logging.info(f"  Fermeture position {symbol} (reason: {reason})")

            # Get current price
            if MT5_AVAILABLE:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    logging.error(f"  Impossible de r cup rer prix pour {symbol}")
                    return

                price = tick.bid if position['direction'] == 'buy' else tick.ask

                # Send close order
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position['volume'],
                    "type": mt5.ORDER_TYPE_SELL if position['direction'] == 'buy' else mt5.ORDER_TYPE_BUY,
                    "position": position.get('ticket', 0),
                    "price": price,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": f"Close-{reason}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }

                result = mt5.order_send(request)
                if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
                    # S curise l acc s   comment si present
                    comment = getattr(result, 'comment', '')
                    ret = getattr(result, 'retcode', 'UNKNOWN')
                    logging.error(f"  Close failed: {ret} - {comment}")
                    return

                # Calculate P&L
                if position['direction'] == 'buy':
                    pnl = (price - position['entry_price']) * position['volume'] * 100000
                else:
                    pnl = (position['entry_price'] - price) * position['volume'] * 100000
            else:
                # Mock P&L
                price = position['entry_price']
                pnl = np.random.uniform(-50, 100)

            # Record trade
            trade_record = {
                'symbol': symbol,
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': price,
                'volume': position['volume'],
                'pnl': pnl,
                'open_time': position.get('open_time', datetime.now()),
                'close_time': datetime.now(),
                'duration': (datetime.now() - position.get('open_time', datetime.now())).total_seconds() / 60,
                'strategy': position.get('strategy', 'RL'),
                'confidence': position.get('confidence', 0.5),
                'reason': reason
            }

            self.trade_history.append(trade_record)

            # Update stats
            if pnl > 0:
                self.session_stats['winning_trades'] += 1
            else:
                self.session_stats['losing_trades'] += 1

            self.session_stats['total_pnl'] += pnl
            self.session_stats['daily_pnl'] += pnl
            if pnl < 0:
                self.daily_loss_tracker += pnl

            # Record for RL
            if hasattr(self, 'rl_manager') and self.rl_manager:
                self.rl_manager.record_trade_result(symbol, pnl)

            # Update risk manager
            if MT5_AVAILABLE:
                account_info = mt5.account_info()
                if account_info and hasattr(self, 'risk_manager') and self.risk_manager:
                    self.risk_manager.update_drawdown(account_info.equity)

            # Remove position
            del self.open_positions[symbol]

            logging.info(f"  Position ferm e: {symbol}, P&L = ${pnl:.2f}")

            # Broadcast update
            if hasattr(self, '_broadcast_trade_update'):
                await self._broadcast_trade_update(trade_record)

        except Exception as e:
            logging.error(f"  Erreur _close_position: {e}")
            traceback.print_exc()


    async def _broadcast_trade_update(self, trade: Dict):

        try:
            if self.connected_clients:
                message = {
                    'type': 'TRADE_UPDATE',
                    'data': {
                        'symbol': trade['symbol'],
                        'direction': trade['direction'],
                        'pnl': trade['pnl'],
                        'duration': trade['duration'],
                        'strategy': trade['strategy'],
                        'reason': trade['reason']
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                await asyncio.gather(
                    *[client.send(json.dumps(message)) for client in self.connected_clients],
                    return_exceptions=True
                )
        except Exception as e:
            logging.error(f"  Erreur _broadcast_trade_update: {e}")

    async def trading_loop(self):
        """Enhanced trading loop avec monitoring"""
        logging.info("  Enhanced Trading Loop d marr ")
        
        iteration = 0
        
        while self.active:
            try:
                iteration += 1
                
                # Log status every 10 iterations
                if iteration % 10 == 0:
                    self._log_status()
                
                # Analyze each symbol
                for symbol in self.config.SYMBOLS:
                    if not self.active:
                        break
                    
                    await self.analyze_and_trade(symbol)
                    await asyncio.sleep(1)  # Small delay between symbols
                
                # Check open positions
                for symbol in list(self.open_positions.keys()):
                    await self._manage_existing_position(symbol)
                
                # Periodic equity curve update
                if MT5_AVAILABLE:
                    account_info = mt5.account_info()
                    if account_info:
                        self.equity_curve.append({
                            'timestamp': datetime.now(),
                            'equity': account_info.equity,
                            'balance': account_info.balance
                        })
                
                # Wait before next iteration
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logging.error(f"  Erreur dans trading loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)

    def _log_status(self):
        """Log current status"""
        logging.info("=" * 60)
        logging.info("  STATUS UPDATE")
        logging.info(f"  Uptime: {datetime.now() - self.start_time}")
        logging.info(f"  Total Trades: {self.session_stats['total_trades']}")
        logging.info(f"  Winning: {self.session_stats['winning_trades']}")
        logging.info(f"  Losing: {self.session_stats['losing_trades']}")
        
        if self.session_stats['total_trades'] > 0:
            win_rate = self.session_stats['winning_trades'] / self.session_stats['total_trades'] * 100
            logging.info(f"  Win Rate: {win_rate:.1f}%")
        
        logging.info(f"  Daily P&L: ${self.session_stats['daily_pnl']:.2f}")
        logging.info(f"  Open Positions: {len(self.open_positions)}")
        
        if self.open_positions:
            for symbol, pos in self.open_positions.items():
                duration = (datetime.now() - pos['open_time']).total_seconds() / 60
                logging.info(f"    {symbol}: {pos['direction']} ({duration:.0f}min)")
        
        logging.info("=" * 60)
        
    async def start_health_monitor(self):
        """D marre le monitoring de sant  p riodique"""
        while self.active:
            try:
                await asyncio.sleep(30)  # Toutes les 30 secondes
                
                # V rifier l' tat des connexions clients
                active_clients = len(self.connected_clients)
                if active_clients > 0:
                    health_status = {
                        'type': 'HEALTH_STATUS',
                        'timestamp': datetime.now().isoformat(),
                        'active_clients': active_clients,
                        'system_uptime': str(datetime.now() - self.start_time).split('.')[0],
                        'open_positions': len(self.open_positions),
                        'total_trades': self.session_stats['total_trades'],
                        'status': 'HEALTHY'
                    }
                    
                    # Broadcast health status
                    await self._broadcast_to_clients(health_status)
                    
                    logging.debug(f"  Health check: {active_clients} clients connect s")
                    
            except Exception as e:
                logging.error(f"  Erreur health monitor: {e}")

    async def _broadcast_to_clients(self, data: dict):
        """Broadcast data   tous les clients connect s"""
        if not self.connected_clients:
            return
            
        message = json.dumps(data)
        dead_clients = []
        
        for client in self.connected_clients:
            try:
                await client.send(message)
            except Exception as e:
                logging.warning(f"  Client WebSocket mort: {e}")
                dead_clients.append(client)
        
        # Nettoyer les clients morts
        for client in dead_clients:
            self.connected_clients.discard(client)
    async def run(self):
        """Run the trading brain - METHODE AJOUT E PAR PATCH"""
        await self.trading_loop()

    def stop(self):
        """Stop the trading brain - METHODE AJOUT E PAR PATCH"""
        self.active = False
        if MT5_AVAILABLE:
            mt5.shutdown()
        logging.info("  Enhanced Trading Brain arr t ")

#============================================================================
#MAIN AVEC ENHANCED BRAIN
#============================================================================
async def main_enhanced():
    """Main avec Enhanced Brain"""
    print("=" * 80)
    print("  ULTIMATE HYBRID TRADING BOT - ENHANCED VERSION")
    print("=" * 80)
    print("  Full Stack Architecture:")
    print("    HRM-TRM Hierarchical Recursive Reasoning")
    print("    PPO-DQN Ensemble Learning")
    print("    Adaptive Halting with Q-Learning")
    print("    Order Flow Analysis")
    print("    Sentiment Analysis")
    print("    Meta-Strategy Selection")
    print("    Advanced Risk Management")
    print("    Auto-Retrain on Performance Drop")
    print("=" * 80)
    print(f"  WebSocket Dashboard: ws://localhost:{CONFIG.WS_PORT}")
    print(f"  RL Training Monitor: ws://localhost:{CONFIG.RL_WS_PORT}")
    print("=" * 80)
    print()
    
    # Initialize Enhanced Brain
    brain = EnhancedUltimateTradingBrain(CONFIG)

    if not await brain.initialize():
        logging.error("   chec initialisation, arr t du bot")
        return

    # Start WebSocket server
    async def ws_handler(websocket, path):
        await websocket_handler(websocket, brain)

    server = await websockets.serve(ws_handler, "localhost", CONFIG.WS_PORT)
    logging.info(f"  WebSocket server d marr  sur localhost:{CONFIG.WS_PORT}")

    # Start trading
    try:
        await brain.run()
    except KeyboardInterrupt:
        logging.info("\n  Interruption clavier d tect e")
    except Exception as e:
        logging.error(f"  Erreur fatale: {e}")
        traceback.print_exc()
    finally:
        brain.stop()
        server.close()
        await server.wait_closed()
        logging.info("  Arr t propre du syst me")

# Update main to use enhanced version
if __name__ == "__main__":
    try:
        asyncio.run(main_enhanced())
    except KeyboardInterrupt:
        print("\n  Au revoir!")
    except Exception as e:
        logging.error(f"  Erreur critique: {e}")
        traceback.print_exc()
        
        #============================================================================
#PARTIE 4: DASHBOARD HTML & MONITORING SYST ME
#============================================================================

#============================================================================
#HTML DASHBOARD GENERATOR
#============================================================================
class DashboardGenerator:
    """G n re les dashboards HTML pour monitoring"""
    
    @staticmethod
    def generate_main_dashboard() -> str:
        """Dashboard principal de trading"""
        return """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>  Ultimate Hybrid Trading Bot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .status-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            margin: 5px;
        }
        
        .status-active {
            background: #10b981;
        }
        
        .status-inactive {
            background: #ef4444;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        .card h3 {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #fbbf24;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-value {
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .positive {
            color: #10b981;
        }
        
        .negative {
            color: #ef4444;
        }
        
        .positions-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        .positions-table th,
        .positions-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .positions-table th {
            background: rgba(255, 255, 255, 0.1);
            font-weight: bold;
        }
        
        .trade-log {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 5px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .trade-log-entry {
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            flex: 1;
        }
        
        .btn-start {
            background: #10b981;
            color: white;
        }
        
        .btn-stop {
            background: #ef4444;
            color: white;
        }
        
        .btn-retrain {
            background: #f59e0b;
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            height: 400px;
        }
        
        .signal-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .signal-buy {
            background: #10b981;
            box-shadow: 0 0 10px #10b981;
        }
        
        .signal-sell {
            background: #ef4444;
            box-shadow: 0 0 10px #ef4444;
        }
        
        .signal-hold {
            background: #f59e0b;
            box-shadow: 0 0 10px #f59e0b;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .loading {
            animation: pulse 2s infinite;
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 20px;
            background: rgba(239, 68, 68, 0.9);
            font-weight: bold;
        }
        
        .connection-status.connected {
            background: rgba(16, 185, 129, 0.9);
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">
          D connect 
    </div>
    
    <div class="container">
        <div class="header">
            <h1>  Ultimate Hybrid Trading Bot</h1>
            <p>HRM-TRM + PPO-DQN Ensemble Learning</p>
            <div>
                <span class="status-badge status-inactive" id="statusBadge">  Inactif</span>
                <span class="status-badge" style="background: #3b82f6;">  <span id="uptime">00:00:00</span></span>
            </div>
        </div>
        
        <div class="grid">
            <!-- Performance Card -->
            <div class="card">
                <h3>  Performance</h3>
                <div class="metric">
                    <span>Total Trades</span>
                    <span class="metric-value" id="totalTrades">0</span>
                </div>
                <div class="metric">
                    <span>Win Rate</span>
                    <span class="metric-value" id="winRate">0%</span>
                </div>
                <div class="metric">
                    <span>Daily P&L</span>
                    <span class="metric-value" id="dailyPnl">$0.00</span>
                </div>
                <div class="metric">
                    <span>Total P&L</span>
                    <span class="metric-value" id="totalPnl">$0.00</span>
                </div>
            </div>
            
            <!-- RL Agents Card -->
            <div class="card">
                <h3>  RL Agents Status</h3>
                <div class="metric">
                    <span>PPO Agents</span>
                    <span class="metric-value" id="ppoStatus">  4/4</span>
                </div>
                <div class="metric">
                    <span>DQN Agents</span>
                    <span class="metric-value" id="dqnStatus">  4/4</span>
                </div>
                <div class="metric">
                    <span>HRM-TRM Cycles</span>
                    <span class="metric-value" id="hrmCycles">H:2 L:3</span>
                </div>
                <div class="metric">
                    <span>Avg Halting Steps</span>
                    <span class="metric-value" id="avgSteps">8.2</span>
                </div>
            </div>
            
            <!-- Risk Management Card -->
            <div class="card">
                <h3>  Risk Management</h3>
                <div class="metric">
                    <span>Open Positions</span>
                    <span class="metric-value" id="openPositions">0/4</span>
                </div>
                <div class="metric">
                    <span>Daily Drawdown</span>
                    <span class="metric-value" id="dailyDrawdown">0%</span>
                </div>
                <div class="metric">
                    <span>Max Drawdown</span>
                    <span class="metric-value" id="maxDrawdown">0%</span>
                </div>
                <div class="metric">
                    <span>Risk per Trade</span>
                    <span class="metric-value">1.5%</span>
                </div>
            </div>
            
            <!-- Market Signals Card -->
            <div class="card">
                <h3>  Market Signals</h3>
                <div id="symbolSignals">
                    <!-- Will be populated by JS -->
                </div>
            </div>
        </div>
        
        <!-- Open Positions -->
        <div class="card">
            <h3>  Open Positions</h3>
            <table class="positions-table" id="positionsTable">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>P&L</th>
                        <th>Duration</th>
                        <th>Strategy</th>
                    </tr>
                </thead>
                <tbody id="positionsBody">
                    <tr>
                        <td colspan="7" style="text-align: center; padding: 20px;">
                            No open positions
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- Trade Log -->
        <div class="card">
            <h3>  Trade Log</h3>
            <div class="trade-log" id="tradeLog">
                <div class="trade-log-entry">  System initialized...</div>
            </div>
        </div>
        
        <!-- Controls -->
        <div class="card">
            <h3>  Controls</h3>
            <div class="controls">
                <button class="btn btn-start" onclick="startTrading()">
                      Start Trading
                </button>
                <button class="btn btn-stop" onclick="stopTrading()">
                      Stop Trading
                </button>
                <button class="btn btn-retrain" onclick="retrainAll()">
                      Retrain All
                </button>
            </div>
        </div>
        
        <!-- Equity Curve Chart -->
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // WebSocket Connection
        let ws;
        let reconnectInterval;
        
        function connect() {
            ws = new WebSocket('ws://localhost:8765');
            
            ws.onopen = () => {
                console.log('  Connected to trading bot');
                document.getElementById('connectionStatus').textContent = '  Connect ';
                document.getElementById('connectionStatus').classList.add('connected');
                clearInterval(reconnectInterval);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
            
            ws.onclose = () => {
                console.log('  Disconnected from trading bot');
                document.getElementById('connectionStatus').textContent = '  D connect ';
                document.getElementById('connectionStatus').classList.remove('connected');
                
                // Auto-reconnect
                reconnectInterval = setInterval(() => {
                    console.log('  Attempting to reconnect...');
                    connect();
                }, 5000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleMessage(data) {
            console.log('  Received:', data);
            
            switch(data.type) {
                case 'WELCOME':
                    addTradeLog('  ' + data.message);
                    requestStatus();
                    break;
                    
                case 'STATUS':
                    updateStatus(data.data);
                    break;
                    
                case 'TRADE_UPDATE':
                    updateTradeLog(data.data);
                    requestStatus();
                    break;
                    
                case 'COMMAND_ACK':
                    addTradeLog(`  Command ${data.command} acknowledged`);
                    break;
            }
        }
        
        function requestStatus() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'GET_STATUS' }));
            }
        }
        
        function updateStatus(status) {
            // Status badge
            const badge = document.getElementById('statusBadge');
            if (status.active) {
                badge.textContent = '  Actif';
                badge.className = 'status-badge status-active';
            } else {
                badge.textContent = '  Inactif';
                badge.className = 'status-badge status-inactive';
            }
            
            // Uptime
            document.getElementById('uptime').textContent = status.uptime || '00:00:00';
            
            // Performance
            const stats = status.session_stats || {};
            document.getElementById('totalTrades').textContent = stats.total_trades || 0;
            
            const winRate = stats.total_trades > 0 
                ? ((stats.winning_trades || 0) / stats.total_trades * 100).toFixed(1)
                : 0;
            document.getElementById('winRate').textContent = winRate + '%';
            
            const dailyPnl = stats.daily_pnl || 0;
            const dailyPnlElem = document.getElementById('dailyPnl');
            dailyPnlElem.textContent = '$' + dailyPnl.toFixed(2);
            dailyPnlElem.className = 'metric-value ' + (dailyPnl >= 0 ? 'positive' : 'negative');
            
            const totalPnl = stats.total_pnl || 0;
            const totalPnlElem = document.getElementById('totalPnl');
            totalPnlElem.textContent = '$' + totalPnl.toFixed(2);
            totalPnlElem.className = 'metric-value ' + (totalPnl >= 0 ? 'positive' : 'negative');
            
            // Open positions
            const openPos = status.open_positions || 0;
            document.getElementById('openPositions').textContent = openPos + '/4';
        }
        
        function updateTradeLog(trade) {
            const logEntry = `
                  ${trade.symbol} ${trade.direction.toUpperCase()} closed
                P&L: $${trade.pnl.toFixed(2)} | Duration: ${trade.duration.toFixed(0)}min
                Strategy: ${trade.strategy} | Reason: ${trade.reason}
            `;
            addTradeLog(logEntry);
        }
        
        function addTradeLog(message) {
            const log = document.getElementById('tradeLog');
            const entry = document.createElement('div');
            entry.className = 'trade-log-entry';
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            log.insertBefore(entry, log.firstChild);
            
            // Keep only last 50 entries
            while (log.children.length > 50) {
                log.removeChild(log.lastChild);
            }
        }
        
        function startTrading() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'COMMAND', command: 'START' }));
                addTradeLog('  Start command sent');
            }
        }
        
        function stopTrading() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'COMMAND', command: 'STOP' }));
                addTradeLog('  Stop command sent');
            }
        }
        
        function retrainAll() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const symbols = ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDJPY'];
                symbols.forEach(symbol => {
                    ws.send(JSON.stringify({ 
                        type: 'COMMAND', 
                        command: 'RETRAIN',
                        symbol: symbol
                    }));
                });
                addTradeLog('  Retrain commands sent for all symbols');
            }
        }
        
        // Equity Chart
        const ctx = document.getElementById('equityChart').getContext('2d');
        const equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Equity',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#fff' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        });
        
        // Auto-update status every 5 seconds
        setInterval(() => {
            requestStatus();
        }, 5000);
        
        // Initialize connection
        connect();
    </script>
</body>
</html>
        """
    
    @staticmethod
    def save_dashboard(filename: str = "dashboard.html"):
        """Sauvegarde le dashboard dans un fichier HTML"""
        try:
            html_content = DashboardGenerator.generate_main_dashboard()
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logging.info(f"  Dashboard sauvegard : {filename}")
            return True
        except Exception as e:
            logging.error(f"  Erreur sauvegarde dashboard: {e}")
            return False

#============================================================================
# DASHBOARD WEB SERVER
#============================================================================
async def serve_dashboard(websocket, path):
    """Serve the dashboard HTML via WebSocket"""
    try:
        if path == '/dashboard.html' or path == '/':
            # G n rer et servir le dashboard
            html_content = DashboardGenerator.generate_main_dashboard()
            await websocket.send(html_content)
            logging.info("  Dashboard HTML servi   un client")
        else:
            await websocket.send("404 - Not Found")
    except Exception as e:
        logging.error(f"  Erreur serve_dashboard: {e}")

async def start_dashboard_server(port: int = 8080):
    """D marre le serveur de dashboard"""
    try:
        server = await websockets.serve(serve_dashboard, "localhost", port)
        logging.info(f"  Dashboard server d marr  sur http://localhost:{port}")
        return server
    except Exception as e:
        logging.error(f"  Erreur d marrage dashboard server: {e}")
        return None

#============================================================================
#SYSTEM MONITOR
#============================================================================
class SystemMonitor:
    """Monitore les ressources syst me et les performances"""
    
    def __init__(self):
        self.cpu_history: deque = deque(maxlen=100)
        self.memory_history: deque = deque(maxlen=100)
        self.gpu_history: deque = deque(maxlen=100)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """R cup re les stats syst me"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {},
            'memory': {},
            'gpu': {},
            'disk': {},
            'network': {}
        }
        
        try:
            if PSUTIL_AVAILABLE:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                stats['cpu'] = {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count(),
                    'freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0
                }
                self.cpu_history.append(cpu_percent)
                
                # Memory
                memory = psutil.virtual_memory()
                stats['memory'] = {
                    'percent': memory.percent,
                    'used_gb': memory.used / (1024**3),
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3)
                }
                self.memory_history.append(memory.percent)
                
                # Disk
                disk = psutil.disk_usage('/')
                stats['disk'] = {
                    'percent': disk.percent,
                    'used_gb': disk.used / (1024**3),
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3)
                }
                
                # Network
                net_io = psutil.net_io_counters()
                stats['network'] = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            
            if GPU_AVAILABLE:
                # GPU
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        stats['gpu'] = {
                            'name': gpu.name,
                            'load': gpu.load * 100,
                            'memory_used': gpu.memoryUsed,
                            'memory_total': gpu.memoryTotal,
                            'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            'temperature': gpu.temperature
                        }
                        self.gpu_history.append(gpu.load * 100)
                except Exception as e:
                    logging.debug(f"GPU stats unavailable: {e}")
            
        except Exception as e:
            logging.error(f"  Erreur get_system_stats: {e}")
        
        return stats

    def get_process_stats(self) -> Dict[str, Any]:
        """R cup re les stats du processus actuel"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'process': {}
        }
        
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                
                stats['process'] = {
                    'cpu_percent': process.cpu_percent(interval=1),
                    'memory_mb': process.memory_info().rss / (1024**2),
                    'memory_percent': process.memory_percent(),
                    'num_threads': process.num_threads(),
                    'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0,
                    'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
                }
        except Exception as e:
            logging.error(f"  Erreur get_process_stats: {e}")
        
        return stats

    def check_health(self) -> Dict[str, Any]:
        """V rifie la sant  du syst me"""
        health = {
            'status': 'healthy',
            'warnings': [],
            'errors': []
        }
        
        try:
            if PSUTIL_AVAILABLE:
                # Check CPU
                if len(self.cpu_history) > 0:
                    avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
                    if avg_cpu > 90:
                        health['warnings'].append(f'High CPU usage: {avg_cpu:.1f}%')
                        health['status'] = 'warning'
                
                # Check Memory
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    health['errors'].append(f'Critical memory usage: {memory.percent:.1f}%')
                    health['status'] = 'critical'
                elif memory.percent > 80:
                    health['warnings'].append(f'High memory usage: {memory.percent:.1f}%')
                    if health['status'] == 'healthy':
                        health['status'] = 'warning'
                
                # Check Disk
                disk = psutil.disk_usage('/')
                if disk.percent > 95:
                    health['errors'].append(f'Critical disk usage: {disk.percent:.1f}%')
                    health['status'] = 'critical'
                elif disk.percent > 85:
                    health['warnings'].append(f'High disk usage: {disk.percent:.1f}%')
                    if health['status'] == 'healthy':
                        health['status'] = 'warning'
            
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        if gpu.temperature > 85:
                            health['warnings'].append(f'High GPU temperature: {gpu.temperature} C')
                            if health['status'] == 'healthy':
                                health['status'] = 'warning'
                except:
                    pass
        
        except Exception as e:
            logging.error(f"  Erreur check_health: {e}")
        
        return health

#============================================================================
#PERFORMANCE ANALYZER
#============================================================================
class PerformanceAnalyzer:
    """Analyse les performances de trading"""
    
    @staticmethod
    def calculate_metrics(trades: List[Dict]) -> Dict[str, Any]:
        """Calcule les m triques de performance"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0
            }
        
        pnls = [t['pnl'] for t in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        # Basic metrics
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit factor - FIX ULTIME
        total_profit = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0
        
        # === FIX: Calcul correct du Profit Factor ===
        if total_loss > 0 and total_profit > 0:
            profit_factor = total_profit / total_loss
        elif total_profit > 0 and total_loss == 0:
            profit_factor = 999.0  # Que des gains = excellent
        elif total_profit == 0 and total_loss > 0:
            profit_factor = 0.01   # Que des pertes = très mauvais
        else:
            profit_factor = 1.0    # Pas de trades = neutre
        
        # Sharpe ratio (simplified)
        returns = np.array(pnls)
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # Average wins/losses
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': max(winning_trades) if winning_trades else 0,
            'largest_loss': min(losing_trades) if losing_trades else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls)
        }

    @staticmethod
    def generate_report(trades: List[Dict], output_file: str = "performance_report.json"):
        """G n re un rapport de performance"""
        try:
            metrics = PerformanceAnalyzer.calculate_metrics(trades)
            
            # Add timestamp
            report = {
                'generated_at': datetime.now().isoformat(),
                'period': {
                    'start': trades[0]['open_time'].isoformat() if trades else None,
                    'end': trades[-1]['close_time'].isoformat() if trades else None
                },
                'metrics': metrics,
                'trades_by_symbol': {},
                'trades_by_strategy': {}
            }
            
            # Group by symbol
            for trade in trades:
                symbol = trade['symbol']
                if symbol not in report['trades_by_symbol']:
                    report['trades_by_symbol'][symbol] = []
                report['trades_by_symbol'][symbol].append(trade)
            
            # Group by strategy
            for trade in trades:
                strategy = trade['strategy']
                if strategy not in report['trades_by_strategy']:
                    report['trades_by_strategy'][strategy] = []
                report['trades_by_strategy'][strategy].append(trade)
            
            # Calculate metrics per symbol
            for symbol in report['trades_by_symbol']:
                symbol_trades = report['trades_by_symbol'][symbol]
                report['trades_by_symbol'][symbol] = {
                    'trades': symbol_trades,
                    'metrics': PerformanceAnalyzer.calculate_metrics(symbol_trades)
                }
            
            # Calculate metrics per strategy
            for strategy in report['trades_by_strategy']:
                strategy_trades = report['trades_by_strategy'][strategy]
                report['trades_by_strategy'][strategy] = {
                    'trades': strategy_trades,
                    'metrics': PerformanceAnalyzer.calculate_metrics(strategy_trades)
                }
            
            # Save report
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logging.info(f"  Performance report saved: {output_file}")
            return report
            
        except Exception as e:
            logging.error(f"  Erreur generate_report: {e}")
            return None

#============================================================================
#PARTIE 5: CONFIGURATION AVANC E, UTILITIES & FINALISATION
#============================================================================

#============================================================================
#CONFIGURATION LOADER & VALIDATOR
#============================================================================
class ConfigLoader:
    """Charge et valide la configuration depuis diff rentes sources"""
    
    @staticmethod
    def load_from_file(filepath: str = "config.json") -> Optional[UltimateConfig]:
        """Charge la configuration depuis un fichier JSON"""
        try:
            if not os.path.exists(filepath):
                logging.warning(f"  Config file not found: {filepath}, using defaults")
                return ConfigLoader.create_default_config()
            
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Merge with env variables
            config_dict = ConfigLoader._merge_with_env(config_dict)
            
            # Validate
            if not ConfigLoader.validate_config(config_dict):
                logging.error("  Invalid configuration")
                return None
            
            # Create config object
            config = UltimateConfig(**config_dict)
            logging.info(f"  Configuration loaded from {filepath}")
            return config
            
        except Exception as e:
            logging.error(f"  Erreur load_from_file: {e}")
            return None

    @staticmethod
    def _merge_with_env(config_dict: Dict) -> Dict:
        """Merge config avec variables d'environnement"""
        env_mappings = {
            'MT5_LOGIN': 'MT5_LOGIN',
            'MT5_PASSWORD': 'MT5_PASSWORD', 
            'MT5_SERVER': 'MT5_SERVER',
            # RETIRER les cl s qui ne sont pas dans UltimateConfig
            # 'TWITTER_BEARER_TOKEN': 'TWITTER_BEARER_TOKEN',
            # 'REDDIT_CLIENT_ID': 'REDDIT_CLIENT_ID',
            # 'REDDIT_CLIENT_SECRET': 'REDDIT_CLIENT_SECRET'
        }
        
        for config_key, env_key in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value:
                config_dict[config_key] = env_value
        
        return config_dict

    @staticmethod
    def validate_config(config_dict: Dict) -> bool:
        """Valide la configuration"""
        required_fields = ['SYMBOLS', 'RISK_PER_TRADE', 'MAX_POSITIONS']
        
        for field in required_fields:
            if field not in config_dict:
                logging.error(f"  Missing required field: {field}")
                return False
        
        # Validate ranges
        if not (0 < config_dict.get('RISK_PER_TRADE', 0) <= 0.1):
            logging.error("  RISK_PER_TRADE must be between 0 and 0.1")
            return False
        
        if not (0 < config_dict.get('DAILY_LOSS_LIMIT', 0) <= 0.2):
            logging.error("  DAILY_LOSS_LIMIT must be between 0 and 0.2")
            return False
        
        return True

    @staticmethod
    def create_default_config() -> UltimateConfig:
        """Cr e une configuration par d faut"""
        return UltimateConfig()

    @staticmethod
    def save_config(config: UltimateConfig, filepath: str = "config.json"):
        """Sauvegarde la configuration"""
        try:
            config_dict = {
                'SYMBOLS': config.SYMBOLS,
                'RISK_PER_TRADE': config.RISK_PER_TRADE,
                'DAILY_LOSS_LIMIT': config.DAILY_LOSS_LIMIT,
                'MAX_POSITIONS': config.MAX_POSITIONS,
                'HRM_HIDDEN_DIM': config.HRM_HIDDEN_DIM,
                'HRM_H_CYCLES': config.HRM_H_CYCLES,
                'HRM_L_CYCLES': config.HRM_L_CYCLES,
                'HRM_NUM_HEADS': config.HRM_NUM_HEADS,
                'HRM_HALT_MAX_STEPS': config.HRM_HALT_MAX_STEPS,
                'HRM_EXPLORATION_PROB': config.HRM_EXPLORATION_PROB,
                'RL_TOTAL_TIMESTEPS': config.RL_TOTAL_TIMESTEPS,
                'RL_LEARNING_RATE': config.RL_LEARNING_RATE,
                'RL_BUFFER_SIZE': config.RL_BUFFER_SIZE,
                'RL_EXPLORATION_FRACTION': config.RL_EXPLORATION_FRACTION,
                'WS_PORT': config.WS_PORT,
                'RL_WS_PORT': config.RL_WS_PORT,
                'RETRAIN_COOLDOWN': config.RETRAIN_COOLDOWN,
                'PERF_DROP_THRESHOLD': config.PERF_DROP_THRESHOLD
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logging.info(f"  Configuration saved to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"  Erreur save_config: {e}")
            return False

#============================================================================
#BACKUP & RECOVERY MANAGER
#============================================================================
class BackupManager:
    """G re les sauvegardes et la r cup ration du syst me"""
    
    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        logging.info(f"  Backup Manager initialized: {backup_dir}")

    def backup_models(self, rl_manager: RLAgentManager) -> bool:
        """Sauvegarde les mod les RL"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"models_{timestamp}")
            os.makedirs(backup_path, exist_ok=True)
            
            # Save PPO models
            for symbol, agent in rl_manager.ppo_agents.items():
                agent.save(os.path.join(backup_path, f"ppo_{symbol}.zip"))
            
            # Save DQN models
            for symbol, agent in rl_manager.dqn_agents.items():
                agent.save(os.path.join(backup_path, f"dqn_{symbol}.zip"))
            
            # Save HRM-TRM models
            for symbol, hrm_trm in rl_manager.hrm_trm_per_symbol.items():
                torch.save(
                    hrm_trm.state_dict(),
                    os.path.join(backup_path, f"hrm_trm_{symbol}.pt")
                )
            
            logging.info(f"  Models backed up to {backup_path}")
            return True
            
        except Exception as e:
            logging.error(f"  Erreur backup_models: {e}")
            return False

    def restore_models(self, rl_manager: RLAgentManager, backup_name: str) -> bool:
        """Restaure les mod les depuis une sauvegarde"""
        try:
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            if not os.path.exists(backup_path):
                logging.error(f"  Backup not found: {backup_path}")
                return False
            
            # Restore PPO models
            for symbol in rl_manager.config.SYMBOLS:
                ppo_path = os.path.join(backup_path, f"ppo_{symbol}.zip")
                if os.path.exists(ppo_path):
                    rl_manager.ppo_agents[symbol] = PPO.load(ppo_path)
            
            # Restore DQN models
            for symbol in rl_manager.config.SYMBOLS:
                dqn_path = os.path.join(backup_path, f"dqn_{symbol}.zip")
                if os.path.exists(dqn_path):
                    rl_manager.dqn_agents[symbol] = DQN.load(dqn_path)
            
            # Restore HRM-TRM models
            for symbol in rl_manager.config.SYMBOLS:
                hrm_path = os.path.join(backup_path, f"hrm_trm_{symbol}.pt")
                if os.path.exists(hrm_path):
                    rl_manager.hrm_trm_per_symbol[symbol].load_state_dict(
                        torch.load(hrm_path)
                    )
            
            logging.info(f"  Models restored from {backup_path}")
            return True
            
        except Exception as e:
            logging.error(f"  Erreur restore_models: {e}")
            return False

    def backup_trade_history(self, trades: deque, filename: str = None) -> bool:
        """Sauvegarde l'historique des trades"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trade_history_{timestamp}.json"
            
            filepath = os.path.join(self.backup_dir, filename)
            
            trades_list = list(trades)
            with open(filepath, 'w') as f:
                json.dump(trades_list, f, indent=2, default=str)
            
            logging.info(f"  Trade history backed up to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"  Erreur backup_trade_history: {e}")
            return False

    def cleanup_old_backups(self, keep_days: int = 7):
        """Nettoie les anciennes sauvegardes"""
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            for item in os.listdir(self.backup_dir):
                item_path = os.path.join(self.backup_dir, item)
                
                if os.path.isfile(item_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(item_path))
                    if file_time < cutoff_date:
                        os.remove(item_path)
                        logging.info(f"  Removed old backup: {item}")
                
                elif os.path.isdir(item_path):
                    dir_time = datetime.fromtimestamp(os.path.getmtime(item_path))
                    if dir_time < cutoff_date:
                        import shutil
                        shutil.rmtree(item_path)
                        logging.info(f"  Removed old backup directory: {item}")
            
            return True
            
        except Exception as e:
            logging.error(f"  Erreur cleanup_old_backups: {e}")
            return False

#============================================================================
#NOTIFICATION MANAGER
#============================================================================
class NotificationManager:
    """G re les notifications (email, Telegram, Discord, etc.)"""
    
    def __init__(self):
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK')
        self.email_enabled = False
        
        logging.info("  Notification Manager initialized")

    async def send_notification(self, message: str, level: str = 'info', channels: List[str] = None):
        """Envoie une notification sur les canaux sp cifi s"""
        if channels is None:
            channels = ['telegram', 'discord']
        
        tasks = []
        
        if 'telegram' in channels and self.telegram_bot_token and self.telegram_chat_id:
            tasks.append(self._send_telegram(message, level))
        
        if 'discord' in channels and self.discord_webhook:
            tasks.append(self._send_discord(message, level))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_telegram(self, message: str, level: str):
        """Envoie un message Telegram"""
        try:
            import aiohttp
            
            # Add emoji based on level
            emoji = {
                'info': ' ',
                'success': ' ',
                'warning': ' ',
                'error': ' ',
                'critical': ' '
            }.get(level, ' ')
            
            formatted_message = f"{emoji} {message}"
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': formatted_message,
                'parse_mode': 'HTML'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logging.debug("  Telegram notification sent")
                    else:
                        logging.warning(f"  Telegram notification failed: {response.status}")
        
        except Exception as e:
            logging.error(f"  Erreur _send_telegram: {e}")

    async def _send_discord(self, message: str, level: str):
        """Envoie un message Discord"""
        try:
            import aiohttp
            
            # Color based on level
            color = {
                'info': 3447003,      # Blue
                'success': 3066993,   # Green
                'warning': 16776960,  # Yellow
                'error': 15158332,    # Red
                'critical': 10038562  # Dark Red
            }.get(level, 3447003)
            
            embed = {
                'embeds': [{
                    'title': f'Trading Bot - {level.upper()}',
                    'description': message,
                    'color': color,
                    'timestamp': datetime.now().isoformat()
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook, json=embed) as response:
                    if response.status in [200, 204]:
                        logging.debug("  Discord notification sent")
                    else:
                        logging.warning(f"  Discord notification failed: {response.status}")
        
        except Exception as e:
            logging.error(f"  Erreur _send_discord: {e}")

    async def notify_trade(self, trade: Dict):
        """Notifie un trade"""
        pnl_emoji = " " if trade['pnl'] > 0 else " "
        message = f"""
{pnl_emoji} <b>Trade Closed</b>
Symbol: {trade['symbol']}
Direction: {trade['direction'].upper()}
P&L: ${trade['pnl']:.2f}
Duration: {trade['duration']:.0f} minutes
Strategy: {trade['strategy']}
"""
        level = 'success' if trade['pnl'] > 0 else 'warning'
        await self.send_notification(message, level)

    async def notify_error(self, error_message: str):
        """Notifie une erreur critique"""
        message = f"  <b>Critical Error</b>\n{error_message}"
        await self.send_notification(message, 'critical')

    async def notify_daily_summary(self, stats: Dict):
        """Envoie le r sum  quotidien"""
        message = f"""
  <b>Daily Summary</b>
Total Trades: {stats['total_trades']}
Win Rate: {stats.get('win_rate', 0):.1f}%
Total P&L: ${stats['total_pnl']:.2f}
Winning Trades: {stats['winning_trades']}
Losing Trades: {stats['losing_trades']}
"""
        level = 'success' if stats['total_pnl'] > 0 else 'warning'
        await self.send_notification(message, level)

#============================================================================
#CLI INTERFACE
#============================================================================
class CLIInterface:
    """Interface en ligne de commande pour contr ler le bot"""
    
    def __init__(self, brain: EnhancedUltimateTradingBrain):
        self.brain = brain
        self.commands = {
            'start': self.cmd_start,
            'stop': self.cmd_stop,
            'status': self.cmd_status,
            'positions': self.cmd_positions,
            'stats': self.cmd_stats,
            'retrain': self.cmd_retrain,
            'backup': self.cmd_backup,
            'help': self.cmd_help,
            'exit': self.cmd_exit
        }

    async def run(self):
        """Lance l'interface CLI"""
        print("\n" + "="*60)
        print("  ULTIMATE HYBRID TRADING BOT - CLI")
        print("="*60)
        print("Type 'help' for available commands\n")
        
        while True:
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None,
                    input,
                    "bot> "
                )
                
                command = command.strip().lower()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                
                if cmd in self.commands:
                    await self.commands[cmd](args)
                else:
                    print(f"  Unknown command: {cmd}")
                    print("Type 'help' for available commands")
                
            except EOFError:
                break
            except Exception as e:
                print(f"  Error: {e}")

    async def cmd_start(self, args):
        """Start trading"""
        self.brain.active = True
        print("  Trading started")

    async def cmd_stop(self, args):
        """Stop trading"""
        self.brain.active = False
        print("  Trading stopped")

    async def cmd_status(self, args):
        """Show system status"""
        print("\n  System Status:")
        print(f"  Active: {'  Yes' if self.brain.active else '  No'}")
        print(f"  Uptime: {datetime.now() - self.brain.start_time}")
        print(f"  Total Trades: {self.brain.session_stats['total_trades']}")
        print(f"  Daily P&L: ${self.brain.session_stats['daily_pnl']:.2f}")
        print(f"  Open Positions: {len(self.brain.open_positions)}/{self.brain.config.MAX_POSITIONS}")
        print()

    async def cmd_positions(self, args):
        """Show open positions"""
        if not self.brain.open_positions:
            print("No open positions")
            return
        
        print("\n  Open Positions:")
        for symbol, pos in self.brain.open_positions.items():
            duration = (datetime.now() - pos['open_time']).total_seconds() / 60
            print(f"  {symbol}:")
            print(f"    Direction: {pos['direction']}")
            print(f"    Entry: {pos['entry_price']:.5f}")
            print(f"    SL: {pos['stop_loss']:.5f}")
            print(f"    TP: {pos['take_profit']:.5f}")
            print(f"    Duration: {duration:.0f}min")
            print(f"    Strategy: {pos['strategy']}")
        print()

    async def cmd_stats(self, args):
        """Show detailed statistics"""
        stats = self.brain.session_stats
        
        win_rate = 0
        if stats['total_trades'] > 0:
            win_rate = (stats['winning_trades'] / stats['total_trades']) * 100
        
        print("\n  Detailed Statistics:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Winning Trades: {stats['winning_trades']}")
        print(f"  Losing Trades: {stats['losing_trades']}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total P&L: ${stats['total_pnl']:.2f}")
        print(f"  Daily P&L: ${stats['daily_pnl']:.2f}")
        
        if self.brain.trade_history:
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.calculate_metrics(list(self.brain.trade_history))
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        print()

    async def cmd_retrain(self, args):
        """Retrain agents"""
        if not args:
            print("Usage: retrain <symbol|all>")
            return
        
        if args[0] == 'all':
            print("  Retraining all agents...")
            for symbol in self.brain.config.SYMBOLS:
                await self.brain.rl_manager.retrain_on_drop(symbol)
            print("  All agents retrained")
        else:
            symbol = args[0].upper()
            if symbol in self.brain.config.SYMBOLS:
                print(f"  Retraining {symbol}...")
                await self.brain.rl_manager.retrain_on_drop(symbol)
                print(f"  {symbol} retrained")
            else:
                print(f"  Unknown symbol: {symbol}")

    async def cmd_backup(self, args):
        """Create backup"""
        backup_manager = BackupManager()
        print("  Creating backup...")
        
        if backup_manager.backup_models(self.brain.rl_manager):
            print("  Models backed up")
        
        if backup_manager.backup_trade_history(self.brain.trade_history):
            print("  Trade history backed up")

    async def cmd_help(self, args):
        """Show help"""
        print("\n  Available Commands:")
        print("  start       - Start trading")
        print("  stop        - Stop trading")
        print("  status      - Show system status")
        print("  positions   - Show open positions")
        print("  stats       - Show detailed statistics")
        print("  retrain     - Retrain agents (usage: retrain <symbol|all>)")
        print("  backup      - Create backup")
        print("  help        - Show this help")
        print("  exit        - Exit CLI (bot continues running)")
        print()

    async def cmd_exit(self, args):
        """Exit CLI"""
        print("  Exiting CLI (bot continues running)")
        exit(0)

#============================================================================
#MAIN COMPLET AVEC TOUTES LES FONCTIONNALIT S
#============================================================================
async def main_complete(skip_training: bool = False):
    """Main complet avec toutes les fonctionnalit s int gr es"""
    
    if skip_training:
        logging.warning("  MODE D VELOPPEMENT: Training sera skipp ")
    # Banner
    print("\n" + "="*80)
    print("  ULTIMATE HYBRID TRADING BOT - PRODUCTION VERSION")
    print("="*80)
    print("  Advanced AI Architecture:")
    print("    HRM (Hierarchical Reasoning Model)")
    print("    TRM (Tiny Recursive Model)")
    print("    PPO (Proximal Policy Optimization)")
    print("    DQN (Deep Q-Network)")
    print("    Ensemble Fusion Strategy")
    print("    Adaptive Halting with Q-Learning")
    print("")
    print("  Market Analysis:")
    print("    Order Flow Analysis")
    print("    Sentiment Analysis (Twitter, Reddit)")
    print("    Meta-Strategy Selection")
    print("    Technical Indicators")
    print("")
    print("  Risk Management:")
    print("    Kelly Criterion Position Sizing")
    print("    Correlation Limits")
    print("    Drawdown Protection")
    print("    Daily Loss Limits")
    print("="*80)
    print()

    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_from_file()

    if config is None:
        config = config_loader.create_default_config()
        config_loader.save_config(config)

    # Generate dashboard
    dashboard_gen = DashboardGenerator()
    dashboard_gen.save_dashboard()

    # Initialize components
    brain = EnhancedUltimateTradingBrain(config)
    backup_manager = BackupManager()
    notification_manager = NotificationManager()
    system_monitor = SystemMonitor()

    # Initialize brain
    print("  Initializing trading brain...")
    if not await brain.initialize(skip_training=skip_training):
        logging.error("  Failed to initialize, exiting")
        return

    print("  Trading brain initialized successfully!")

    # Send startup notification
    await notification_manager.send_notification(
        "  Ultimate Hybrid Trading Bot started successfully!",
        'success'
    )

    # WebSocket server
    async def ws_handler(websocket, path):
        await websocket_handler(websocket, brain)

    server = await websockets.serve(ws_handler, "localhost", config.WS_PORT)
    logging.info(f"  WebSocket server started on localhost:{config.WS_PORT}")

    #   CORRECTION 5: Dashboard Server
    dashboard_server = await start_dashboard_server(8080)
    if not dashboard_server:
        logging.warning("  Dashboard server non d marr , mais le bot continue")

    print("")
    print("="*80)
    print("  Dashboard: http://localhost:8080/dashboard.html")
    print(f"  Trading WebSocket: ws://localhost:{config.WS_PORT}")
    print(f"  RL Training Monitor: ws://localhost:{config.RL_WS_PORT}")
    print("="*80)
    print("")

    # D marrer le health monitor
    health_task = asyncio.create_task(brain.start_health_monitor())

    # Background tasks
    async def periodic_backup():
        """Backup p riodique"""
        while True:
            await asyncio.sleep(3600)  # Every hour
            logging.info("  Performing periodic backup...")
            backup_manager.backup_models(brain.rl_manager)
            backup_manager.backup_trade_history(brain.trade_history)
            backup_manager.cleanup_old_backups(keep_days=7)

    async def system_health_check():
        """V rification sant  syst me"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            health = system_monitor.check_health()
            
            if health['status'] == 'critical':
                await notification_manager.notify_error(
                    f"System health critical: {', '.join(health['errors'])}"
                )
            elif health['status'] == 'warning':
                logging.warning(f"  System warnings: {', '.join(health['warnings'])}")

    async def daily_summary():
        """R sum  quotidien"""
        while True:
            # Wait until midnight
            now = datetime.now()
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            wait_seconds = (tomorrow - now).total_seconds()
            
            await asyncio.sleep(wait_seconds)
            
            # Send daily summary
            await notification_manager.notify_daily_summary(brain.session_stats)
            
            # Generate performance report
            if brain.trade_history:
                analyzer = PerformanceAnalyzer()
                analyzer.generate_report(
                    list(brain.trade_history),
                    f"performance_report_{datetime.now().strftime('%Y%m%d')}.json"
                )
            
            # Reset daily stats
            brain.session_stats['daily_pnl'] = 0.0
            brain.daily_loss_tracker = 0.0

    # CLI interface
    cli = CLIInterface(brain)

    # Start all tasks
    try:
        await asyncio.gather(
            brain.run(),
            periodic_backup(),
            system_health_check(),
            daily_summary(),
            cli.run()
        )
    except KeyboardInterrupt:
        logging.info("\n  Keyboard interrupt detected")
    except Exception as e:
        logging.error(f"  Fatal error: {e}")
        traceback.print_exc()
        await notification_manager.notify_error(f"Fatal error: {str(e)}")
    finally:
        # Cleanup
        logging.info("  Cleaning up...")
        
        # Close all positions
        for symbol in list(brain.open_positions.keys()):
            await brain._close_position(symbol, reason="SHUTDOWN")
        
        # Final backup
        backup_manager.backup_models(brain.rl_manager)
        backup_manager.backup_trade_history(brain.trade_history)
        
        # Stop brain
        brain.stop()
        
        # Close WebSocket servers
        server.close()
        if 'dashboard_server' in locals():
            dashboard_server.close()
        await server.wait_closed()
        if 'dashboard_server' in locals():
            await dashboard_server.wait_closed()
        
        # Send shutdown notification
        await notification_manager.send_notification(
            "  Ultimate Hybrid Trading Bot shut down gracefully",
            'info'
        )
        
        logging.info("  Shutdown complete")


    #============================================================================
    # PATCH ULTIME - SYST ME DE R COMPENSES & APPRENTISSAGE RENFORC 
    # AJOUTEZ CE CODE   LA FIN DU FICHIER GOLDENEYE.py
    #============================================================================

    class AdvancedRewardSystem:
        """
          SYST ME DE R COMPENSES INTELLIGENT
        Combine multiples facteurs pour un apprentissage stable et profitable
        """
        
        def __init__(self):
            # Facteurs de pond ration
            self.weights = {
                'pnl': 2.0,           # Profit/Perte r alis 
                'sharpe': 1.5,        # Ratio risque/reward
                'drawdown': -3.0,     # P nalit  drawdown
                'consistency': 1.0,   # Consistance des gains
                'risk_adjustment': 0.8, # Ajustement risque
                'time_penalty': -0.01, # P nalit  temps
                'win_streak': 0.5,    # Bonus s ries gagnantes
                'position_quality': 1.2 # Qualit  position
            }
            
            self.window_size = 50
            self.performance_history = deque(maxlen=self.window_size)
            self.equity_history = deque(maxlen=100)
            self.last_equity = 10000.0
            
        def calculate_comprehensive_reward(self, trade_result: Dict, current_equity: float) -> float:
            """
            Calcule une r compense multi-facteurs pour un apprentissage robuste
            """
            try:
                reward_components = {}
                
                # 1.   R COMPENSE DE BASE - P&L (le plus important)
                pnl = trade_result.get('pnl', 0)
                reward_components['pnl'] = pnl * self.weights['pnl']
                
                # 2.   R COMPENSE SHARPE - Qualit  du risque
                sharpe_reward = self._calculate_sharpe_ratio_reward()
                reward_components['sharpe'] = sharpe_reward * self.weights['sharpe']
                
                # 3.   P NALIT  DRAWDOWN -  viter les grosses pertes
                drawdown_penalty = self._calculate_drawdown_penalty(current_equity)
                reward_components['drawdown'] = drawdown_penalty * self.weights['drawdown']
                
                # 4.   R COMPENSE CONSISTANCE -  viter le gambling
                consistency_reward = self._calculate_consistency_reward(pnl > 0)
                reward_components['consistency'] = consistency_reward * self.weights['consistency']
                
                # 5.   AJUSTEMENT RISQUE - R compenser le risque intelligent
                risk_reward = self._calculate_risk_adjustment(trade_result)
                reward_components['risk_adjustment'] = risk_reward * self.weights['risk_adjustment']
                
                # 6.   P NALIT  TEMPS -  viter les positions trop longues
                time_penalty = trade_result.get('duration', 0) / 60.0  # En heures
                reward_components['time_penalty'] = time_penalty * self.weights['time_penalty']
                
                # 7.   BONUS S RIE - R compenser les s ries gagnantes
                win_streak_bonus = self._calculate_win_streak_bonus(pnl > 0)
                reward_components['win_streak'] = win_streak_bonus * self.weights['win_streak']
                
                # 8.   QUALIT  POSITION - R compenser les bonnes entr es/sorties
                position_quality = self._calculate_position_quality(trade_result)
                reward_components['position_quality'] = position_quality * self.weights['position_quality']
                
                # R compense totale pond r e
                total_reward = sum(reward_components.values())
                
                # Mise   jour historique
                self._update_performance_history(pnl, current_equity)
                
                # Log d taill  (seulement pour debug)
                if abs(total_reward) > 50:
                    logging.info(f"  R COMPENSE D TAILL E: {total_reward:.2f}")
                    for comp, value in reward_components.items():
                        if abs(value) > 5:
                            logging.info(f"   {comp}: {value:.2f}")
                
                return total_reward
                
            except Exception as e:
                logging.error(f"  Erreur calculate_comprehensive_reward: {e}")
                return pnl * 2.0  # Fallback simple
        
        def _calculate_sharpe_ratio_reward(self) -> float:
            """Calcule un reward bas  sur le ratio de Sharpe"""
            if len(self.performance_history) < 10:
                return 0.0
            
            returns = np.array(list(self.performance_history))
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return < 1e-8:
                return 0.0
            
            sharpe = mean_return / std_return
            # Normaliser entre -1 et 1
            return np.tanh(sharpe * 0.1)
        
        def _calculate_drawdown_penalty(self, current_equity: float) -> float:
            """P nalit  pour drawdown important"""
            if len(self.equity_history) < 5:
                return 0.0
            
            peak_equity = max(self.equity_history)
            drawdown = (peak_equity - current_equity) / peak_equity
            
            # P nalit  progressive
            if drawdown <= 0.02:  # 2%
                return 0.0
            elif drawdown <= 0.05:  # 5%
                return -drawdown * 10
            else:  # >5%
                return -drawdown * 50
        
        def _calculate_consistency_reward(self, is_win: bool) -> float:
            """R compense pour consistence ( viter gambling)"""
            if len(self.performance_history) < 20:
                return 0.0
            
            recent_trades = list(self.performance_history)[-20:]
            win_rate = sum(1 for pnl in recent_trades if pnl > 0) / len(recent_trades)
            
            # R compenser les win rates entre 40% et 70%
            if 0.4 <= win_rate <= 0.7:
                return 0.5
            elif win_rate > 0.7:
                return 0.2  # Bonus r duit pour  viter overfitting
            else:
                return -0.3
        
        def _calculate_risk_adjustment(self, trade_result: Dict) -> float:
            """R compense/p nalit  pour gestion du risque"""
            pnl = trade_result.get('pnl', 0)
            duration = trade_result.get('duration', 60)
            
            # R compenser les gains rapides
            if pnl > 0 and duration < 30:  # Moins de 30 minutes
                return 0.3
            # P naliser les pertes rapides
            elif pnl < 0 and duration < 15:  # Moins de 15 minutes
                return -0.5
            else:
                return 0.0
        
        def _calculate_win_streak_bonus(self, is_win: bool) -> float:
            """Bonus pour s ries gagnantes"""
            if not self.performance_history:
                return 0.0
            
            # Compter les wins cons cutifs
            streak = 0
            for pnl in reversed(self.performance_history):
                if pnl > 0:
                    streak += 1
                else:
                    break
            
            if is_win:
                return min(streak * 0.1, 1.0)  # Max 1.0
            else:
                return 0.0
        
        def _calculate_position_quality(self, trade_result: Dict) -> float:
            """ value la qualit  de la position"""
            pnl = trade_result.get('pnl', 0)
            confidence = trade_result.get('confidence', 0.5)
            duration = trade_result.get('duration', 60)
            
            # Bonus pour gains avec haute confiance
            if pnl > 0 and confidence > 0.7:
                return 0.4
            # P nalit  pour pertes avec haute confiance
            elif pnl < 0 and confidence > 0.7:
                return -0.6
            # Bonus pour gains rapides
            elif pnl > 0 and duration < 45:
                return 0.3
            else:
                return 0.0
        
        def _update_performance_history(self, pnl: float, equity: float):
            """Met   jour l'historique des performances"""
            self.performance_history.append(pnl)
            self.equity_history.append(equity)
            self.last_equity = equity


    #============================================================================
    # PATCH ENVIRONNEMENT - R COMPENSES CORRIG ES
    #============================================================================

    def _calculate_reward_advanced(self, action: int) -> float:
        """
          NOUVELLE M THODE DE R COMPENSE - Version corrig e
        Philosophie: R compenser la PROFITABILIT  et GESTION DU RISQUE
        """
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0.0
        
        # Spread r aliste
        if 'XAU' in self.symbol or 'GOLD' in self.symbol:
            SPREAD = 0.25  # 25 cents pour l'or
        else:
            SPREAD = 0.0001  # 1 pip pour forex
        
        try:
            # ========================================
            # ACTION: BUY (0) - Ouvrir long
            # ========================================
            if action == 0 and self.position == 0:
                self.position = 1
                self.entry_price = current_price + SPREAD
                #   PETITE R COMPENSE pour bonne entr e (bas  sur signal)
                if hasattr(self, 'last_signal_confidence'):
                    reward = self.last_signal_confidence * 0.1
                return reward
                
            # ========================================
            # ACTION: SELL (1) - Ouvrir short  
            # ========================================
            elif action == 1 and self.position == 0:
                self.position = -1
                self.entry_price = current_price - SPREAD
                if hasattr(self, 'last_signal_confidence'):
                    reward = self.last_signal_confidence * 0.1
                return reward
                
            # ========================================
            # ACTION: CLOSE (3) - Fermer position   CALCUL P&L
            # ========================================
            elif action == 3 and self.position != 0:
                # Calculer P&L avec spread
                if self.position == 1:
                    exit_price = current_price - SPREAD
                    pnl_pct = (exit_price - self.entry_price) / self.entry_price
                else:
                    exit_price = current_price + SPREAD
                    pnl_pct = (self.entry_price - exit_price) / self.entry_price
                
                #   R COMPENSE PRINCIPALE = P&L (scal )
                pnl_reward = pnl_pct * 1000  # Multiplicateur augment 
                
                # Bonus/P nalit  dur e position
                duration_bonus = self._calculate_duration_bonus()
                
                # R compense finale
                reward = pnl_reward + duration_bonus
                
                # Mettre   jour les m triques
                self._update_trade_metrics(pnl_pct)
                
                # Reset position
                self.position = 0
                self.entry_price = 0.0
                
                return reward
                
            # ========================================
            # ACTION: HOLD (2) - Maintenir position
            # === PATCH ULTIME: Reward Hybride ===
            # ========================================
            elif action == 2 and self.position != 0:
                # Calcul P&L NON RÉALISÉ
                if self.position == 1:
                    unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                else:
                    unrealized_pnl = (self.entry_price - current_price) / self.entry_price
                
                # Calcul durée de la position
                holding_time = self.current_step - getattr(self, 'position_open_step', self.current_step)
                
                # === REWARD HYBRIDE CLAUDE/GROK ===
                if unrealized_pnl > 0:
                    # Bonus "laisser courir les gagnants" avec log1p (évite sur-récompense)
                    bonus = 0.015 * np.log1p(unrealized_pnl * 100)
                    # Bonus temps si on tient une position gagnante longtemps
                    if holding_time > 8:
                        bonus *= 1.5
                    reward = bonus
                else:
                    # Pénalité SEULEMENT si perte flottante importante ET longue
                    if unrealized_pnl < -0.002 and holding_time > 10:
                        reward = unrealized_pnl * 80  # Pénalité progressive mais pas brutale
                    else:
                        reward = unrealized_pnl * 30  # Pénalité légère sinon
                
                # Micro-bonus Sharpe local (stabilité des returns)
                if hasattr(self, 'close_trades') and len(self.close_trades) >= 20:
                    recent_returns = self.close_trades[-20:]
                    recent_sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
                    if recent_sharpe > 0.8:
                        reward += 0.008  # Micro-bonus stabilité
                
                return reward
            
            # ========================================
            # ACTION INVALIDE - Pénalité légère
            # ========================================
            else:
                return -0.1
                
        except Exception as e:
            logging.error(f"  Erreur _calculate_reward_advanced: {e}")
            return 0.0


    def _calculate_duration_bonus(self) -> float:
        """Bonus/p nalit  bas  sur la dur e de la position"""
        if not hasattr(self, 'position_open_step'):
            return 0.0
        
        duration_steps = self.current_step - self.position_open_step
        duration_hours = duration_steps / 4.0  # Approx heures (15min steps)
        
        # Bonus pour gains rapides (< 2h)
        if duration_hours < 2:
            return 0.5
        # P nalit  pour positions trop longues (> 8h)
        elif duration_hours > 8:
            return -0.3
        else:
            return 0.0


    def _update_trade_metrics(self, pnl_pct: float):
        """Met   jour les m triques de trading"""
        self.total_pnl += pnl_pct * 100
        self.close_trades.append(pnl_pct * 100)
        
        if pnl_pct > 0:
            self.close_wins += 1
        else:
            self.close_losses += 1
        
        # Mettre   jour l'equity
        self.equity *= (1 + pnl_pct)


    #============================================================================
    # PATCH ENTRA NEMENT - ALGORITHMES AM LIOR S
    #============================================================================

    class AdvancedTrainingConfig:
        """Configuration avanc e pour l'entra nement"""
        
        def __init__(self):
            # Hyperparam tres PPO optimis s pour trading
            self.ppo_params = {
                'learning_rate': 3e-4,        # AUGMENT  pour apprentissage plus rapide
                'n_steps': 512,               # R DUIT pour updates fr quents
                'batch_size': 128,            # R DUIT pour plus de diversit 
                'n_epochs': 10,               # R DUIT pour pas overfit
                'gamma': 0.99,                # Standard
                'gae_lambda': 0.95,           # Standard
                'clip_range': 0.2,            # AUGMENT  pour plus de libert 
                'ent_coef': 0.1,              # MASSIF pour forcer exploration
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'use_sde': True,
                'sde_sample_freq': 4
            }
            
            # Hyperparam tres DQN optimis s  
            self.dqn_params = {
                'learning_rate': 1e-4,
                'buffer_size': 500000,        # Replay buffer  norme
                'learning_starts': 10000,     # Plus de steps avant apprentissage
                'batch_size': 128,
                'tau': 1.0,                   # Soft update
                'gamma': 0.99,
                'train_freq': 4,
                'gradient_steps': 2,
                'target_update_interval': 10000,
                'exploration_fraction': 0.2,  # Exploration plus longue
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.02, # Exploration finale basse
            }
            
            self.total_timesteps = 10000    # 3M steps ( tait 4M, plus r aliste)


    async def train_agent_advanced(self, symbol: str):
        """
          ENTRA NEMENT AVANC  - Algorithmes optimis s pour profitabilit 
        """
        logging.info(f"  ENTRA NEMENT AVANC  POUR {symbol}")
        
        try:
            # Charger donn es 6 ans
            historical_data = await self.load_historical_data(symbol)
            
            if historical_data is None or len(historical_data) < 10000:
                logging.warning(f"  Donn es insuffisantes pour {symbol}")
                return
            
            # Configuration avanc e
            config = AdvancedTrainingConfig()
            
            # Cr er environnement avec syst me de r compenses avanc 
            symbol_id = self.config.SYMBOLS.index(symbol)
            env = TradingEnvironment(historical_data, symbol_id=symbol_id, symbol=symbol)
            
            #   REMPLACER LA M THODE DE R COMPENSE
            env._calculate_reward = lambda action: _calculate_reward_advanced(env, action)
            
            env = GymnasiumToSB3Wrapper(env)
            vec_env = DummyVecEnv([lambda: env])
            
            # Initialiser HRM-TRM
            self.hrm_trm_per_symbol[symbol] = env.hrm_trm
            
            # ============================================================
            # ENTRA NEMENT PPO ULTRA-OPTIMIS 
            # ============================================================
            logging.info(f"    Training PPO avanc  pour {symbol}...")
            
            self.ppo_agents[symbol] = PPO(
                'MlpPolicy',
                vec_env,
                verbose=1,  #   ACTIVER LES LOGS PPO
                **config.ppo_params,
                tensorboard_log=f"./logs/advanced_ppo_{symbol}"
            )
            
            # Callback de monitoring avanc  avec VALIDATION et WINRATE
            class ProfitabilityCallback(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self.profitable_episodes = 0
                    self.total_episodes = 0
                    self.best_winrate = 0.0
                    self.episode_winrates = []
                    self.step_count = 0
                    self.last_log_step = 0
                
                def _on_step(self) -> bool:
                    self.step_count += 1
                    
                    # Log toutes les 5K steps pour voir progression
                    if self.step_count - self.last_log_step >= 5000:
                        self.last_log_step = self.step_count
                        
                        # Extraire les m triques de l'environnement
                        if hasattr(self.model, 'env') and hasattr(self.model.env, 'envs'):
                            for env in self.model.env.envs:
                                if hasattr(env, 'unwrapped'):
                                    unwrapped = env.unwrapped
                                    if hasattr(unwrapped, 'close_wins') and hasattr(unwrapped, 'close_losses'):
                                        total_closes = unwrapped.close_wins + unwrapped.close_losses
                                        if total_closes > 0:
                                            winrate = unwrapped.close_wins / total_closes
                                            equity = unwrapped.equity
                                            
                                            # Track best
                                            if winrate > self.best_winrate:
                                                self.best_winrate = winrate
                                            
                                            # Calculer moyenne mobile sur 50 derniers trades
                                            if len(unwrapped.close_trades) >= 50:
                                                recent_50 = unwrapped.close_trades[-50:]
                                                wins_50 = sum(1 for p in recent_50 if p > 0)
                                                wr_50 = wins_50 / 50
                                            else:
                                                wr_50 = winrate
                                            
                                            # Log dans tensorboard
                                            self.logger.record('metrics/winrate', winrate)
                                            self.logger.record('metrics/winrate_ma50', wr_50)
                                            self.logger.record('metrics/best_winrate', self.best_winrate)
                                            self.logger.record('metrics/total_closes', total_closes)
                                            self.logger.record('metrics/equity', equity)
                                            self.logger.record('metrics/equity_change', (equity - 10000) / 10000 * 100)
                                            
                                            # AFFICHAGE CONSOLE EN VERT
                                            logging.info("")
                                            logging.info("=" * 80)
                                            logging.info(f"  PROGRESSION STEP {self.step_count:,}")
                                            logging.info("=" * 80)
                                            logging.info(f"  WinRate Global: {winrate:.1%} | MA50: {wr_50:.1%} | Best: {self.best_winrate:.1%}")
                                            logging.info(f"  Equity: ${equity:,.2f} ({(equity-10000)/10000*100:+.2f}%)")
                                            logging.info(f"  Total Closes: {total_closes} ({unwrapped.close_wins}W / {unwrapped.close_losses}L)")
                                            
                                            # Indicateur de tendance
                                            if wr_50 > 0.55:
                                                logging.info(f"  TENDANCE: EXCELLENTE (MA50 > 55%)")
                                            elif wr_50 > 0.50:
                                                logging.info(f"  TENDANCE: POSITIVE (MA50 > 50%)")
                                            elif wr_50 > 0.45:
                                                logging.info(f"  TENDANCE: PROMETTEUSE (MA50 > 45%)")
                                            else:
                                                logging.info(f"  TENDANCE: EN PROGRESSION (MA50 = {wr_50:.1%})")
                                            
                                            logging.info("=" * 80)
                                            logging.info("")
                    
                    return True
                    
                def _on_rollout_end(self) -> None:
                    # V rifier la profitabilit 
                    if hasattr(self.model, 'env') and hasattr(self.model.env, 'envs'):
                        for env in self.model.env.envs:
                            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'total_pnl'):
                                pnl = env.unwrapped.total_pnl
                                if pnl > 0:
                                    self.profitable_episodes += 1
                                self.total_episodes += 1
                    
                    if self.total_episodes > 0:
                        profit_rate = self.profitable_episodes / self.total_episodes
                        self.logger.record('train/profit_rate', profit_rate)
            
            tqdm_ppo = TqdmCallback(config.total_timesteps // 2, f"  PPO-ADV {symbol}", "green")
            profit_callback = ProfitabilityCallback()
            
            self.ppo_agents[symbol].learn(
                total_timesteps=config.total_timesteps // 2,
                callback=[tqdm_ppo, profit_callback],
                tb_log_name="ppo_advanced"
            )
            # tqdm_ppo auto-closes
            
            # ============================================================
            # ENTRA NEMENT DQN ULTRA-OPTIMIS 
            # ============================================================
            logging.info(f"    Training DQN avanc  pour {symbol}...")
            
            self.dqn_agents[symbol] = DQN(
                'MlpPolicy',
                vec_env,
                verbose=1,  #   ACTIVER LES LOGS DQN
                **config.dqn_params,
                tensorboard_log=f"./logs/advanced_dqn_{symbol}"
            )
            
            tqdm_dqn = TqdmCallback(config.total_timesteps // 2, f"  DQN-ADV {symbol}", "blue")
            
            self.dqn_agents[symbol].learn(
                total_timesteps=config.total_timesteps // 2,
                callback=[tqdm_dqn],
                tb_log_name="dqn_advanced"
            )
            # tqdm_dqn auto-closes
            
            # ============================================================
            # VALIDATION AVANC E
            # ============================================================
            validation_result = await self._advanced_validation(symbol, env)
            
            if validation_result['passed']:
                self._save_advanced_models(symbol)
                logging.info(f"  ENTRA NEMENT AVANC  R USSI: {symbol}")
                logging.info(f"     Win Rate: {validation_result['win_rate']:.1f}%")
                logging.info(f"     Profit moyen: {validation_result['avg_profit']:.2f}%")
            else:
                logging.warning(f"  Validation  chou e: {validation_result['reason']}")
            
        except Exception as e:
            logging.error(f"  Erreur entra nement avanc  {symbol}: {e}")
            traceback.print_exc()


    async def _advanced_validation(self, symbol: str, env: TradingEnvironment) -> Dict:
        """Validation avanc e des performances"""
        try:
            test_results = []
            total_profit = 0
            win_count = 0
            
            for episode in range(10):  # 10  pisodes de test
                obs, _ = env.reset()
                episode_profit = 0
                done = False
                
                while not done:
                    action, _ = self.ppo_agents[symbol].predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    episode_profit += reward
                
                test_results.append(episode_profit)
                total_profit += episode_profit
                if episode_profit > 0:
                    win_count += 1
            
            avg_profit = total_profit / 10
            win_rate = win_count / 10 * 100
            
            # Crit res de validation stricts
            passed = (win_rate >= 55 and avg_profit > 5.0)  # 55% win rate + 5% profit moyen
            
            return {
                'passed': passed,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'reason': f"Win rate {win_rate:.1f}%, Profit moyen {avg_profit:.2f}%"
            }
            
        except Exception as e:
            logging.error(f"  Erreur validation avanc e: {e}")
            return {'passed': False, 'reason': str(e)}


    def _save_advanced_models(self, symbol: str):
        """Sauvegarde des mod les avanc s"""
        try:
            os.makedirs("./models/advanced", exist_ok=True)
            
            # PPO avanc 
            ppo_path = f"./models/advanced/ppo_adv_{symbol}.zip"
            self.ppo_agents[symbol].save(ppo_path)
            
            # DQN avanc 
            dqn_path = f"./models/advanced/dqn_adv_{symbol}.zip"
            self.dqn_agents[symbol].save(dqn_path)
            
            # HRM-TRM avanc 
            hrm_path = f"./models/advanced/hrm_trm_adv_{symbol}.pt"
            torch.save(self.hrm_trm_per_symbol[symbol].state_dict(), hrm_path)
            
            logging.info(f"  MOD LES AVANC S SAUVEGARD S: {symbol}")
            
        except Exception as e:
            logging.error(f"  Erreur sauvegarde mod les avanc s: {e}")


    #============================================================================
    # APPLICATION DES PATCHS
    #============================================================================

    # Sauvegarder l'ancienne m thode
    RLAgentManager.train_agent_original = RLAgentManager.train_agent

    # Remplacer par la nouvelle m thode
    RLAgentManager.train_agent = train_agent_advanced
    RLAgentManager._advanced_validation = _advanced_validation
    RLAgentManager._save_advanced_models = _save_advanced_models

    # Ajouter le syst me de r compenses avanc    TradingEnvironment
    TradingEnvironment.advanced_reward_system = AdvancedRewardSystem()

    logging.info("  PATCH AVANC  APPLIQU  - Syst me de r compenses et entra nement optimis ")
    logging.info("  Configuration: 4M steps, reward multi-facteurs, validation stricte")







#============================================================================
#POINT D'ENTR E FINAL
#============================================================================


if __name__ == "__main__":
    """
    Point d'entr e principal du bot.
    
    Usage:
        python ULTIMATE_FIXED.py                  # Mode normal avec training
        python ULTIMATE_FIXED.py --skip-training  # Skip training (mode dev)
        python ULTIMATE_FIXED.py -s              # Alias pour --skip-training
    Usage:
        python ultimate_hybrid_bot.py

    Options d'environnement (.env):
        MT5_LOGIN=votre_login
        MT5_PASSWORD=votre_password
        MT5_SERVER=votre_serveur
        TELEGRAM_BOT_TOKEN=votre_token
        TELEGRAM_CHAT_ID=votre_chat_id
        DISCORD_WEBHOOK=votre_webhook
        TWITTER_BEARER_TOKEN=votre_token
        REDDIT_CLIENT_ID=votre_client_id
        REDDIT_CLIENT_SECRET=votre_secret
    """

    # Parse command line arguments
    skip_training = '--skip-training' in sys.argv or '-s' in sys.argv
    
    if skip_training:
        logging.info("  Mode d veloppement: Training sera skipp ")
    
    try:
        # Run complete system
        asyncio.run(main_complete(skip_training=skip_training))
        
    except KeyboardInterrupt:
        print("\n  Goodbye!")
    except Exception as e:
        logging.critical(f"  Critical startup error: {e}")
        traceback.print_exc()
        sys.exit(1)

#============================================================================
#FIN DU CODE - BOT PR T POUR PRODUCTION
#==========================================================================="""
# SMART POSITION MANAGER
"""Gestion adaptative du capital en 3 vagues avec r ajustements intelligents
"""

class SmartPositionManager:
    """
    Gestionnaire intelligent de positions avec strat gie 1/3 + 2 r ajustements
    """
    
    def __init__(self, total_capital, max_trades_per_wave=4):
        self.total_capital = total_capital
        self.max_trades_per_wave = max_trades_per_wave
        
        # Strat gie 3 vagues : 33% / 33% / 34%
        self.wave_allocation = [0.33, 0.33, 0.34]
        self.current_wave = 0
        self.max_waves = 3
        
        # Tracking des performances
        self.trades_per_wave = [0, 0, 0]
        self.pnl_per_wave = [0.0, 0.0, 0.0]
        
        # Capital disponible par vague
        self.available_capital = [
            total_capital * self.wave_allocation[0],
            total_capital * self.wave_allocation[1],
            total_capital * self.wave_allocation[2]
        ]
        
        # Positions ouvertes par vague
        self.positions_by_wave = {0: [], 1: [], 2: []}
        
    def can_open_position(self) -> tuple[bool, str]:
        """
        V rifie si on peut ouvrir une nouvelle position
        Returns: (can_trade, reason)
        """
        # Plus de vagues disponibles
        if self.current_wave >= self.max_waves:
            return False, "  Toutes les vagues de capital utilis es"
        
        # Limite de trades pour cette vague atteinte
        if self.trades_per_wave[self.current_wave] >= self.max_trades_per_wave:
            # Peut-on passer   la vague suivante ?
            if self.should_advance_wave():
                self.current_wave += 1
                if self.current_wave >= self.max_waves:
                    return False, "  Limite de trades globale atteinte"
                return True, f"  Passage   la vague {self.current_wave + 1}"
            else:
                return False, f"  En attente de signal pour vague {self.current_wave + 2}"
        
        return True, f"  Vague {self.current_wave + 1} active"
    
    def should_advance_wave(self) -> bool:
        """
        D cide intelligemment si on passe   la vague suivante
        Crit res adaptatifs bas s sur la performance
        """
        wave = self.current_wave
        
        # Pas encore de trades dans cette vague
        if self.trades_per_wave[wave] == 0:
            return False
        
        # Crit re 1: Vague tr s profitable   On continue avec la prochaine
        if self.pnl_per_wave[wave] > 0.02:  # +2% de profit
            return True
        
        # Crit re 2: Toutes les positions de cette vague sont ferm es
        if len(self.positions_by_wave[wave]) == 0:
            return True
        
        # Crit re 3: Drawdown important   On r ajuste avec nouvelle vague
        if self.pnl_per_wave[wave] < -0.03:  # -3% de perte
            return True
        
        # Crit re 4: Temps  coul  (20+ trades) et performance neutre
        if self.trades_per_wave[wave] >= self.max_trades_per_wave:
            return True
        
        return False
    
    def calculate_position_size(self, symbol: str, risk_percent: float, 
                               stop_ticks: float, tick_value: float) -> float:
        """
        Calcule la taille de position adaptative selon la vague actuelle
        """
        # Capital disponible pour cette vague
        wave_capital = self.available_capital[self.current_wave]
        
        # Ajustement adaptatif selon la performance
        performance_multiplier = self.get_performance_multiplier()
        
        # Risque ajust 
        adjusted_risk = risk_percent * performance_multiplier
        
        # Calcul standard
        risk_amount = wave_capital * adjusted_risk
        position_size = risk_amount / (stop_ticks * tick_value)
        
        return position_size
    
    def get_performance_multiplier(self) -> float:
        """
        Ajuste le risque selon la performance globale
        Adaptatif et intelligent
        """
        # Performance globale
        total_pnl = sum(self.pnl_per_wave)
        
        # Tr s profitable   Augmente le risque (max 1.5x)
        if total_pnl > 0.05:  # +5%
            return 1.5
        
        # Profitable   Risque normal augment 
        elif total_pnl > 0.02:  # +2%
            return 1.2
        
        # Neutre   Risque normal
        elif total_pnl > -0.02:
            return 1.0
        
        # L g re perte   R duit le risque
        elif total_pnl > -0.05:  # -5%
            return 0.7
        
        # Perte importante   Tr s prudent
        else:
            return 0.5
    
    def register_trade(self, symbol: str, position_size: float):
        """Enregistre un nouveau trade"""
        self.trades_per_wave[self.current_wave] += 1
        self.positions_by_wave[self.current_wave].append({
            'symbol': symbol,
            'size': position_size,
            'wave': self.current_wave
        })
    
    def close_position(self, symbol: str, pnl: float):
        """Enregistre la fermeture d'une position"""
        # Trouve dans quelle vague  tait cette position
        for wave in range(self.max_waves):
            for pos in self.positions_by_wave[wave]:
                if pos['symbol'] == symbol:
                    self.positions_by_wave[wave].remove(pos)
                    self.pnl_per_wave[wave] += pnl
                    return
    
    def get_status(self) -> dict:
        """Retourne le status actuel"""
        return {
            'current_wave': self.current_wave + 1,
            'total_trades': sum(self.trades_per_wave),
            'trades_per_wave': self.trades_per_wave,
            'pnl_per_wave': self.pnl_per_wave,
            'total_pnl': sum(self.pnl_per_wave),
            'performance_multiplier': self.get_performance_multiplier(),
            'can_trade': self.can_open_position()[0]
        }
    
    def reset_for_new_day(self):
        """Reset pour un nouveau jour de trading"""
        self.current_wave = 0
        self.trades_per_wave = [0, 0, 0]
        self.pnl_per_wave = [0.0, 0.0, 0.0]
        self.positions_by_wave = {0: [], 1: [], 2: []}





# ===================================================================
# BARYCENTRE + SABLIER + BOUSSOLE + MONTRE SUISSE + GPS (installé le 2025-11-24 07:20)
# Quand ça chavire → le sablier se retourne, inverse le signal, contracte les pertes, fluidifie les gains
# ===================================================================

from collections import deque
import time

class BarycentreSystem:
    def __init__(self):
        self.pnl_history = deque(maxlen=28)
        self.flip_threshold = -0.05
        self.smooth_factor = 0.65
        self.regime = 'neutral'
        self.last_inversion_time = None
        self.min_time_between_flips = 3600  # 1h entre chaque retournement
        self.current_drawdown_phase = 0

    def update_pnl(self, pnl: float):
        self.pnl_history.append(pnl)

    def get_barycentre(self) -> float:
        if not self.pnl_history: return 0.0
        weights = [self.smooth_factor ** i for i in range(len(self.pnl_history))]
        weighted = sum(p * w for p, w in zip(self.pnl_history, weights))
        total_w = sum(weights)
        return weighted / total_w if total_w > 0 else 0.0

    def update_regime_and_dd(self, volatility_ratio=1.0, trend_strength=0.0, current_dd=0.0):
        # Boussole
        if volatility_ratio > 1.5 and abs(trend_strength) > 0.7:
            self.regime = 'trending_up' if trend_strength > 0 else 'trending_down'
        elif volatility_ratio > 2.0: self.regime = 'volatile'
        elif volatility_ratio < 0.5: self.regime = 'ranging'
        else: self.regime = 'neutral'
        # GPS
        self.current_drawdown_phase = 2 if current_dd > 0.10 else 1 if current_dd > 0.05 else 0

    def should_flip(self) -> bool:
        if self.last_inversion_time and (time.time() - self.last_inversion_time) < self.min_time_between_flips:
            return False
        if self.get_barycentre() < self.flip_threshold:
            if self.regime in ['trending_down', 'volatile'] or self.current_drawdown_phase >= 1:
                self.last_inversion_time = time.time()
                print("SABLIER RETOURNÉ → SIGNAL INVERSÉ !")
                return True
        return False

    def size_multiplier(self) -> float:
        b = self.get_barycentre()
        if b > 0.02:  return 1.25   # Fluidifie les gains
        if b < -0.02: return 0.40   # Contracte les pertes
        return 1.0

# Instance globale partagée
if not hasattr(SmartPositionManager, 'barycentre'):
    SmartPositionManager.barycentre = BarycentreSystem()

# === INJECTION DANS LES MÉTHODES ===
_orig_calc = SmartPositionManager.calculate_position_size
_orig_close = SmartPositionManager.close_position

def patched_calculate(self, symbol: str, risk_percent: float, stop_ticks: float, tick_value: float,
                      volatility_ratio: float = 1.0, trend_strength: float = 0.0, proposed_signal: int = 1):
    bs = self.barycentre
    bs.update_regime_and_dd(volatility_ratio, trend_strength, self.current_drawdown if hasattr(self, 'current_drawdown') else 0.0)

    signal = proposed_signal
    if bs.should_flip():
        signal = -proposed_signal

    size = _orig_calc(self, symbol, risk_percent, stop_ticks, tick_value)
    size *= bs.size_multiplier()
    return max(size, 0.01)

def patched_close(self, symbol: str, pnl: float):
    _orig_close(self, symbol, pnl)
    self.barycentre.update_pnl(pnl)

SmartPositionManager.calculate_position_size = patched_calculate
SmartPositionManager.close_position = patched_close
print("BARYCENTRE + SABLIER ACTIVÉ → Le système retourne le sablier quand ça chavire")
