import json
import numpy as np
import websocket
import gymnasium as gym

from gymnasium import spaces


class MinecraftDragonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.ws = websocket.create_connection("ws://localhost:8765", timeout=10)

        # 12 discrete actions, defined in bot.js.
        self.action_space = spaces.Discrete(14)

        # dx, dy, dz, distance, yawSin, yawCos,
        # playerHealth, onGround, dragonHealth,
        # dragonVisible, perched
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )

        self.previous_state = None
        self.steps = 0
        self.max_episode_steps = 3000
        
        self.ep_damage_reward = 0.0
        self.ep_aim_reward = 0.0
        self.ep_health_penalty = 0.0
        self.ep_center_reward = 0.0
        self.ep_courage_reward = 0.0
        self.ep_distance_reward = 0.0
        self.ep_void_penalty = 0.0

    def _request(self, message):
        self.ws.send(json.dumps(message))
        response = json.loads(self.ws.recv())

        if "error" in response:
            raise RuntimeError(response["error"])

        return response["state"]

    def _observation(self, state):
        dragon_health = state["dragonHealth"]

        values = [
            state["dx"],
            state["dy"],
            state["dz"],
            state["distanceNorm"],
            state["yawSin"],
            state["yawCos"],
            state["playerHealth"],
            state["onGround"],
            (dragon_health / 200.0) if dragon_health is not None else 0.0,
            state["dragonVisible"],
            state["perched"],
            state["yawErrorNorm"],
            state["pitchErrorNorm"],
            state["aimError"],
            state["centerDistanceNorm"],
            state["portalYawErrorNorm"],
        ]

        return np.clip(
            np.asarray(values, dtype=np.float32),
            -1.0,
            1.0,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        
        self.ep_damage_reward = 0.0
        self.ep_aim_reward = 0.0
        self.ep_health_penalty = 0.0
        self.ep_center_reward = 0.0
        self.ep_courage_reward = 0.0
        self.ep_distance_reward = 0.0
        self.ep_void_penalty = 0.0
        
        state = self._request({"type": "reset"})

        if state is None:
            raise RuntimeError(
                "Minecraft returned no observation. "
                "Wait for the bot to spawn before training."
            )

        self.previous_state = state
        return self._observation(state), {}

    def step(self, action):
        old = self.previous_state
        new = self._request(
            {
                "type": "step",
                "action": int(action),
            }
        )

        reward_distance = 0.0
        reward_damage = 0.0
        reward_health = 0.0
        reward_time = -0.05
        reward_victory = 0.0
        reward_aim = 0.0
        reward_center = 0.0
        reward_courage = 0.0
        reward_void = 0.0

        # Encourage useful positioning during a perch.
        if old["perched"] and new["perched"]:
            reward_distance += 2.0 * (old["distance"] - new["distance"])

        # Reward actual dragon damage when health is exposed.
        old_dh = old["dragonHealth"]
        new_dh = new["dragonHealth"]

        if old_dh is not None and new_dh is not None:
            reward_damage += 0.5 * (old_dh - new_dh)

        # Penalize losing player health.
        reward_health -= 1.0 * (old["playerHealth"] - new["playerHealth"])

        if old["dragonVisible"] and new["dragonVisible"]:
            reward_aim = 2.0 * (old["aimError"] - new["aimError"])

            if (
                action in [6, 7, 8, 11]
                and new["aimError"] < 0.2
                and new["distance"] < 10
            ):
                reward_courage = 0.5

        # if new["centerDistance"] > 50:
        #     reward_center = -0.1 * (new["centerDistance"] - 50)
        
        reward_center = 0.5 * (old["centerDistance"] - new["centerDistance"])
        
        fell_in_void = new["dy"] < -20
        if fell_in_void :
            reward_void = -50.0

        reward = (
            reward_distance
            + reward_damage
            + reward_health
            + reward_time
            + reward_aim
            + reward_center
            + reward_courage
            + reward_void
        )

        self.steps += 1
        self.previous_state = new
        
        self.ep_damage_reward += reward_damage
        self.ep_aim_reward += reward_aim
        self.ep_health_penalty += reward_health
        self.ep_center_reward += reward_center
        self.ep_courage_reward += reward_courage
        self.ep_distance_reward += reward_distance
        self.ep_void_penalty += reward_void

        # Temporary workaround for not spawning the dragon
        dragon_defeated = new_dh is not None and new_dh <= 25.0
        bot_died = new["playerHealth"] <= 0

        terminated = bot_died or dragon_defeated or fell_in_void

        truncated = self.steps >= self.max_episode_steps

        if dragon_defeated:
            reward_victory = 500.0
            reward += reward_victory

        info = {
            "player_health": new["playerHealth"],
            "dragon_health": new["dragonHealth"],
            "distance": new["distance"],
            "perched": new["perched"],
            "aim_error": new["aimError"],
            "yaw_error": new["yawErrorNorm"],
            "pitch_error": new["pitchErrorNorm"],
            
            # Exported cumulative rewards for TensorBoard
            "reward_damage": self.ep_damage_reward,
            "reward_health": self.ep_health_penalty,
            "reward_aim": self.ep_aim_reward,
            "reward_center_total": self.ep_center_reward,
            "reward_courage_total": self.ep_courage_reward,
            "reward_distance_total": self.ep_distance_reward,
            "reward_void_total": self.ep_void_penalty,
            
            # One-time event rewards
            "reward_victory": reward_victory, 
            "reward_time_step": reward_time, 
        }

        return (
            self._observation(new),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.ws.close()
