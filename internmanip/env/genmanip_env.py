from internmanip.env.base import EnvWrapper
from internmanip.configs.env.env_cfg import EnvCfg
from internmanip.benchmarks.genmanip.create_env import create_env


class GenmanipEnv(EnvWrapper):
    def __init__(self, config: EnvCfg):
        super().__init__(config)
        print(f'genmanip env settings: {self.config.env_settings}')

        self._config, self._env = create_env(self.config.env_settings)

    def warm_up(self, steps):
        self._env.warm_up(steps=steps)

    def reset(self, env_reset_ids=[]):
        if env_reset_ids:
            return self._env.reset(env_reset_ids)
        else:
            return self._env.reset()

    def step(self, all_env_action):
        return self._env.step(action=[{'robot':action} for action in all_env_action])

    def close(self):
        self._env.close()

    def get_obs(self):
        return self._env.get_observations()

    def get_info(self):
        pass

    @property
    def simulation_app(self):
        return self._env.simulation_app

    @property
    def runner(self):
        return self._env.runner
