class Light(gym.Env):
    def reset(self):
        inital_state = super.reset()
        inital_state.randomization_settings["light_pos"] = [0.0, 3.0, 0.0, 1.0]
        return inital_state