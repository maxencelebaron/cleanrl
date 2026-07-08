import gymnasium as gym
import numpy as np


class RenderFix(gym.Wrapper):
    """Upscale MinAtar frames and cast to uint8 for RecordVideo compatibility.
    """

    SCALE = 16

    def render(self):
        frame = self.env.render()
        if frame is None:
            return frame
        frame = (frame * 255).astype(np.uint8)
        return frame.repeat(self.SCALE, axis=0).repeat(self.SCALE, axis=1)
