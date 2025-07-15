from abc import abstractmethod

from torch import Tensor


class Schedule:
    def __init__(self):
        pass

    @abstractmethod
    def beta_schedule(self, total_steps: int) -> Tensor:
        """
        Generate a beta schedule for the diffusion process.

        Args:
            total_steps (int): Total number of diffusion steps.

        Returns:
            Tensor: A tensor containing the beta values for each step.
        """
        pass
