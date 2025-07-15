import click

from libs.config import Paths
from libs.models.simple_unet import SimpleUNet
from libs.sampler import SamplerConfig, Sampler
from libs.schedules.cosine_beta import CosineBeta


@click.command()
@click.option("--num-steps", default=1000, help="Number of diffusion steps")
@click.option("--batch-size", default=128, help="Batch size for sampling")
def main(num_steps: int, batch_size: int):
    """
    Main function to run the sample script.

    Args:
        num_steps (int): Number of diffusion steps for the sampling process.
        batch_size (int): Batch size for sampling the model.
    """

    config = SamplerConfig()

    config.paths = Paths()

    config.num_steps = num_steps
    config.batch_size = batch_size

    config.model = SimpleUNet(base_channels=config.base_channels).to(config.device)
    config.schedule = CosineBeta()

    sampler = Sampler(config)

    sampler.sample()

if __name__ == "__main__":
    main()