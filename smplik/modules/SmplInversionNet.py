import torch

from smplik.modules.kernel_regression import GaussianKernelRegression


class SmplInversionBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.male = None
        self.female = None
        self.neutral = None

    def forward(self, x: torch.Tensor, gender: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        male_betas = self.male(x, *args, **kwargs)
        female_betas = self.female(x, *args, **kwargs)
        neutral_betas = self.neutral(x, *args, **kwargs)

        is_male = gender == 0
        is_female = gender == 1
        betas = torch.where(is_male, male_betas, torch.where(is_female, female_betas, neutral_betas))

        return betas


class SmplInversionGaussian(SmplInversionBase):
    def __init__(self, male_batch, female_batch, neutral_batch):
        super().__init__()
        self.male = GaussianKernelRegression(source=male_batch["output"], target=male_batch["features"])
        self.female = GaussianKernelRegression(source=female_batch["output"], target=female_batch["features"])
        self.neutral = GaussianKernelRegression(source=neutral_batch["output"], target=neutral_batch["features"])
