from typing import List
import torch


class SumStructureKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kernel: torch.Tensor,
        structure_slices: List[slice],
        training_slices: List[slice],
    ):
        output = torch.zeros(
            (len(structure_slices), len(training_slices)),
            device=kernel.device,
        )
        for i, structure_i in enumerate(structure_slices):
            for j, structure_j in enumerate(training_slices):
                output[i, j] = kernel[structure_i, structure_j].sum()

        ctx.structure_slices = structure_slices
        ctx.training_slices = training_slices
        ctx.save_for_backward(kernel)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        kernel = ctx.saved_tensors[0]

        if kernel.requires_grad:
            grad_input = torch.zeros_like(kernel)
            for i, structure_i in enumerate(ctx.structure_slices):
                for j, structure_j in enumerate(ctx.training_slices):
                    grad_input[structure_i, structure_j] = grad_output[i, j]

        return grad_input, None, None


class CosineKernel(torch.nn.Module):
    def __init__(self, support_points, detach_support_points, zeta=2):
        super().__init__()

        assert torch.allclose(
            torch.linalg.norm(support_points, dim=1),
            torch.tensor(1.0, device=support_points.device),
        )

        self.detach_support_points = detach_support_points
        if self.detach_support_points:
            support_points = support_points.detach()

        self.register_buffer("support_points", support_points)
        self.zeta = zeta

    def forward(self, power_spectrum: torch.Tensor):
        """Compute K_NM kernel between passed environments and support points"""
        norm = torch.linalg.norm(power_spectrum, dim=1, keepdim=True)
        normalized_power_spectrum = power_spectrum / norm
        return torch.pow(normalized_power_spectrum @ self.support_points.T, self.zeta)

    def compute_KMM(self):
        # suport points are already normalized
        return torch.pow(self.support_points @ self.support_points.T, self.zeta)

    def update_support_points(self, support_points):
        if self.detach_support_points:
            support_points = support_points.detach()

        self.support_points[:] = support_points
