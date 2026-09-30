"""Knowledge distillation engine for transferring knowledge from large teacher models to compact student models.

Implements Hinton temperature-scaled KL-divergence distillation loss, response-based distillation,
and student model training workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training."""

    temperature: float = 4.0
    alpha: float = 0.5  # Weight balance: alpha * soft_loss + (1 - alpha) * hard_loss
    learning_rate: float = 1e-3
    epochs: int = 5
    batch_size: int = 32


class DistillationLoss(nn.Module):
    """Combined loss function for knowledge distillation.

    Loss = alpha * T^2 * KL_Divergence(softmax(student/T), softmax(teacher/T))
         + (1 - alpha) * Task_Loss(student, target)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        task_loss_fn: nn.Module | None = None,
    ) -> None:
        """Initialize distillation loss."""
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.task_loss_fn = task_loss_fn or nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute distillation loss."""
        # Temperature-scaled soft targets
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature**2)

        if targets is not None and self.alpha < 1.0:
            hard_loss = self.task_loss_fn(student_logits, targets)
            total_loss = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
        else:
            total_loss = soft_loss

        return total_loss


class KnowledgeDistiller:
    """Orchestrates knowledge distillation from teacher to student model."""

    def __init__(self, config: DistillationConfig | None = None) -> None:
        """Initialize distiller with configuration."""
        self.config = config or DistillationConfig()
        self.loss_fn = DistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha,
        )

    def distill_step(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        """Perform a single distillation forward and backward step."""
        student_model.train()
        teacher_model.eval()

        with torch.no_grad():
            teacher_logits = teacher_model(inputs)

        student_logits = student_model(inputs)
        loss = self.loss_fn(student_logits, teacher_logits, targets)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return float(loss.item())

    def distill(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        dataloader: Any,
        epochs: int | None = None,
        learning_rate: float | None = None,
        device: str = "cpu",
    ) -> nn.Module:
        """Run full knowledge distillation training loop."""
        num_epochs = epochs or self.config.epochs
        lr = learning_rate or self.config.learning_rate

        student = student_model.to(device)
        teacher = teacher_model.to(device)
        teacher.eval()

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        logger.info("Starting knowledge distillation for %d epochs...", num_epochs)
        for epoch in range(num_epochs):
            total_loss = 0.0
            steps = 0
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    inputs, targets = batch.to(device), None

                loss_val = self.distill_step(student, teacher, inputs, targets, optimizer)
                total_loss += loss_val
                steps += 1

            avg_loss = total_loss / max(1, steps)
            logger.info("Epoch [%d/%d] - Distillation Loss: %.4f", epoch + 1, num_epochs, avg_loss)

        return student
