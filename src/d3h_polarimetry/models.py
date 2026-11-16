from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from enum import Enum


class FFTMode(Enum):
    """Supported D3h FFT modes: fixed bin 4 and rotating bin 6; phase divided by 6."""

    D3HFIXED = "d3hfixed"
    D3HROTATING = "d3hrotating"


def d3h_fixed_fft(
    a_0: float, a_4: float, phi: float, alpha: np.ndarray
) -> np.ndarray:
    """FFT-based theoretical model for SHG (D3HFIXED mode)."""
    return a_0 + a_4 * np.cos(2 * (3 * phi - 2 * alpha))


def d3h_rotating_fft(
    a_0: float, a_6: float, phi: float, alpha: np.ndarray
) -> np.ndarray:
    """FFT-based theoretical model for SHG (D3HROTATING mode)."""
    return a_0 + a_6 * np.cos(6 * (phi - alpha))


class PhysicalModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the model."""
        pass

    @property
    @abstractmethod
    def parameter_names(self) -> Tuple[str, ...]:
        """Names of the fit parameters."""
        pass

    @abstractmethod
    def model_function(self, alpha: np.ndarray, *params) -> np.ndarray:
        pass

    def get_initial_guess(self) -> Tuple[float, ...]:
        """Return initial parameter guesses for fitting. Override for custom values."""
        return (100.0, 100.0, np.pi / 4)

    def get_bounds(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Return (lower_bounds, upper_bounds) for parameters. Override for custom values."""
        return ([0, 0, 0], [100000, 100000, np.pi / 3])


class D3hFixedModel(PhysicalModel):
    @property
    def name(self) -> str:
        return "D3h fixed analyzer (d3hfixed)"

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return ("A", "B", "phi")

    def model_function(
        self, alpha: np.ndarray, A: float, B: float, phi: float
    ) -> np.ndarray:
        return B + A * np.cos(3 * phi - 2 * alpha) ** 2


class D3hRotatingModel(PhysicalModel):
    @property
    def name(self) -> str:
        return "D3h rotating analyzer (d3hrotating)"

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return ("A", "B", "phi")

    def model_function(
        self, alpha: np.ndarray, A: float, B: float, phi: float
    ) -> np.ndarray:
        return B + A * np.cos(3 * phi - 3 * alpha) ** 2
