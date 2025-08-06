"""
pennylane_qaoa.py
====================
End-to-end helper to run a QAOA algorithm with PennyLane on the portfolio-optimisation
LP instances used in this repository.
Updated for PennyLane v0.42.1 compatibility.

API
---
run_qaoa(lp_file: str,
        p: int = 3,
        backend: str = "default.qubit",
        shots: int | None = None,
        optimizer: str = "adam",
        max_iter: int = 200,
        lr: float = 0.05,
        verbose: bool = True,
        **backend_kwargs) -> dict
    Returns a dictionary with keys:
        best_x:      np.ndarray (0/1 bit-string)
        best_cost:   float (portfolio objective)
        params:      optimised (gamma, beta) tensor (shape (p,2))
        cost_history:list[float]
Notes
-----
• Requires a helper ``build_ising`` that converts the LP file into Ising
  coefficients (J, h, const).  *Implemented in step_1.py; add it if missing.*
• PennyLane QAOA implementation is kept lightweight and backend-agnostic.
• Updated for PennyLane v0.42.1 compatibility.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# --- Import Ising builder --------------------------------------------------
try:
    from step_1 import build_ising  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("step_1.build_ising not found – please implement it first.") from exc
# ---------------------------------------------------------------------------

def _ising_to_hamiltonian(J: np.ndarray, h: np.ndarray, const: float = 0.0):
    """Convert Ising arrays to a PennyLane Hamiltonian H = ∑ Jij Zi Zj + ∑ hi Zi + const"""
    n = len(h)
    coeffs: list[float] = []
    obs: list[qml.operation.Operator] = []  # Updated: Operator instead of Observable
    
    # ZZ terms
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                coeffs.append(J[i, j])
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    
    # Z terms
    for i in range(n):
        if h[i] != 0.0:
            coeffs.append(h[i])
            obs.append(qml.PauliZ(i))
    
    if const != 0.0:
        coeffs.append(const)
        obs.append(qml.Identity(0))  # constant term
    
    return qml.Hamiltonian(coeffs, obs)

# ---------------------------------------------------------------------------
def _select_device(backend: str, n_qubits: int, shots: int | None, **kwargs):
    """Return a PennyLane device.

    Special cases
    -------------
    • ``backend='auto'``          → choose ``default.mps`` for n>25 else ``default.qubit``.
    • ``backend='default.mps'``   → matrix-product-state simulator (memory-efficient for >30 qubits, low depth).
    """
    if backend == "auto":
        backend = "default.mps" if n_qubits > 25 else "default.qubit"
    
    if backend in {"default.qubit", "default.mps", "lightning.qubit", "lightning.gpu"}:
        return qml.device(backend, wires=n_qubits, shots=shots, **kwargs)

    if backend.startswith("qiskit"):
        # Example backend="qiskit.ibmq"  , pass 'backend_name' via kwargs
        return qml.device(backend, wires=n_qubits, shots=shots, **kwargs)

    # Fallback – let PennyLane handle other targets (e.g. braket.local.qubit)
    return qml.device(backend, wires=n_qubits, shots=shots, **kwargs)

# ---------------------------------------------------------------------------
def run_qaoa(
    lp_file: str | Path,
    p: int = 3,
    backend: str = "default.qubit",
    shots: int | None = None,
    optimizer: str = "adam",
    max_iter: int = 200,
    lr: float = 0.05,
    verbose: bool = True,
    **backend_kwargs: Any,
):
    """Run QAOA on a portfolio LP instance and return results.
    
    Parameters
    ----------
    lp_file : path to LP file.
    p : QAOA depth.
    backend : PennyLane backend string.  E.g. ``"default.qubit"`` or
               ``"qiskit.ibmq"``.
    shots : Number of shots for sampling devices. ``None`` ⇒ analytic mode.
    optimizer : {'adam','cobyla','spsa'}.
    max_iter : optimisation steps.
    lr : learning rate (if applicable).
    verbose : print progress.
    backend_kwargs : forwarded to ``qml.device``.
    """
    # --- Build Ising Hamiltonian -----------------------------------------
    J, h, const = build_ising(str(lp_file))
    n_qubits = len(h)
    H = _ising_to_hamiltonian(J, h, const)
    
    # --- Device & circuit -------------------------------------------------
    dev = _select_device(backend, n_qubits, shots, **backend_kwargs)
    
    # Pre-compute lists for ZZ and Z terms for param-shift efficiency
    zz_idx: list[tuple[int, int]] = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits) if J[i, j] != 0.0]
    z_idx: list[int] = [i for i in range(n_qubits) if h[i] != 0.0]
    
    def cost_layer(gamma):
        for (i, j) in zz_idx:
            qml.IsingZZ(2 * gamma * J[i, j], wires=[i, j])
        for i in z_idx:
            qml.RZ(2 * gamma * h[i], wires=i)
        if const != 0.0:
            qml.PhaseShift(2 * gamma * const, wires=0)  # global phase – ignored in expval
    
    def mixer_layer(beta):
        for i in range(n_qubits):
            qml.RX(2 * beta, wires=i)
    
    # Updated: Use interface="auto" instead of "autograd" for better compatibility
    @qml.qnode(dev, interface="auto")
    def circuit(params):  # params shape (p,2) → γ,β per layer
        # initial state |+>^n
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        
        # alternating layers
        for layer in range(p):
            gamma, beta = params[layer]
            cost_layer(gamma)
            mixer_layer(beta)
        
        return qml.expval(H)
    
    # PennyLane returns autograd objects; convert to pnp.ndarray for safety
    params = pnp.random.uniform(low=0.0, high=np.pi, size=(p, 2))
    
    # --- Optimiser --------------------------------------------------------
    if optimizer == "adam":
        opt = qml.AdamOptimizer(lr)
    elif optimizer == "cobyla":
        opt = qml.COBYLAOptimizer(maxiter=max_iter)
    elif optimizer == "spsa":
        opt = qml.SPSAOptimizer(lr)
    else:
        raise ValueError("Unknown optimizer")
    
    cost_progress: list[float] = []
    for it in range(max_iter):
        if optimizer == "cobyla":
            params = opt.step(circuit, params)
            cost_val = circuit(params)
        else:
            params, cost_val = opt.step_and_cost(circuit, params)
        cost_progress.append(float(cost_val))
        if verbose and it % 10 == 0:
            print(f"Iter {it:3d}  cost={cost_val:.6f}")
    
    # --- Sampling to obtain bitstring ------------------------------------
    sample_dev = _select_device(backend, n_qubits, shots or 1024, **backend_kwargs)
    
    # Updated: Use interface="auto" instead of "autograd"
    @qml.qnode(sample_dev, interface="auto")
    def sample_circuit(params):
        # reuse circuit body
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for layer in range(p):
            gamma, beta = params[layer]
            cost_layer(gamma)
            mixer_layer(beta)
        return qml.sample(wires=range(n_qubits))
    
    bitstrings = sample_circuit(params)
    
    # Updated: Use terms() method instead of obs and coeffs directly
    H_matrix = H.matrix()
    energies = [H_matrix.diagonal()[int("".join(str(b) for b in bs), 2)] for bs in bitstrings]
    best_idx = int(np.argmin(energies))
    best_bs = bitstrings[best_idx]
    best_cost = float(energies[best_idx])
    
    return {
        "best_x": np.array(best_bs),
        "best_cost": best_cost,
        "params": np.array(params),
        "cost_history": cost_progress,
    }
