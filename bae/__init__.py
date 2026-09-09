import warnings

import torch


# BAE constructs sparse tensors from generated Jacobian data. Leave invariant
# validation disabled globally; malformed sparse tensors are allowed to fail
# when they are used instead.
torch.sparse.check_sparse_tensor_invariants.disable()

# Sparse BSR support is intentionally used by BAE despite PyTorch labelling it
# beta. Keep that informational warning out of normal pipeline output.
warnings.filterwarnings(
    "ignore",
    message=r"Sparse BSR tensor support is in beta state\..*",
    category=UserWarning,
)

from .utils.pypose_compile import maybe_install_pypose_torch_compile_monkeypatch
from .utils.pypose_ambient_grad import maybe_install_pypose_ambient_grad_monkeypatch

maybe_install_pypose_torch_compile_monkeypatch()
maybe_install_pypose_ambient_grad_monkeypatch()
