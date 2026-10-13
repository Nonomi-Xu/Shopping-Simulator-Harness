"""Opt-in interaction branch: partial requirement disclosure, user dialogue,
long-term preferences and the episode/experiment evaluation layers.

This package is not part of the offline Test1 mainline. The boundary is
one-way and enforced by ``src/harness/test/test_branch_boundary.py``:

* ``src.harness.branch`` may import ``src.harness`` (the mainline);
* ``src.harness`` must never import ``src.harness.branch``.

Only the ``shopping-agent`` CLI discovers this package, and it does so
optionally, so deleting ``src/harness/branch/`` leaves the mainline intact.
"""

__all__: list[str] = []
