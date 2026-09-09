"""Reserved boundary for music-specific MERT representations."""

from typing import NoReturn


def not_implemented() -> NoReturn:
    """Document the intentional post-MVP boundary.

    TODO: Add versioned MERT segment embeddings and a dedicated vector column/table.
    """

    raise NotImplementedError("MERT analysis is planned after CLAP retrieval is validated")
