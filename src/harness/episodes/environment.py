"""Actor environment owns public data only; it cannot retrieve evaluator answers."""
from ..interactive import InteractiveHarness


class EpisodeEnvironment(InteractiveHarness):
    """Reuse the interactive Session and checkpoint protocol with a SQLite catalog."""

    @property
    def manifest(self):
        return {**super().manifest, "episode_environment_version": "public-episode-env-v1"}
