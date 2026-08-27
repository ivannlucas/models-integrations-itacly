"""Configuration models for the optimization pipeline."""

from pathlib import Path

from pydantic import BaseModel, FilePath, PositiveInt


class PathsConfig(BaseModel):
    """Resolved paths required by the project.

    Attributes:
        project_root: Absolute path to the project root.
        dataset_path: Absolute path to the training dataset.
        neat_config_path: Absolute path to the NEAT configuration file.
    """

    project_root: Path
    dataset_path: FilePath
    neat_config_path: FilePath


class EvolutionConfig(BaseModel):
    """Hyperparameters used for neuroevolution.

    Attributes:
        sample_size: Number of rows sampled for fitness evaluation.
        generations: Number of NEAT generations.
        random_state: Random seed for deterministic sampling.
        strategies: Ordered list of selectable strategies.
    """

    sample_size: PositiveInt = 1000
    generations: PositiveInt = 50
    random_state: int = 42
    strategies: tuple[str, ...] = (
        "Biomass combustion",
        "Animal feed",
        "Composting",
        "Biochar",
    )


class AppConfig(BaseModel):
    """Root configuration object for the optimization pipeline.

    Attributes:
        paths: Validated and resolved project paths.
        evolution: Evolution hyperparameters.
    """

    paths: PathsConfig
    evolution: EvolutionConfig

    @classmethod
    def build(
        cls,
        project_root: Path,
        dataset_relative_path: str,
        neat_config_relative_path: str,
        sample_size: int,
        generations: int,
        random_state: int,
    ) -> "AppConfig":
        """Build an application config from relative paths.

        Args:
            project_root: Project root directory.
            dataset_relative_path: Dataset path relative to project root.
            neat_config_relative_path: NEAT config path relative to project root.
            sample_size: Number of sampled rows for evaluation.
            generations: Number of generations for evolution.
            random_state: Random seed.

        Returns:
            AppConfig: Fully validated application configuration.
        """

        resolved_root = project_root.resolve()
        paths = PathsConfig(
            project_root=resolved_root,
            dataset_path=resolved_root / dataset_relative_path,
            neat_config_path=resolved_root / neat_config_relative_path,
        )
        evolution = EvolutionConfig(
            sample_size=sample_size,
            generations=generations,
            random_state=random_state,
        )
        return cls(paths=paths, evolution=evolution)
