"""Generate the versionable PD12M EDA plot cache with maintained renderers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.config.config_paths import ProjectPaths, default_paths, select_dataset
from src.data.data_metadata import add_jpeg_metadata
from src.data.data_preparation import prepare_dataset, resolve_split_config, select_eda_population
from src.data.data_synthetic import validate_proxy_dataset
from src.presentation.presentation_cache import resolve_cached_figure_path, save_figure
from src.presentation.presentation_widgets import make_eda_plot_specs

DEFAULT_SEED = 42
EXPECTED_PD12M_GROUPS = 500


def _prepare_pd12m_dataframe(paths: ProjectPaths, *, seed: int, expected_groups: int) -> pd.DataFrame:
    selection = select_dataset(paths, source="synthetic")
    if selection.cache_namespace != "pd12m" or selection.root.resolve() != paths.pd12m.resolve():
        raise RuntimeError("Explicit synthetic selection did not resolve exclusively to PD12M.")
    groups = validate_proxy_dataset(selection.root)
    if groups != expected_groups:
        raise ValueError(f"Expected {expected_groups} complete PD12M groups, found {groups}.")

    split_config = resolve_split_config(selection, seed=seed)
    prepared = prepare_dataset(selection, split_config=split_config)
    eda_index = select_eda_population(
        prepared.index,
        selection=selection,
        alaska2_group_count=1,
        seed=seed,
    )
    pd12m_root = paths.pd12m.resolve()
    alaska2_root = paths.alaska2.resolve()
    for raw_path in eda_index["path"]:
        image_path = Path(str(raw_path)).resolve()
        if pd12m_root not in image_path.parents or image_path == pd12m_root:
            raise RuntimeError(f"EDA input escaped the PD12M root: {image_path}")
        if alaska2_root in image_path.parents or image_path == alaska2_root:
            raise RuntimeError(f"ALASKA2 input reached PD12M cache generation: {image_path}")
    dataframe = add_jpeg_metadata(eda_index, strict=True, show_progress=True)
    if dataframe["source_id"].nunique() != groups or len(dataframe) != groups * 4:
        raise RuntimeError("PD12M metadata extraction changed the complete-group population.")
    return dataframe


def generate_cache(*, paths: ProjectPaths, seed: int, expected_groups: int) -> tuple[Path, ...]:
    """Render and validate every static notebook view supported by the cache contract."""
    dataframe = _prepare_pd12m_dataframe(paths, seed=seed, expected_groups=expected_groups)
    catalog = make_eda_plot_specs(dataframe, "PD12M", seed=seed)
    specs = tuple(spec for section in catalog.values() for spec in section)
    generated: list[Path] = []

    for spec in (spec for spec in specs if spec.prebuild_cache):
        result = spec.render()
        if not isinstance(result, Figure):
            raise TypeError(f"Static cache view {spec.title!r} did not return a Matplotlib figure.")
        try:
            destination = save_figure(
                result,
                "pd12m",
                spec.cache_name,
                paths=paths,
                renderer=spec.cache_renderer,
                parameters=spec.cache_parameters,
                seed=spec.cache_seed,
                source_groups=spec.source_groups,
                image_count=spec.image_count,
            )
        finally:
            plt.close(result)

        resolved = resolve_cached_figure_path(
            "pd12m",
            spec.cache_name,
            paths=paths,
            renderer=spec.cache_renderer,
            parameters=spec.cache_parameters,
            seed=spec.cache_seed,
            source_groups=spec.source_groups,
            image_count=spec.image_count,
        )
        if resolved != destination:
            raise RuntimeError(f"Generated cache entry did not validate: {destination}")
        generated.append(destination)
        print(f"prepared {spec.cache_name}: {destination.stat().st_size} bytes")

    if plt.get_fignums():
        raise RuntimeError(f"Matplotlib figures remained open: {plt.get_fignums()}")
    dynamic_titles = (spec.title for spec in specs if not spec.prebuild_cache)
    print(f"prepared and validated {len(generated)} static plots")
    print("dynamic views: " + "; ".join(dynamic_titles))
    return tuple(generated)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-groups", type=int, default=EXPECTED_PD12M_GROUPS)
    arguments = parser.parse_args(argv)
    if arguments.expected_groups <= 0:
        parser.error("--expected-groups must be positive")
    generate_cache(
        paths=default_paths(),
        seed=arguments.seed,
        expected_groups=arguments.expected_groups,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
