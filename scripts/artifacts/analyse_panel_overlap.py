"""Analyse gene-panel overlap between HESCAPE and BEAT default panels."""

from pathlib import Path

import yaml
from loguru import logger

HESCAPE_PANELS = {
    'human-immuno-oncology': Path('panels/hescape/human-immuno-oncology.yaml'),
    'lung-healthy': Path('panels/hescape/lung-healthy.yaml'),
}
BEAT_PANEL = Path('panels/beat/default.yaml')


def load_panel(path: Path) -> dict[str, list[str]]:
    return yaml.safe_load(path.read_text())


def overlap_stats(a: set[str], b: set[str]) -> dict:
    intersection = a & b
    union = a | b
    only_a = a - b
    only_b = b - a
    jaccard = len(intersection) / len(union) if union else 0.0
    return dict(
        size_a=len(a),
        size_b=len(b),
        intersection=len(intersection),
        only_a=sorted(only_a),
        only_b=sorted(only_b),
        jaccard=jaccard,
    )


def analyse_panel_overlap() -> None:
    beat = load_panel(BEAT_PANEL)
    beat_target = set(beat['target_panel'])
    beat_source = set(beat['source_panel'])
    beat_full = beat_target | beat_source

    for name, path in HESCAPE_PANELS.items():
        panel = load_panel(path)
        hescape_target = set(panel['target_panel'])
        hescape_source = set(panel['source_panel'])
        hescape_full = hescape_target | hescape_source

        logger.info(f'=== {name} vs beat/default ===')

        for label, a, b in [
            ('target', hescape_target, beat_target),
            ('source', hescape_source, beat_source),
            ('full  ', hescape_full, beat_full),
        ]:
            s = overlap_stats(a, b)
            logger.info(
                f'  [{label}] hescape={s["size_a"]} beat={s["size_b"]} '
                f'intersection={s["intersection"]} jaccard={s["jaccard"]:.3f} '
                f'only_in_hescape={len(s["only_a"])} only_in_beat={len(s["only_b"])}'
            )
            if s['only_a']:
                logger.info(f'    only in hescape {label}: {s["only_a"]}')


if __name__ == '__main__':
    analyse_panel_overlap()
