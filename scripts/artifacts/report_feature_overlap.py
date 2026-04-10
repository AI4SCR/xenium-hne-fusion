"""Report sample feature-universe overlap for an artifacts config and save a heatmap."""

import sys

from dotenv import load_dotenv

load_dotenv()

from xenium_hne_fusion.panel_overlap import report_feature_overlap
from xenium_hne_fusion.processing_cli import parse_artifacts_args


def main(argv: list[str] | None = None) -> int:
    artifacts_cfg, _ = parse_artifacts_args(argv)
    report, output_path = report_feature_overlap(artifacts_cfg)
    print(report)
    print()
    print(f'Plot saved to {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
