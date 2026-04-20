from pathlib import Path

from scripts.eval.plot_wandb_scores import _build_parser


def test_eval_plot_cli_reads_color_by_splits_from_eval_config(tmp_path: Path):
    config_path = tmp_path / 'eval.yaml'
    config_path.write_text(
        '\n'.join(
            [
                'project: xe-hne-fus-expr',
                'target: expression',
                'name: hest1k',
                'items_path: all.json',
                'metadata_dir: hescape/breast',
                'baseline: vision',
                'color_by_splits: true',
                'sort_by_score: false',
            ]
        )
    )

    args = _build_parser().parse_args(['--config', str(config_path)]).as_dict()

    assert args['color_by_splits'] is True
    assert args['sort_by_score'] is False
