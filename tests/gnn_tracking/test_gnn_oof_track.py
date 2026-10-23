"""models/gnn_tracking/oof_track.py: settings and run-folder checks (no model is run)."""
import json
import tempfile
import unittest
from pathlib import Path

from models.gnn_tracking.oof_track import main, parse_sets


class TestOofTrack(unittest.TestCase):
    def test_known_settings_are_parsed_as_json(self):
        allowed = [dict(gap_round=True, link_thresh=0.5), dict(div_thresh=0.5)]
        self.assertEqual(parse_sets(['link_thresh=0.4', 'div_thresh=0.6'], allowed), dict(link_thresh=0.4, div_thresh=0.6))

    def test_unknown_or_malformed_setting_raises(self):
        for item in ('pos_weight=1', 'link_thresh'):
            with self.assertRaises(ValueError):
                parse_sets([item], [dict(link_thresh=0.5)])

    def test_folder_without_cv_checkpoints_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'train'
            run.mkdir()
            (run / 'config.json').write_text(json.dumps(dict(cv=False)), encoding='utf-8')
            for folder in (run, Path(tmp) / 'missing'):
                with self.assertRaises(SystemExit):
                    main(['--cv-run', str(folder), '--output-root', str(Path(tmp) / 'out')])
            (run / 'checkpoints').mkdir()
            with self.assertRaises(SystemExit):
                main(['--cv-run', str(run), '--output-root', str(Path(tmp) / 'out')])
            self.assertFalse((Path(tmp) / 'out').exists())


if __name__ == '__main__':
    unittest.main()
