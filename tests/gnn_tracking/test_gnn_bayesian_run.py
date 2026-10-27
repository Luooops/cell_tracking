"""models/gnn_bayesian/run.py: what each command trains on and tracks (synthetic dataset folders, no training)."""
import argparse
import tempfile
import unittest
from pathlib import Path

from models.gnn_bayesian.run import units

FOLDS = [['a1', 'a2'], ['b1', 'b2'], ['c1', 'c2']]
TEST = ['t1', 't2']


class TestUnits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.roots = {}
        for name, seqs in (('hela', ['01', '02']), ('sim', ['01', '02'])):
            for s in seqs:
                (root / name / f'{s}_GT' / 'TRA').mkdir(parents=True)
                (root / name / s).mkdir()
            self.roots[name] = root / name
        self.extra = [f'{d}/{s}' for d in ('hela', 'sim') for s in ('01', '02')]

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, command, dataset=None, held_out=None):
        args = argparse.Namespace(command=command, scripta='scr', dataset=dataset, held_out=held_out, folds=[0, 1, 2],
                                  scripta_spec=dict(folds=FOLDS), dev_wells=[w for f in FOLDS for w in f],
                                  test_wells=TEST, extra_roots=self.roots, dest=Path('out'))
        return units(args, self.extra)

    def test_no_command_but_final_sees_a_test_well(self):
        for command, dataset in (('cv', None), ('loso', 'sim'), ('unseen', 'sim')):
            for _, _, specs, track in self.plan(command, dataset, ['01', '02'] if command == 'loso' else None):
                touched = {s.split('/')[1] for s in specs + [t[0] for t in track] if s.startswith('scr/')}
                self.assertFalse(touched & set(TEST), command)
        (_, _, specs, track), = self.plan('final')
        self.assertFalse({s.split('/')[1] for s in specs if s.startswith('scr/')} & set(TEST))
        self.assertEqual([t[0] for t in track], ['scr/t1', 'scr/t2'])

    def test_cv_holds_out_its_fold_and_keeps_every_extra_sequence(self):
        for k, (unit, prefix, specs, track) in enumerate(self.plan('cv')):
            self.assertEqual(prefix, f'fold{k}')
            self.assertEqual({t[0] for t in track}, {f'scr/{w}' for w in FOLDS[k]})
            self.assertFalse({t[0] for t in track} & set(specs))
            self.assertTrue(set(self.extra) <= set(specs))

    def test_loso_and_unseen_hold_out_what_they_test(self):
        for unit, prefix, specs, track in self.plan('loso', 'sim', ['01', '02']):
            self.assertEqual([t[0] for t in track], [f'sim/{unit}'])
            self.assertNotIn(f'sim/{unit}', specs)
            self.assertIn('hela/01', specs)
        (unit, prefix, specs, track), = self.plan('unseen', 'sim')
        self.assertFalse([s for s in specs if s.startswith('sim/')])
        self.assertEqual([t[0] for t in track], ['sim/01', 'sim/02'])
        self.assertEqual(str(track[0][2]('gnn_tracking')), str(Path('out') / 'unseen_sim' / 'gnn_tracking'))


if __name__ == '__main__':
    unittest.main()
