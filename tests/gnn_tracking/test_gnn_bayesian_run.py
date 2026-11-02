"""models/gnn_bayesian/run.py: what each command trains on and tracks (synthetic dataset folders, no training)."""
import argparse
import tempfile
import unittest
from pathlib import Path

from models.gnn_bayesian.run import extra_sequences, units

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

    def args(self, command, dataset=None, held_out=None, extra_test=None, unseen=()):
        return argparse.Namespace(command=command, scripta='scr', dataset=dataset, held_out=held_out, folds=[0, 1, 2],
                                  scripta_spec=dict(folds=FOLDS), dev_wells=[w for f in FOLDS for w in f],
                                  test_wells=TEST, extra=['hela', 'sim'], extra_roots=self.roots,
                                  extra_test=extra_test or {'hela': [], 'sim': []}, unseen=list(unseen),
                                  unseen_roots={d: self.roots[d] for d in unseen}, dest=Path('out'))

    def plan(self, command, dataset=None, held_out=None):
        return units(self.args(command, dataset, held_out), self.extra)

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

    def test_declared_test_sequences_of_an_extra_dataset_only_tracked_by_final(self):
        args = self.args('cv', extra_test={'hela': [], 'sim': ['02']})
        extra = extra_sequences(args)
        self.assertEqual(extra, ['hela/01', 'hela/02', 'sim/01'])
        for _, _, specs, track in units(args, extra):
            self.assertNotIn('sim/02', specs + [t[0] for t in track])
        args.command = 'final'
        (_, _, specs, track), = units(args, extra)
        self.assertNotIn('sim/02', specs)
        self.assertEqual([t[0] for t in track], ['scr/t1', 'scr/t2', 'sim/02'])
        self.assertEqual(str(track[-1][2]('gnn_division')), str(Path('out') / 'sim' / 'gnn_division'))

    def test_transfer_trains_without_scripta_and_tracks_every_scripta_well(self):
        (unit, prefix, specs, track), = self.plan('transfer')
        self.assertEqual(prefix, 'transfer')
        self.assertEqual(specs, self.extra)
        self.assertEqual([t[0] for t in track], [f'scr/{w}' for f in FOLDS for w in f] + ['scr/t1', 'scr/t2'])
        self.assertEqual(str(track[0][2]('gnn_division')), str(Path('out') / 'dev' / 'gnn_division'))
        self.assertEqual(str(track[-1][2]('gnn_division')), str(Path('out') / 'test' / 'gnn_division'))

    def test_transfer_trains_on_chosen_videos_and_tracks_the_others(self):
        args = self.args('transfer', extra_test={'hela': [], 'sim': ['02']})
        args.train_videos, args.track_videos = ['sim/01'], ['hela/01', 'sim/02']
        extra = extra_sequences(args)
        self.assertEqual(extra, ['sim/01'])
        (_, _, specs, track), = units(args, extra)
        self.assertEqual(specs, ['sim/01'])
        self.assertEqual([t[0] for t in track][-2:], ['hela/01', 'sim/02'])
        self.assertEqual([t[1] for t in track][-2:], ['hela', 'sim'])
        self.assertEqual(str(track[-1][2]('gnn_division')), str(Path('out') / 'sim' / 'gnn_division'))

    def test_cv_with_chosen_videos_trains_on_them_and_tracks_scripta_only(self):
        args = self.args('cv')
        args.train_videos = ['sim/01', 'sim/02']
        extra = extra_sequences(args)
        for k, (_, _, specs, track) in enumerate(units(args, extra)):
            self.assertEqual([s for s in specs if not s.startswith('scr/')], ['sim/01', 'sim/02'])
            self.assertEqual([t[0] for t in track], [f'scr/{w}' for w in FOLDS[k]])

    def test_final_tracks_unseen_datasets_without_training_on_them(self):
        args = self.args('final', unseen=['hela'])
        args.extra, args.extra_roots = ['sim'], {'sim': self.roots['sim']}
        args.extra_test = {'sim': []}
        extra = extra_sequences(args)
        (_, _, specs, track), = units(args, extra)
        self.assertFalse([s for s in specs if s.startswith('hela/')])
        self.assertEqual([t[0] for t in track], ['scr/t1', 'scr/t2', 'hela/01', 'hela/02'])
        self.assertEqual(str(track[-1][2]('gnn_tracking')), str(Path('out') / 'unseen_hela' / 'gnn_tracking'))


if __name__ == '__main__':
    unittest.main()
