"""Known-type persistence tests using synthetic records, never market results."""
from dataclasses import asdict, replace
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_training_data import Fixture, Tick, ROOT, TZ
from paperbot_r2 import training_codec as codec, model_training, classifier, hmm
from paperbot_r2.contracts import digest


def reseal(cls, **fields):
    return cls(**fields, seal=digest({k: asdict(v) if hasattr(v, '__dataclass_fields__') else
        [asdict(a) for a in v] if k == 'attempts' else v for k, v in fields.items()}))


class TrainingCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tz = zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT / 'outputs/zoneinfo')]); zoneinfo.ZoneInfo.clear_cache()

    @classmethod
    def tearDownClass(cls):
        zoneinfo.reset_tzpath(cls.old_tz); zoneinfo.ZoneInfo.clear_cache()

    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True); env.start(); self.addCleanup(env.stop)
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.f = Fixture(Path(temp.name)); self.f.finish()
        self.snapshot = self.f.read()
        self.fits = model_training.fit_training_snapshot(self.snapshot, timezone_file=TZ,
            clock=Tick(self.f.at + timedelta(seconds=1)))

    def test_actual_snapshot_and_all_fixed_attempts_roundtrip_exactly(self):
        self.assertEqual(codec.decode_snapshot(codec.encode(self.snapshot)), self.snapshot)
        self.assertEqual(codec.decode_fits(codec.encode(self.fits)), self.fits)
        self.assertTrue(any(a.status == 'failed' and a.payload is None for a in self.fits.attempts))

    def test_unknown_fields_tampering_nonstandard_json_and_size_are_rejected(self):
        data = json.loads(codec.encode(self.snapshot))
        data['unexpected'] = 'field'
        for value in (data, '{"ready":true,"ready":false}', '{"ready":NaN}', ' ' * (8388608 + 1)):
            with self.assertRaises(codec.TrainingCodecError):
                codec.decode_snapshot(value)
        data = json.loads(codec.encode(self.fits)); data['attempts'][0]['symbol'] = 'MSFT'
        with self.assertRaises(codec.TrainingCodecError):
            codec.decode_fits(data)

    def test_successful_nb_branch_restores_nested_tuples(self):
        # Pure constructor/serialization fixture; no source-receipt claim.
        first = self.snapshot.symbols[0].examples[0]
        rows = []
        for i in range(64):
            event = first.features.event_at + timedelta(seconds=900 * i)
            vector = replace(first.features, values=(.001 * (i % 7), .01 + .001 * (i % 9), .2 + .01 * (i % 11)),
                event_at=event, available_at=event + timedelta(seconds=3), source_id=digest(dict(synthetic_vector=i)))
            rows.append(replace(first, features=vector, target_start=event + timedelta(seconds=900),
                target_end=event + timedelta(seconds=1800), outcome_available_at=event + timedelta(seconds=1801),
                label=i % 2, outcome_id=digest(dict(synthetic_outcome=i))))
        attempt = next(a for a in self.fits.attempts if a.symbol == 'AAPL' and a.family == 'gaussian_nb')
        model = classifier.fit_classifier(rows, as_of=attempt.started_at, variance_floor=1e-6)
        fields = {k: v for k, v in asdict(attempt).items() if k != 'seal'}
        fields.update(status='succeeded', failure_code=None,
            payload=model_training.FittedPayload(model, None), input_count=64,
            input_rows_sha256=digest([asdict(r) for r in rows]))
        replacement = reseal(model_training.FitAttempt, **fields)
        fields = {k: v for k, v in asdict(self.fits).items() if k != 'seal'}
        fields['attempts'] = tuple(replacement if a is attempt else a for a in self.fits.attempts)
        fits = reseal(model_training.TrainingFits, **fields)
        self.assertEqual(codec.decode_fits(codec.encode(fits)), fits)

    def test_hmm_negative_infinity_is_tagged_and_reconstructed(self):
        attempt = next(a for a in self.fits.attempts if a.symbol == 'AAPL' and a.family == 'hmm')
        self.assertEqual(attempt.status, 'succeeded', 'synthetic HMM fixture must fit')
        model = replace(attempt.payload.model, final_log_posterior=(-math.inf, 0.))
        state = hmm.initial_hmm_state(model)
        fields = {k: v for k, v in asdict(attempt).items() if k != 'seal'}
        fields['payload'] = model_training.FittedPayload(model, state)
        replacement = reseal(model_training.FitAttempt, **fields)
        fields = {k: v for k, v in asdict(self.fits).items() if k != 'seal'}
        fields['attempts'] = tuple(replacement if a is attempt else a for a in self.fits.attempts)
        fits = reseal(model_training.TrainingFits, **fields)
        encoded = codec.encode(fits)
        self.assertNotIn('-Infinity', encoded)
        self.assertIn('negative_infinity', encoded)
        self.assertEqual(codec.decode_fits(encoded), fits)


if __name__ == '__main__':
    unittest.main()
