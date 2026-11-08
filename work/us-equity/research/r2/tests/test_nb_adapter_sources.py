"""Successful NB through real temporary sources, never invented receipts.

All prices/times are deliberately synthetic. No market result/order authority
is claimed; the original journal must perform its normal canonical checks.
"""
from datetime import timedelta
from dataclasses import asdict
from pathlib import Path
import os
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path.insert(0,str(Path(__file__).resolve().parent))
import issue_fixture
from test_training_data import ROOT,TZ,Tick,Fixture,DAYS,SYMBOLS
from test_canonical_history import HANDOFF
from paperbot_us.data import Calendar
from paperbot_r2 import receipt_book,feature_pipeline,common_outcome_journal,validation_issue_journal as issuer,classifier
from paperbot_r2.contracts import digest


class ForwardTrainingFixture(Fixture):
    """Stream four AAPL sessions to produce64first common labels on time."""
    def __init__(self,directory):
        self.directory=directory
        self.sessions=tuple(issue_fixture.IssueWorld.session(day) for day in DAYS)
        self.before=self.sessions[0].opens-timedelta(minutes=30)
        raw_plan=receipt_book.ReceiptPlan(SYMBOLS,DAYS[0],'2026-11-25',issuer.inputs.TIMEZONE_SHA256)
        self.raw=receipt_book.ReceiptBook(directory/'raw.sqlite');self.raw.initialize(raw_plan,now=self.before)
        calendar=Calendar(list(self.sessions),start=DAYS[0],end='2026-11-25')
        for day in DAYS:self.raw.record_calendar(calendar,day=day,received_at=self.before)
        paused=tuple((issuer.aware('2026-11-10T00:00:00+00:00')+timedelta(days=i)).date().isoformat() for i in range(14))
        plan=feature_pipeline.PipelinePlan(SYMBOLS,DAYS[0],'2026-11-25',digest(asdict(raw_plan)),
            receipt_book._source(),raw_plan.timezone_sha256,paused_days=paused)
        self.features=feature_pipeline.FeatureJournal(directory/'features.sqlite');self.features.initialize(plan,now=self.before)
        self.outcomes=common_outcome_journal.OutcomeJournal(directory/'common.sqlite')
        self.outcomes.initialize(self.features,self.raw,clock=lambda:self.before)
        self.at=self.before
        for session in self.sessions[:4]:
            for offset in range(26):
                bar=self.bar(session,offset,'AAPL');self.raw.observe([bar],now=bar.received_at+timedelta(seconds=1))
                self.at=bar.received_at+timedelta(seconds=2)
                feature_pipeline.derive_once(self.features,raw_book=self.raw,clock=Tick(self.at))
                if offset>=10:
                    key=feature_pipeline._entry_key('AAPL',session.opens+timedelta(seconds=900*(offset-1)))
                    self.outcomes.record_first(self.features,self.raw,key,clock=Tick(self.at+timedelta(seconds=1)))
                self.at+=timedelta(seconds=2)
        self.keys=tuple(self.features.snapshot()['entries'])


class NBSourceAdapterTests(unittest.TestCase):
    def test_successful_nb_original_and_successor_use_actual_acknowledged_models(self):
        previous=zoneinfo.TZPATH;zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()
        self.addCleanup(zoneinfo.reset_tzpath,previous);self.addCleanup(zoneinfo.ZoneInfo.clear_cache)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        with (patch.dict(os.environ,{},clear=True),
              patch.object(socket,'create_connection',side_effect=AssertionError('no network')),
              patch.object(issue_fixture,'Fixture',ForwardTrainingFixture)):
            world=issue_fixture.IssueWorld(Path(temp.name),issue_source_sha256=issuer.source_sha256())
            original=world.training.load_acknowledged(clock=Tick(HANDOFF))
            source=next(s for s in original.training.symbols if s.symbol=='AAPL')
            attempt=next(a for a in original.fits.attempts if a.symbol=='AAPL' and a.family=='gaussian_nb')
            self.assertEqual(len(source.examples),64);self.assertEqual(attempt.status,'succeeded')
            self.assertGreaterEqual(min(attempt.payload.model.class_counts),16)
            journal=issuer.ValidationIssueJournal(Path(temp.name)/('r2-validation-'+world.identity+'.sqlite'))
            journal.initialize(world.registry,world.identity,world.training,world.source_pairs(),
                timezone_file=TZ,clock=Tick(HANDOFF+timedelta(minutes=1)))
            for day in ('2026-11-24','2026-11-27'):
                key=world.receive_until(day,9);clock=Tick(world.at)
                journal.issue_once(key,registry=world.registry,training_journal=world.training,pairs=world.source_pairs(),
                                   timezone_file=TZ,clock=clock)
                receipt=journal.load_issued(key,clock=clock)
                row=next(r for r in receipt.panel.rows if r.family=='gaussian_nb')
                expected=classifier.classify_regime(attempt.payload.model,receipt.issue_input.bundle.vector,
                    issued_at=receipt.issue_input.available_at,target_start=receipt.panel.target_start,target_end=receipt.panel.target_end)
                self.assertEqual(row.status,'computed');self.assertEqual(row.probability,expected.probabilities[1])
                self.assertEqual(row.model_sha256,attempt.seal)
            self.assertEqual(world.training.snapshot()['reserved_trial_units'],15)
            self.assertEqual(world.training.load_acknowledged(clock=clock).fits.seal,original.fits.seal)
