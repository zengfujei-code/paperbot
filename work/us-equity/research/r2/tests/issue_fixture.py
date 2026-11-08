"""Synthetic future training/registration/two-segment world; no market evidence."""
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from test_training_data import Fixture, Tick, TZ, SYMBOLS
from test_canonical_history import legacy_fixture, HANDOFF
from paperbot_us.data import Bar, Calendar, Session
from paperbot_r2 import canonical_history as canonical, common_registration as common
from paperbot_r2 import receipt_book, feature_pipeline
from paperbot_r2.contracts import digest
from paperbot_r2.training_journal import TrainingJournal


class IssueWorld:
    def __init__(self, directory, *, issue_source_sha256='c'*64):
        self.directory=Path(directory);self.f=Fixture(self.directory);self.f.finish()
        self.training=TrainingJournal(self.directory/'training.sqlite')
        clock=Tick(self.f.at+timedelta(seconds=1))
        self.training.initialize(timezone_file=TZ,clock=clock)
        self.training.run_once(self.f.raw,self.f.features,self.f.outcomes,timezone_file=TZ,clock=clock)
        self.old,_=legacy_fixture(self.directory/'shared.sqlite')
        self.registry=canonical.CanonicalRegistry(self.old.path);clock=Tick(HANDOFF)
        anchor=canonical.inspect_legacy(self.old,timezone_file=TZ,clock=clock)
        self.registry.continue_legacy(expected_history=anchor,timezone_file=TZ,clock=clock)
        key=common.import_training(self.registry,self.training,timezone_file=TZ,clock=clock)
        days=common.VALIDATION_DAYS+common.TEST_DAYS
        calendar=Calendar([self.session(d) for d in days],start=days[0],end=days[-1])
        record=self.registry.snapshot(timezone_file=TZ,clock=clock)['training_imports'][key]
        plan=common.build_common_plan(record,calendar,calendar_received_at=HANDOFF,
            issue_source_sha256=issue_source_sha256,configuration_sha256='f'*64)
        self.identity=common.reserve_common(self.registry,plan,timezone_file=TZ,clock=clock)
        self.registration=common.load_registration(self.registry,self.identity,timezone_file=TZ,clock=clock)
        self.training_receipt=self.training.load_acknowledged(clock=clock)
        plan=receipt_book.ReceiptPlan(SYMBOLS,'2026-11-27','2026-12-11',self.f.raw.snapshot()['plan'].timezone_sha256)
        self.next_raw=receipt_book.ReceiptBook(self.directory/'next-raw.sqlite')
        self.next_raw.initialize(plan,now=HANDOFF)
        self.next_features=feature_pipeline.FeatureJournal(self.directory/'next-features.sqlite')
        feature_plan=feature_pipeline.PipelinePlan(SYMBOLS,plan.start_day,plan.end_day,
            digest(asdict(plan)),receipt_book._source(),plan.timezone_sha256)
        self.next_features.initialize(feature_plan,now=HANDOFF)
        self.at=HANDOFF;self.fed={}

    @staticmethod
    def session(day):
        return Session.from_alpaca(dict(date=day,open='09:30',close='13:00' if day=='2026-11-27' else '16:00'))

    def pair(self,day):
        return (self.f.raw,self.f.features) if day<='2026-11-25' else (self.next_raw,self.next_features)

    def receive_until(self,day,count,*,omit=(),derive=True):
        session=self.session(day);raw,features=self.pair(day)
        if day not in raw.snapshot()['calendars']:
            raw.record_calendar(Calendar([session],start=day,end=day),day=day,
                                received_at=session.opens-timedelta(minutes=30))
        seen=self.fed.setdefault(day,set())
        for i in range(count):
            if i in seen:continue
            seen.add(i)
            if i in omit:continue
            start=session.opens+timedelta(seconds=900*i)
            bars=[]
            for symbol,multiplier in zip(SYMBOLS,(1.,2.,4.)):
                close=(100.+(.4,-.3,.2,-.1,.6,-.4,.1,-.2,.5)[i%9])*multiplier
                bars.append(Bar(symbol,start,start+timedelta(seconds=901),100.*multiplier,
                    102.*multiplier,98.*multiplier,close,100.+i))
            raw.observe(bars,now=bars[0].received_at+timedelta(seconds=1))
            self.at=bars[0].received_at+timedelta(seconds=2)
        if derive:feature_pipeline.derive_once(features,raw_book=raw,clock=Tick(self.at))
        self.at+=timedelta(seconds=1)
        return feature_pipeline._entry_key('AAPL',session.opens+timedelta(seconds=900*count))

    def source_pairs(self):
        from paperbot_r2.forecast_inputs import SourcePair
        return (SourcePair(self.f.raw,self.f.features),SourcePair(self.next_raw,self.next_features))
