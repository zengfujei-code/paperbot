from contextlib import closing
from dataclasses import replace
from datetime import date,datetime,timedelta,timezone
import importlib,sqlite3,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from paperbot_us.baseline_study import BaselineCandidate
from paperbot_us.data import Bar,Calendar,Session
from paperbot_us.experiments import ExperimentRegistry
from paperbot_us.research import ResearchDataset


class ProspectiveTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('paperbot_us.prospective')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.registry=ExperimentRegistry(self.root/'shared.sqlite');self.registry.initialize()
        days=['2026-10-01','2026-10-02','2026-10-05','2026-10-06','2026-10-07','2026-10-08']
        sessions=[Session.from_alpaca(dict(date=d,open='09:30',close='10:30')) for d in days]
        self.calendar=Calendar(sessions,start=days[0],end=days[-1])
        self.now=datetime(2026,10,5,12,tzinfo=timezone.utc);downloaded=self.now-timedelta(days=1)
        training_calendar=Calendar(sessions[:2],start=days[0],end=days[1])
        bars=[Bar('AAPL',s.opens+timedelta(minutes=15*i),downloaded,100+i,102+i,99+i,101+i,10000)
              for s in sessions[:2] for i in range(4)]
        self.training=ResearchDataset.from_bars(bars,calendar=training_calendar,downloaded_at=downloaded,
                                              availability_policy='vendor_history_assumed_at_close')
        self.validation=(date(2026,10,6),);self.test=(date(2026,10,8),)
        self.context=(date(2026,10,5),date(2026,10,7))
        self.candidates=(BaselineCandidate(.5,.5,6),BaselineCandidate(.94,.5,6))

    def create(self,**changes):
        values=dict(calendar=self.calendar,validation_days=self.validation,test_days=self.test,
            context_days=self.context,candidates=self.candidates,registry=self.registry,
            directory=self.root/'campaign',now=self.now,domain='synthetic')
        values.update(changes);return self.m.create_campaign(self.training,**values)

    def bar(self,day,index,received=None,close=None):
        session=self.calendar.sessions[day];starts=session.opens+timedelta(minutes=15*index)
        price=100+index if close is None else close
        return Bar('AAPL',starts,received or starts+timedelta(minutes=15,seconds=1),price,price+1,price-1,price,10000)

    def feed_day(self,campaign,day):
        rows=[]
        for i in range(4):
            row=self.bar(day,i);rows.append(row)
            campaign.observe(rows,now=row.received_at,calendar_verified=True)

    def finish_validation(self,campaign):
        self.feed_day(campaign,self.validation[0])
        campaign.observe([],now=self.calendar.sessions[self.validation[-1]].closes+timedelta(seconds=3601),calendar_verified=True)

    def test_create_fits_now_and_binds_future_calendar_and_original_registry(self):
        campaign=self.create();state=campaign.snapshot()
        self.assertTrue(all(model.fitted_at==self.now for model in state['models'].values()))
        self.assertEqual(state['forecasts'],{})
        self.assertEqual(self.registry.load(campaign.identity)['plan']['split']['validation'],['2026-10-06'])
        reopened=self.m.open_campaign(campaign.directory,campaign.configuration_sha256)
        self.assertEqual(reopened.snapshot()['models'],state['models'])
        with self.assertRaises(self.m.CampaignError):self.m.open_campaign(campaign.directory,'0'*64)

    def test_creation_after_validation_begins_cannot_backdate_a_campaign(self):
        with self.assertRaises(self.m.CampaignError):self.create(now=self.calendar.sessions[self.validation[0]].opens)

    def test_forecast_is_issued_before_outcome_then_settled_from_actual_receipts(self):
        campaign=self.create();anchor=self.bar(self.validation[0],0)
        campaign.observe([anchor],now=anchor.received_at,calendar_verified=True)
        before=campaign.snapshot();self.assertEqual(len(before['forecasts']),2)
        for forecast in before['forecasts'].values():
            self.assertLess(forecast['issued_at'],forecast['target_ends_at']);self.assertIsNone(forecast['outcome_return'])
            self.assertEqual(forecast['future_horizon_seconds'],899)
            self.assertEqual(forecast['target_elapsed_seconds'],1)
            self.assertEqual(forecast['variance_unit'],'fraction_squared_per_15m')
        target=self.bar(self.validation[0],1)
        campaign.observe([anchor,target],now=target.received_at,calendar_verified=True)
        after=campaign.snapshot()
        for key,forecast in before['forecasts'].items():
            current=after['forecasts'][key]
            self.assertEqual(current['variance'],forecast['variance'])
            self.assertAlmostEqual(current['outcome_return'],.01)
            self.assertEqual(current['outcome_received_at'],target.received_at)

    def test_outage_backfill_updates_observed_state_but_creates_no_past_forecasts(self):
        campaign=self.create();session=self.calendar.sessions[self.validation[0]]
        now=session.closes+timedelta(seconds=30)
        rows=[self.bar(session.day,i,received=now) for i in range(4)]
        campaign.observe(rows,now=now,calendar_verified=True);state=campaign.snapshot()
        self.assertFalse(state['forecasts'])
        self.assertTrue(all(s.observations==3 and s.updated_at==now for s in state['states'].values()))

    def test_duplicate_and_revised_bars_do_not_rewrite_forecasts_or_first_prices(self):
        campaign=self.create();anchor=self.bar(self.validation[0],0)
        campaign.observe([anchor],now=anchor.received_at,calendar_verified=True);before=campaign.snapshot()
        campaign.observe([anchor],now=anchor.received_at+timedelta(seconds=1),calendar_verified=True)
        self.assertEqual(campaign.snapshot()['records'],before['records'])
        revised=replace(anchor,close=101,high=102,received_at=anchor.received_at+timedelta(seconds=10))
        campaign.observe([revised],now=revised.received_at,calendar_verified=True);after=campaign.snapshot()
        self.assertEqual(after['forecasts'],before['forecasts']);self.assertEqual(after['revision_count'],1)

    def test_future_receipts_and_unverified_calendar_cannot_issue_forecasts(self):
        campaign=self.create();anchor=self.bar(self.validation[0],0)
        campaign.observe([anchor],now=anchor.received_at,calendar_verified=False)
        self.assertFalse(campaign.snapshot()['forecasts'])
        with self.assertRaises(self.m.CampaignError):campaign.observe([anchor],now=anchor.ends,calendar_verified=True)

    def test_validation_freezes_one_candidate_before_selected_only_test_and_completion(self):
        campaign=self.create();self.finish_validation(campaign)
        selected=campaign.snapshot()['selected'];self.assertIsNotNone(selected)
        self.feed_day(campaign,self.test[0])
        campaign.observe([],now=self.calendar.sessions[self.test[-1]].closes+timedelta(seconds=3601),calendar_verified=True)
        state=campaign.snapshot();test_forecasts=[p for p in state['forecasts'].values() if p['phase']=='test']
        self.assertEqual(len(test_forecasts),3);self.assertEqual({p['candidate_id'] for p in test_forecasts},{selected})
        self.assertTrue(self.registry.load(campaign.identity)['completed'])
        self.assertFalse(state['strategy_accepted'])

    def test_missing_data_or_missed_selection_deadline_never_uses_test_for_selection(self):
        campaign=self.create();session=self.calendar.sessions[self.validation[0]]
        anchor=self.bar(session.day,0);campaign.observe([anchor],now=anchor.received_at,calendar_verified=True)
        campaign.observe([],now=session.closes+timedelta(seconds=3601),calendar_verified=True)
        self.assertIsNone(campaign.snapshot()['selected'])
        self.assertFalse(self.registry.load(campaign.identity)['completed'])

    def test_missed_pretest_selection_deadline_blocks_all_test_forecasts(self):
        campaign=self.create();self.feed_day(campaign,self.validation[0])
        target=self.bar(self.test[0],0)
        campaign.observe([target],now=target.received_at,calendar_verified=True)
        self.assertIsNone(campaign.snapshot()['selected'])
        self.assertFalse(any(p['phase']=='test' for p in campaign.snapshot()['forecasts'].values()))

    def test_corrupt_archive_is_rejected_instead_of_initializing_a_new_history(self):
        campaign=self.create()
        with closing(sqlite3.connect(campaign.directory/'forecasts.sqlite')) as db:
            with db:db.execute("UPDATE observations SET digest=? WHERE seq=1",('0'*64,))
        with self.assertRaises(self.m.CampaignError):campaign.snapshot()

    def test_receipt_available_at_deadline_is_ingested_before_phase_finalization(self):
        campaign=self.create();day=self.validation[0]
        for i in range(3):
            bar=self.bar(day,i);campaign.observe([bar],now=bar.received_at,calendar_verified=True)
        deadline=self.calendar.sessions[day].closes+timedelta(seconds=3600)
        last=self.bar(day,3,received=deadline)
        campaign.observe([last],now=deadline+timedelta(seconds=1),calendar_verified=True)
        self.assertIsNotNone(campaign.snapshot()['selected'])

    def test_market_campaign_requires_retained_matching_daily_calendar_proof(self):
        campaign=self.create(domain='us_equity');day=self.validation[0];bar=self.bar(day,0)
        campaign.observe([bar],now=bar.received_at,calendar_verified=True)
        self.assertFalse(campaign.snapshot()['forecasts'])
        session=self.calendar.sessions[day];one=Calendar([session],start=day.isoformat(),end=day.isoformat())
        self.assertTrue(campaign.record_calendar(one,received_at=bar.received_at))
        campaign.observe([bar],now=bar.received_at,calendar_verified=True)
        self.assertEqual(len(campaign.snapshot()['forecasts']),2)

    def test_phase_commit_recovers_its_existing_report_before_ingesting_new_context(self):
        campaign=self.create();self.feed_day(campaign,self.validation[0])
        deadline=self.calendar.sessions[self.validation[0]].closes+timedelta(seconds=3601)
        with patch.object(campaign.registry,'select',side_effect=RuntimeError('injected interruption')):
            with self.assertRaises(self.m.CampaignError):campaign.observe([],now=deadline,calendar_verified=True)
        context=self.bar(self.context[-1],0)
        campaign.observe([context],now=context.received_at,calendar_verified=True)
        self.assertIsNotNone(campaign.snapshot()['selected'])

    def test_pre_cutoff_registry_selection_recovers_after_test_opens_without_new_selection(self):
        campaign=self.create();self.feed_day(campaign,self.validation[0]);append=campaign._append
        def interrupted(kind,now,**body):
            if kind=='phase_result':raise RuntimeError('injected interruption')
            return append(kind,now,**body)
        deadline=self.calendar.sessions[self.validation[0]].closes+timedelta(seconds=3601)
        with patch.object(campaign,'_append',side_effect=interrupted):
            with self.assertRaises(self.m.CampaignError):campaign.observe([],now=deadline,calendar_verified=True)
        selected=self.registry.load(campaign.identity)['selected'];self.assertIsNotNone(selected)
        reopened=self.m.open_campaign(campaign.directory,campaign.configuration_sha256)
        reopened.observe([],now=self.calendar.sessions[self.test[0]].opens+timedelta(seconds=1),calendar_verified=True)
        self.assertEqual(reopened.snapshot()['selected'],selected)

    def test_a_different_feed_cannot_enter_the_frozen_iex_campaign(self):
        campaign=self.create();bar=replace(self.bar(self.validation[0],0),feed='sip')
        with self.assertRaises(self.m.CampaignError):campaign.observe([bar],now=bar.received_at,calendar_verified=True)
        self.assertFalse(campaign.snapshot()['bars'])


if __name__=='__main__':unittest.main()
