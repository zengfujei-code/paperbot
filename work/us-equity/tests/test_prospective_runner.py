from datetime import timedelta
import importlib,json,unittest
import test_prospective as fixtures
from paperbot_us.data import Calendar


class ProspectiveRunnerTests(unittest.TestCase):
    setUp=fixtures.ProspectiveTests.setUp
    create=fixtures.ProspectiveTests.create
    bar=fixtures.ProspectiveTests.bar

    def fake(self,campaign,day,rows,*,calendar_error=False,data_error=False):
        owner=self
        class Client:
            def __init__(self):self.calls=[]
            def calendar(self,start,end):
                self.calls.append('calendar')
                if calendar_error:raise RuntimeError('SENSITIVE_TEST_MARKER')
                owner.assertEqual(start,end)
                s=campaign.calendar.sessions.get(day)
                return Calendar([] if s is None else [s],start=start,end=end)
            def bars(self,symbols,*,start,end):
                self.calls.append('bars');owner.assertEqual(symbols,['AAPL'])
                owner.assertLess(start,end)
                if data_error:raise RuntimeError('SENSITIVE_TEST_MARKER')
                return {'AAPL':rows}
            def account(self):raise AssertionError('account access forbidden')
            def order(self,*args,**kwargs):raise AssertionError('order access forbidden')
        return Client()

    def test_active_collector_uses_only_calendar_and_bars_and_keeps_daily_proof(self):
        runner=importlib.import_module('paperbot_us.prospective_runner')
        campaign=self.create(domain='us_equity');day=self.validation[0];bar=self.bar(day,0)
        client=self.fake(campaign,day,[bar])
        result=runner.collect_once(campaign,client=client,clock=lambda:bar.received_at)
        self.assertEqual(client.calls,['calendar','bars']);self.assertEqual(result['forecast_count'],2)
        self.assertFalse(result['execution_enabled']);self.assertTrue(result['research_only'])
        runner.collect_once(campaign,client=client,clock=lambda:bar.received_at+timedelta(seconds=10))
        self.assertEqual(client.calls,['calendar','bars','bars'])
        self.assertEqual(len(campaign.snapshot()['forecasts']),2)

    def test_before_open_does_not_request_future_bars_or_create_forecasts(self):
        runner=importlib.import_module('paperbot_us.prospective_runner')
        campaign=self.create(domain='us_equity');day=self.context[0];client=self.fake(campaign,day,[])
        result=runner.collect_once(campaign,client=client,clock=lambda:self.now)
        self.assertEqual(client.calls,['calendar']);self.assertEqual(result['forecast_count'],0)

    def test_calendar_failure_is_sanitized_and_does_not_touch_account_or_orders(self):
        runner=importlib.import_module('paperbot_us.prospective_runner')
        campaign=self.create(domain='us_equity');day=self.validation[0];bar=self.bar(day,0)
        client=self.fake(campaign,day,[bar],calendar_error=True)
        result=runner.collect_once(campaign,client=client,clock=lambda:bar.received_at)
        self.assertEqual(client.calls,['calendar']);self.assertEqual(result['status'],'calendar_unavailable')
        self.assertNotIn('SENSITIVE_TEST_MARKER',json.dumps(result))

    def test_data_failure_does_not_publish_fabricated_forecasts(self):
        runner=importlib.import_module('paperbot_us.prospective_runner')
        campaign=self.create(domain='us_equity');day=self.validation[0];bar=self.bar(day,0)
        client=self.fake(campaign,day,[bar],data_error=True)
        result=runner.collect_once(campaign,client=client,clock=lambda:bar.received_at)
        self.assertEqual(result['status'],'data_unavailable');self.assertEqual(result['forecast_count'],0)
        self.assertNotIn('SENSITIVE_TEST_MARKER',json.dumps(result))

    def test_open_incomplete_bar_is_never_processed_as_a_closed_outcome(self):
        runner=importlib.import_module('paperbot_us.prospective_runner')
        campaign=self.create(domain='us_equity');day=self.validation[0];closed=self.bar(day,0)
        from dataclasses import replace
        incomplete=replace(self.bar(day,1),received_at=closed.received_at)
        client=self.fake(campaign,day,[closed,incomplete])
        result=runner.collect_once(campaign,client=client,clock=lambda:closed.received_at)
        self.assertEqual(result['forecast_count'],2)
        self.assertFalse(any(p['outcome_return'] is not None for p in campaign.snapshot()['forecasts'].values()))

    def test_inclusive_end_postmarket_bar_does_not_poison_regular_session_backfill(self):
        runner=importlib.import_module('paperbot_us.prospective_runner')
        campaign=self.create(domain='us_equity');day=self.context[0];session=self.calendar.sessions[day]
        now=session.closes+timedelta(minutes=16)
        rows=[self.bar(day,i,received=now) for i in range(5)]
        client=self.fake(campaign,day,rows)
        result=runner.collect_once(campaign,client=client,clock=lambda:now)
        self.assertEqual(result['status'],'awaiting_session')
        self.assertEqual(len(campaign.snapshot()['bars']),4)


if __name__=='__main__':unittest.main()
