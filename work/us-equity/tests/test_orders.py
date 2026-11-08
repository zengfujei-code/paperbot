from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import datetime,timedelta,timezone
from decimal import Decimal
import json
import inspect
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
from paperbot_us.data import Calendar,Session
from paperbot_us.risk import OrderIntent,Portfolio,Quote,RiskState,RiskLimits,Position,PendingOrder


class OrderLedgerTests(unittest.TestCase):
    def setUp(self):
        from paperbot_us.orders import OrderLedger,LedgerError
        self.Ledger,self.Error=OrderLedger,LedgerError
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'orders.sqlite'
        self.ledger=self.Ledger(self.path);self.ledger.initialize()
        self.now=datetime(2026,10,23,15,tzinfo=timezone.utc)
        self.calendar=Calendar([Session.from_alpaca({'date':'2026-10-23','open':'09:30','close':'16:00'})],start='2026-10-23',end='2026-10-23')
        self.intent=OrderIntent('a'*64,'AAPL','buy',5,'150.1','145',self.now+timedelta(seconds=20))
        self.portfolio=Portfolio(self.now,'10000','10000','10000','10000')
        self.quote=Quote('AAPL','150','150.1',self.now,self.now)
        self.args=dict(portfolio=self.portfolio,quote=self.quote,state=RiskState(pause_entries=False),limits=RiskLimits(),calendar=self.calendar,now=self.now)
        self.broker_id='00000000-0000-4000-8000-000000000001'

    def admit(self,intent=None,**changes): return self.ledger.admit(intent or self.intent,**{**self.args,**changes})
    def prepare(self,**changes): return self.ledger.prepare_submission(self.intent.intent_id,**{**self.args,**changes})

    def observe(self,**changes):
        return self.ledger.observe(self.intent.intent_id,**{**dict(status='partially_filled',filled_quantity=3,
            average_fill_price_usd='149.8',broker_order_id=self.broker_id,
            broker_created_at=self.now,broker_event_at=self.now+timedelta(seconds=2),received_at=self.now+timedelta(seconds=3)),**changes})

    def test_admission_is_durable_and_same_intent_returns_same_record(self):
        first=self.admit()
        self.assertEqual(first['state'],'reserved')
        self.assertEqual(self.Ledger(self.path).load(self.intent.intent_id),first)
        self.assertEqual(self.admit(),first)
        self.assertEqual(self.ledger.verify()['events'],1)
        with self.assertRaises(self.Error): self.admit(replace(self.intent,quantity=4))

    def test_full_order_snapshot_includes_terminal_history_and_checks_older_events(self):
        self.assertTrue(hasattr(self.ledger,'all_orders'),'full verified accounting order snapshot required')
        self.admit();self.prepare();self.observe(status='filled',filled_quantity=5,average_fill_price_usd='150')
        values=self.ledger.all_orders();self.assertEqual(len(values),1);self.assertEqual(values[0]['state'],'filled')
        self.assertEqual(self.ledger.active(),[])
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE events SET digest='broken' WHERE sequence=1");db.commit()
        with self.assertRaises(self.Error): self.ledger.all_orders()

    def test_actual_broker_creation_is_required_persisted_and_immutable(self):
        self.assertIn('broker_created_at',inspect.signature(self.ledger.observe).parameters)
        self.admit();self.prepare();first=self.observe(broker_created_at=self.now)
        self.assertEqual(first['broker_created_at'],self.now.isoformat(timespec='microseconds'))
        with self.assertRaises(self.Error): self.observe(broker_created_at=self.now+timedelta(seconds=1))

    def test_risk_rejection_is_audited_and_cannot_prepare_submission(self):
        result=self.admit(state=RiskState())
        self.assertEqual(result['state'],'rejected_by_risk')
        self.assertIn('entries_paused',result['risk']['violations'])
        self.assertIsNone(self.prepare())
        self.assertEqual(self.ledger.verify()['events'],1)

    def test_concurrent_admissions_include_durable_reservations(self):
        barrier=threading.Barrier(2)
        def admit(identity):
            barrier.wait()
            return self.Ledger(self.path).admit(replace(self.intent,intent_id=identity,quantity=4),**self.args)
        with ThreadPoolExecutor(max_workers=2) as workers:
            results=list(workers.map(admit,['a'*64,'b'*64]))
        self.assertEqual(sorted(r['state'] for r in results),['rejected_by_risk','reserved'])
        rejected=next(r for r in results if r['state']=='rejected_by_risk')
        self.assertIn('symbol_exposure_limit',rejected['risk']['violations'])
        self.assertEqual(self.ledger.verify()['orders'],2)

    def test_submission_is_marked_unknown_durably_before_io_and_claimed_once(self):
        self.admit()
        ticket=self.prepare()
        self.assertEqual(ticket['state'],'submission_unknown')
        self.assertEqual(self.Ledger(self.path).load(self.intent.intent_id)['state'],'submission_unknown')
        self.assertIsNone(self.prepare())
        self.assertEqual(self.ledger.verify()['events'],2)

    def test_prepare_rechecks_risk_and_expiration_before_any_submission(self):
        self.admit()
        self.assertIsNone(self.prepare(state=RiskState(halt_orders=True)))
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'rejected_by_risk')
        other=replace(self.intent,intent_id='b'*64)
        self.admit(other)
        self.assertIsNone(self.ledger.prepare_submission(other.intent_id,**{**self.args,'now':self.now+timedelta(seconds=25)}))
        self.assertEqual(self.ledger.load(other.intent_id)['state'],'expired')

    def test_partial_fill_duplicate_and_old_events_preserve_cumulative_economics(self):
        self.admit();self.prepare();first=self.observe()
        self.assertEqual(first['filled_quantity'],3)
        self.assertEqual(Decimal(first['filled_notional_usd']),Decimal('449.4'))
        self.assertEqual(self.observe(),first)
        stale=self.observe(status='open',filled_quantity=0,average_fill_price_usd=None,
                           broker_event_at=self.now+timedelta(seconds=1),received_at=self.now+timedelta(seconds=4))
        self.assertEqual(stale,first)
        self.assertEqual(self.ledger.verify()['events'],3)

    def test_broker_unknown_result_keeps_reservations_and_blocks_new_entries(self):
        self.admit();self.prepare()
        result=self.admit(replace(self.intent,intent_id='b'*64,quantity=1))
        self.assertEqual(result['state'],'rejected_by_risk')
        self.assertIn('account_unreconciled',result['risk']['violations'])

    def test_partial_remaining_quantity_is_merged_once_with_broker_book(self):
        self.admit();self.prepare();self.observe()
        at=self.now+timedelta(seconds=4)
        account=Portfolio(at,'9550.6','10000.6','10000','10000.6',
            positions=(Position('AAPL',3,'150',at),),
            pending_orders=(PendingOrder(self.intent.intent_id,'AAPL','buy',2,'150.1'),))
        result=self.admit(replace(self.intent,intent_id='b'*64,quantity=3),portfolio=account,
                          quote=replace(self.quote,as_of=at,received_at=at),now=at)
        self.assertIn('symbol_exposure_limit',result['risk']['violations'])
        self.assertEqual(len(result['context']['portfolio']['pending_orders']),1)

    def test_rejected_storage_write_rolls_back_order_and_its_audit_event(self):
        with patch.object(self.ledger,'_persist',side_effect=OSError('disk full fixture')):
            with self.assertRaises(self.Error): self.admit()
        self.assertEqual(self.ledger.verify(),dict(orders=0,events=0,integrity_verified=True))

    def test_invalid_price_or_counter_regression_never_releases_a_reservation(self):
        self.admit();self.prepare();self.observe()
        for changes in ({'filled_quantity':2},{'average_fill_price_usd':'151'},
                        {'filled_quantity':5,'status':'partially_filled'}):
            with self.subTest(changes=changes),self.assertRaises(self.Error): self.observe(**changes)
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'partially_filled')

    def test_changed_payload_or_missing_database_cannot_be_used_silently(self):
        self.admit()
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE orders SET payload='{}'");db.commit()
        with self.assertRaises(self.Error): self.ledger.load(self.intent.intent_id)
        absent=Path(self.temp.name)/'absent.sqlite'
        with self.assertRaises(self.Error): self.Ledger(absent).load(self.intent.intent_id)
        self.assertFalse(absent.exists())

    def test_each_incremental_fill_must_respect_its_limit(self):
        self.admit();self.prepare();first=self.observe()
        with self.assertRaises(self.Error):
            self.observe(filled_quantity=4,average_fill_price_usd='150',
                         broker_event_at=self.now+timedelta(seconds=4),received_at=self.now+timedelta(seconds=5))
        self.assertEqual(self.ledger.load(self.intent.intent_id),first)
        # The symmetric sell case also has a plausible cumulative average but
        # an impossible fourth fill below the sell limit.
        sell=replace(self.intent,intent_id='b'*64,side='sell',limit_price_usd=Decimal('149.9'),stop_price_usd=None)
        account=Portfolio(self.now,'9250','10000','10000','10000',positions=(Position('AAPL',5,'150',self.now),))
        args={**self.args,'portfolio':account}
        # The earlier buy has unknown account consequences; isolate the sell
        # lifecycle in a separate durable ledger rather than bypassing that veto.
        other=self.Ledger(Path(self.temp.name)/'sell.sqlite');other.initialize()
        other.admit(sell,**args);other.prepare_submission(sell.intent_id,**args)
        changes=dict(status='partially_filled',filled_quantity=3,average_fill_price_usd='150.2',
                     broker_order_id=self.broker_id,broker_created_at=self.now,broker_event_at=self.now+timedelta(seconds=2),
                     received_at=self.now+timedelta(seconds=3))
        other.observe(sell.intent_id,**changes)
        with self.assertRaises(self.Error):
            other.observe(sell.intent_id,**{**changes,'filled_quantity':4,'average_fill_price_usd':'150',
                          'broker_event_at':self.now+timedelta(seconds=4),'received_at':self.now+timedelta(seconds=5)})

    def test_event_before_submission_never_releases_unknown_reservation(self):
        self.admit()
        claimed=self.prepare(now=self.now+timedelta(seconds=1))
        for event_at in (self.now-timedelta(days=1),self.now):
            with self.subTest(event_at=event_at),self.assertRaises(self.Error):
                self.observe(status='canceled',filled_quantity=0,average_fill_price_usd=None,
                             broker_event_at=event_at)
            self.assertEqual(self.ledger.load(self.intent.intent_id),claimed)
            self.assertEqual(len(self.ledger.active()),1)

    def test_existing_broker_order_prevents_new_submission_and_keeps_reservation(self):
        self.admit()
        account=replace(self.portfolio,pending_orders=(PendingOrder(self.intent.intent_id,'AAPL','buy',5,'150.1'),))
        self.assertIsNone(self.prepare(portfolio=account))
        record=self.ledger.load(self.intent.intent_id)
        self.assertEqual(record['state'],'submission_unknown')
        self.assertIn('broker_order_already_exists',record['context']['reconciliation_findings'])
        self.assertEqual(len(self.ledger.active()),1)
        self.assertIsNone(self.prepare(portfolio=account))

    def test_expired_local_intent_cannot_release_an_existing_broker_order(self):
        self.admit()
        account=replace(self.portfolio,pending_orders=(PendingOrder(self.intent.intent_id,'AAPL','buy',5,'150.1'),))
        self.assertIsNone(self.prepare(portfolio=account,now=self.now+timedelta(seconds=25)))
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'submission_unknown')
        self.assertEqual(len(self.ledger.active()),1)

    def test_identical_newer_broker_state_advances_watermark_before_old_terminal_event(self):
        self.admit();self.prepare()
        first=self.observe(status='open',filled_quantity=0,average_fill_price_usd=None)
        newer=self.observe(status='open',filled_quantity=0,average_fill_price_usd=None,
                           broker_event_at=self.now+timedelta(seconds=10),received_at=self.now+timedelta(seconds=11))
        self.assertGreater(newer['broker_updated_at'],first['broker_updated_at'])
        stale=self.observe(status='canceled',filled_quantity=0,average_fill_price_usd=None,
                           broker_event_at=self.now+timedelta(seconds=5),received_at=self.now+timedelta(seconds=12))
        self.assertEqual(stale,newer)
        self.assertEqual(len(self.ledger.active()),1)


if __name__=='__main__': unittest.main()
