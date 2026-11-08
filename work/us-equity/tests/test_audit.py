import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import dataclass,replace
from datetime import datetime,timezone,timedelta
from pathlib import Path
from paperbot_us.alpaca import Asset,UniverseSnapshot
from paperbot_us.data import Calendar,Session,Snapshot
from paperbot_us.scanner import ScanPolicy


class AuditTests(unittest.TestCase):
    def setUp(self):
        from paperbot_us.audit import ScanArchive,ArchiveError,make_scan_record,replay
        self.Archive,self.Error,self.make,self.replay=ScanArchive,ArchiveError,make_scan_record,replay
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'scans.sqlite'
        now=datetime(2026,10,23,15,tzinfo=timezone.utc)
        self.inputs=dict(universe=UniverseSnapshot((Asset('public-instrument','AAPL','NASDAQ',True),),now,0),
            snapshots={'AAPL':Snapshot('AAPL',100,now-timedelta(seconds=1),99.99,100.01,10,10,
                        now-timedelta(seconds=2),100000,now.date(),now)},
            calendar=Calendar([Session.from_alpaca({'date':'2026-10-23','open':'09:30','close':'16:00'})],
                              start='2026-10-23',end='2026-10-23'),as_of=now,policy=ScanPolicy())

    def test_durable_roundtrip_and_offline_replay(self):
        record=self.make(**self.inputs)
        archive=self.Archive(self.path)
        self.assertTrue(archive.append(record))
        loaded=archive.load(record['run_id'])
        self.assertEqual(loaded,record)
        self.assertEqual(self.replay(loaded),record['result'])
        self.assertEqual(loaded['inputs']['snapshots']['AAPL']['feed'],'iex')
        self.assertIn('code_sha256',record)

    def test_identical_record_is_idempotent_and_order_independent(self):
        record=self.make(**self.inputs)
        archive=self.Archive(self.path)
        archive.append(record)
        self.assertFalse(archive.append(json.loads(json.dumps(record))))
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM scans').fetchone()[0],1)

    def test_tampering_and_version_changes_are_rejected(self):
        record=self.make(**self.inputs)
        archive=self.Archive(self.path)
        archive.append(record)
        changed=json.loads(json.dumps(record))
        changed['result']['status']='fake'
        with self.assertRaises(self.Error): archive.append(changed)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('UPDATE scans SET payload=?',(json.dumps(changed),))
            db.commit()
        with self.assertRaises(self.Error): archive.load(record['run_id'])
        record['code_sha256']='0'*64
        with self.assertRaises(self.Error): self.replay(record)

    def test_read_of_missing_archive_does_not_create_database(self):
        archive=self.Archive(self.path)
        with self.assertRaises(self.Error): archive.load('0'*64)
        self.assertFalse(self.path.exists())

    def test_missing_data_is_preserved_as_a_rejection(self):
        self.inputs['snapshots']={}
        record=self.make(**self.inputs)
        self.assertEqual(self.replay(record)['rejected'],{'AAPL':'snapshot_missing'})
        self.assertEqual(record['result']['candidates'],[])

    def test_session_subclass_metadata_is_never_archived(self):
        @dataclass(frozen=True)
        class ExtraSession(Session):
            diagnostic: str='TEST_ONLY_PRIVATE_METADATA'
        old=self.inputs['calendar'].sessions[self.inputs['as_of'].date()]
        extended=ExtraSession(old.day,old.opens,old.closes)
        self.inputs['calendar']=Calendar([extended],start='2026-10-23',end='2026-10-23')
        record=self.make(**self.inputs)
        self.assertNotIn('TEST_ONLY_PRIVATE_METADATA',json.dumps(record))
        self.assertEqual(self.replay(record),record['result'])

    def test_full_observed_universe_is_stored_once_and_linked_to_batch(self):
        full=replace(self.inputs['universe'],assets=(self.inputs['universe'].assets[0],Asset('second','MSFT','NASDAQ',True)))
        record=self.make(**self.inputs,full_universe=full,batch_offset=0)
        archive=self.Archive(self.path)
        archive.append(record,full_universe=full)
        loaded=archive.load(record['run_id'])
        pool=archive.load_universe(loaded['inputs']['selection']['pool_sha256'])
        self.assertEqual([a['symbol'] for a in pool['assets']],['AAPL','MSFT'])
        self.assertEqual(loaded['inputs']['selection']['total'],2)
        self.assertFalse(archive.append(record,full_universe=full))
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM universes').fetchone()[0],1)
        with self.assertRaises(self.Error):
            self.make(**self.inputs,full_universe=full,batch_offset=1)

    def test_tampered_pool_blocks_loading_its_batches(self):
        record=self.make(**self.inputs)
        archive=self.Archive(self.path)
        archive.append(record)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE universes SET payload='{}'")
            db.commit()
        with self.assertRaises(self.Error): archive.load(record['run_id'])


if __name__=='__main__': unittest.main()
