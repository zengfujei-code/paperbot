"""Private standard-library worker: original empty-book evidence only."""
from contextlib import closing,contextmanager,ExitStack
from datetime import datetime
from decimal import Decimal
import hashlib
import importlib.util
import os
from pathlib import Path
import sqlite3
import sys

spec=importlib.util.spec_from_file_location('_paperbot_book_guards',Path(__file__).with_name('replay_worker.py'))
guards=importlib.util.module_from_spec(spec); spec.loader.exec_module(guards)
p=guards.protocol


def version():
    digest=hashlib.sha256()
    for name in ('book_replay.py','book_worker.py','replay_bridge.py','replay_worker.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


def exact(value,keys): guards.exact(value,set(keys.split()))


def empty_result(result):
    if (result.reconciled is not True or result.violations or result.inventory
            or result.expected_cash_usd!=10000 or result.equity_usd!=10000
            or any(getattr(result,k)!=0 for k in ('profit_usd','realized_usd','unrealized_usd','fees_usd'))):
        raise ValueError('original account is not pristine')


@contextmanager
def readonly(name):
    path=Path(name)
    if (not path.is_absolute() or str(path.resolve())!=str(path) or path.is_symlink()
            or not path.is_file() or path.stat().st_size>64*1024*1024 or path.stat().st_nlink!=1):
        raise ValueError('bounded canonical original database required')
    with path.open('rb') as handle: header=handle.read(20)
    if header[:16]!=b'SQLite format 3\0' or header[18:20]!=b'\x01\x01': raise ValueError('DELETE database required')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5,isolation_level=None)) as db:
        db.row_factory=sqlite3.Row; db.execute('PRAGMA query_only=ON'); db.set_authorizer(guards.authorize)
        db.execute('BEGIN'); db.execute('SELECT name FROM sqlite_master LIMIT 1').fetchone()
        if db.execute('PRAGMA journal_mode').fetchone()[0]!='delete': raise ValueError('DELETE database required')
        yield db
        db.commit()


def account_shape(raw,accounting,*,initial=False):
    exact(raw,'received_at cash_usd equity_usd blocked account_sha256'+('' if initial else ' positions'))
    p.identity(raw['account_sha256']); guards.stamp(raw['received_at'])
    if (not initial and raw['positions']!=[]) or raw['blocked'] is not False: raise ValueError('empty active account required')
    return accounting.AccountSnapshot(**{**raw,'positions':()})


def window_shape(raw,accounting,*,activities=True):
    exact(raw,'after until received_at complete'+(' activities' if activities else ''))
    if raw['complete'] is not True or (activities and raw['activities']!=[]): raise ValueError('complete empty history required')
    return accounting.ActivityWindow(**{**raw,'activities':()})


def original_proof(db,monitor,bootstrap,accounting,anchor):
    _,count,head=monitor.ObservationArchive('unused')._verify_chain(db)
    if count!=1 or head!=anchor.evidence_sha256: raise ValueError('original initial proof required')
    row=db.execute('SELECT payload FROM observations WHERE seq=1').fetchone()
    raw=p.strict_json(row[0])
    exact(raw,'kind schema generated_at code_sha256 mode capital_usd anchor_as_of accounts '
              'empty_positions_received_at empty_history_received_at window ledger_verified_empty '
              'paper_started_at execution_enabled observing_started_at')
    if (raw['kind']!='bootstrap_proof' or type(raw['schema']) is not int or raw['schema']!=1
            or raw['code_sha256']!=bootstrap._code_hash() or raw['mode']!='paper'
            or type(raw['capital_usd']) is not int or raw['capital_usd']!=10000
            or raw['anchor_as_of']!=anchor.as_of.isoformat() or raw['ledger_verified_empty'] is not True
            or raw['paper_started_at'] is not None or raw['execution_enabled'] is not False
            or raw['observing_started_at']!=raw['generated_at']): raise ValueError('initial proof binding')
    guards.number(raw['generated_at'])
    generated=datetime.fromtimestamp(raw['generated_at'],tz=anchor.as_of.tzinfo)
    if not 0<=(generated-anchor.as_of).total_seconds()<=30: raise ValueError('initial proof freshness')
    if type(raw['accounts']) is not list or len(raw['accounts'])!=2: raise ValueError('initial account reads')
    accounts=[account_shape(a,accounting,initial=True) for a in raw['accounts']]
    if (accounts[0].received_at!=anchor.as_of or any(a.account_sha256!=anchor.account_sha256
            or a.cash_usd!=10000 or a.equity_usd!=10000 for a in accounts)
            or not accounts[0].received_at<=accounts[1].received_at<=generated): raise ValueError('initial account identity')
    window=window_shape(raw['window'],accounting)
    empty_result(accounting.reconcile(anchor=anchor,window=window,orders=(),account=accounts[-1],now=generated))
    for key in ('empty_positions_received_at','empty_history_received_at'):
        values=raw[key]
        if type(values) is not list or len(values)!=2: raise ValueError('initial empty read receipts')
        for value in values:
            guards.stamp(value)
            if not anchor.as_of<=accounting.utc_time(value)<=generated: raise ValueError('initial read time')
    return dict(records=1,digest=head)


def main():
    request=sys.stdin.buffer.read(32769)
    if len(request)>32768: raise ValueError('request budget')
    value=p.strict_json(request)
    exact(value,'schema root manifest release_sha256 validator_sha256 book baseline ledger timezone_root timezone_sha256')
    if type(value['schema']) is not int or value['schema']!=1 or value['validator_sha256']!=version():
        raise ValueError('validator binding')
    manifest=value['manifest']; exact(manifest,'schema package_sha256 files')
    if type(manifest['schema']) is not int or manifest['schema']!=1: raise ValueError('source schema')
    release=p.PinnedRelease(value['root'],files=manifest['files'],package_sha256=manifest['package_sha256'])
    if release.verify()!=value['release_sha256']: raise ValueError('source binding')
    import zoneinfo
    if p.timezone_digest(value['timezone_root'])!=value['timezone_sha256']: raise ValueError('timezone binding')
    zoneinfo.reset_tzpath([value['timezone_root']])
    if len({value[k] for k in ('book','baseline','ledger')})!=3: raise ValueError('separate evidence files')
    sys.addaudithook(guards.deny_side_effects); sys.path.insert(0,str(release.root))
    from paperbot_us import accountbook,accounting,bootstrap,monitor,orders
    with ExitStack() as stack:
        book_db,proof_db,ledger_db=[stack.enter_context(readonly(value[k])) for k in ('book','baseline','ledger')]
        book=accountbook.AccountBook(value['book'])
        anchor,events,_,latest,count,head=book._validate(book_db)
        if events or not 1<=count<=100 or latest is None: raise ValueError('bounded pristine original snapshots required')
        anchor_raw=p.strict_json(book_db.execute('SELECT anchor FROM metadata').fetchone()[0])
        exact(anchor_raw,'as_of cash_usd evidence_sha256 account_sha256')
        p.identity(anchor_raw['account_sha256']); p.identity(anchor_raw['evidence_sha256'])
        baseline=original_proof(proof_db,monitor,bootstrap,accounting,anchor)
        if [row[0] for row in ledger_db.execute('SELECT version FROM metadata')]!=[1]: raise ValueError('original ledger schema')
        ledger=orders.OrderLedger(value['ledger'])._verify(ledger_db)
        if ledger['orders'] or ledger['events']: raise ValueError('original ledger is not empty')
        last_as_of=anchor.as_of
        for row in book_db.execute('SELECT payload FROM snapshots ORDER BY sequence'):
            record=p.strict_json(row[0])
            exact(record,'schema anchor_sha256 code_sha256 activity_count activities_sha256 inputs result')
            if record['code_sha256']!=accountbook.code_hash() or type(record['activity_count']) is not int or record['activity_count']!=0:
                raise ValueError('original snapshot version/activity')
            inputs=record['inputs']; exact(inputs,'window orders account now')
            if inputs['orders']!=[]: raise ValueError('original orders are not empty')
            window=window_shape(inputs['window'],accounting,activities=False)
            account=account_shape(inputs['account'],accounting)
            guards.stamp(inputs['now']); now=accounting.utc_time(inputs['now'])
            result=accounting.reconcile(anchor=anchor,window=window,orders=(),account=account,now=now)
            empty_result(result)
            expected=p.strict_json(accountbook._json(result))
            if record['result']!=expected: raise ValueError('original replay differs')
            last_as_of=now
        output=dict(schema=1,kind='pristine_book',release_sha256=release.digest,
            validator_sha256=version(),timezone_sha256=value['timezone_sha256'],anchor=anchor_raw,
            book=dict(snapshots=count,activities=0,digest=head,code_sha256=accountbook.code_hash(),
                      last_as_of=last_as_of.isoformat()),baseline=baseline,ledger=ledger,
            summary=dict(cash_usd='10000',equity_usd='10000',high_water_usd='10000',
                         never_traded=True,execution_enabled=False,paper_started_at=None))
        for key in ('book','baseline','ledger'):
            output[key]['file_sha256']=hashlib.sha256(p.regular_bytes(Path(value[key]),64*1024*1024)).hexdigest()
        if release.verify()!=value['release_sha256'] or version()!=value['validator_sha256']:
            raise ValueError('source changed during replay')
        if p.timezone_digest(value['timezone_root'])!=value['timezone_sha256']: raise ValueError('timezone changed')
    text=p.canonical(output)
    if len(text.encode())>16384: raise ValueError('proof budget')
    sys.stdout.buffer.write(text.encode()+b'\n'); sys.stdout.buffer.flush()


if __name__=='__main__':
    try: main()
    except BaseException:
        print('{"status":"original_pristine_book_unavailable"}',flush=True)
        raise SystemExit(1)
