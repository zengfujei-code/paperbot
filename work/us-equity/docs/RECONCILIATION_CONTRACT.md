# Paper execution accounting and reconciliation

Continue P3 without changing production scanner/shadow source or BTC state.
Initial cash is exactly virtual USD 10,000 with an explicitly observed empty
account anchor. No history reset, fabricated initial position, margin, shorting
or implicit account funding. Data input and persistence use Python stdlib.

Use complete bounded activity windows from the fixed paper host, starting after
the anchor and ending at an explicit UTC time. Fetch all activity types, not
only fills: accept US-equity whole-share owned fills and nonpositive USD fees;
block deposits, withdrawals, dividends, corporate actions, corrections and other
unsupported events until a separately implemented policy can explain them.
Hash activity IDs; retain broker order UUID only in private evidence to match
our durable intent. Never retain account IDs, raw provider bodies or headers.
Pagination exhaustion, repeated IDs/tokens, future events and malformed values
are failures, not an empty complete history. Event time and receipt time differ;
late publication requires overlapping/full anchor-window retrieval and explicit
reconciliation against every known order's cumulative fills.

The pure accounting engine computes cash, long share quantities, average-cost
inventory, realized PnL and fees from normalized individual fills. It verifies
ownership, per-fill limit prices, cumulative quantities/notionals and the broker's
current cash/position marks/equity. Non-atomic REST snapshots may temporarily
disagree; disagreement blocks entries rather than inventing a correction.
Broker-reported accounting marks are distinct from independently fresh quote
marks used for conservative risk. Neither receipt time nor accounting balance
proves executable quote freshness. Stops and gap losses remain separate.

Persist immutable normalized activities once. A snapshot references the exact
activity prefix and normalized order/account inputs, code hash and result; keep
a digest chain and explicit anchor. Missing or conflicting previously observed
activity blocks a refresh. Commit before returning an accounting receipt to risk
or publication. Preserve source versions for offline replay. No schema migration
or production execution is part of writing these modules.

Acceptance: hand-calculated buy/partial-sell/fee oracle; duplicate/conflicting
events; missing/foreign/overfilled/limit-breaking fills; cash/position/equity
mismatch; stale and future receipts; partial pagination; late event recovery;
restart and historical replay; interrupted/disk-full commit; no order calls in
read/reconciliation; no private data in public projection. Independent review,
Linux tests and actual read-only paper evidence precede execution activation.

API facts checked 2026-10-24:
https://docs.alpaca.markets/us/docs/account-activities
https://docs.alpaca.markets/us/reference/getaccountactivities-2
https://docs.alpaca.markets/us/reference/getaccountactivitiesbyactivitytype-1
The feed paginates with the last activity ID; initial page size is 100. The API
filters activity creation time, so fill event time is not a completeness watermark.
