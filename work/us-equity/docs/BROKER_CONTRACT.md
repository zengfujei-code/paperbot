# Paper gateway contract — checked 2026-10-24

This interface is separate from the production GET-only collector. It is not
deployed or permitted to bypass independent risk, the durable ledger, account
reconciliation or strategy acceptance.

- Only `https://paper-api.alpaca.markets` is supported. Paper uses separate keys;
  IEX is the initial Paper Only data entitlement. Broker paper fills do not
  measure real market impact, queue position or latency costs. Those remain
  independent research assumptions. [Official paper specification](https://docs.alpaca.markets/us/docs/paper-trading)
- An equity order can carry a unique client ID. Our deterministic 44-character
  ID fits the documented 128-character bound. Initial requests are whole-share
  limit, DAY, regular-session only and simple order class. No market fallback,
  shorting, fractional notional, extended hours or live host is exposed.
  [Official create endpoint](https://docs.alpaca.markets/us/reference/postorder)
- Unknown submission outcomes require lookup using the same client ID. The
  adapter makes one POST attempt; it never infers that a timeout means rejection
  and never retries a mutation automatically. A lookup miss is not permission
  to resend an unknown intent. [Official lookup endpoint](https://docs.alpaca.markets/us/reference/getorderbyclientorderid)
- Open-order reads request the documented maximum of 500. A full response is
  treated as possibly truncated and blocks reconciliation. Unknown orders and
  unsupported order classes/statuses block entries instead of being ignored.
  [Official order listing](https://docs.alpaca.markets/us/v1.1/reference/getallorders-1)
- Cancellation acceptance is a request acknowledgement, not a terminal broker
  state. Keep reservations until confirmed cancellation/fill. Failed or unknown
  cancellation must be reconciled. [Official cancel endpoint](https://docs.alpaca.markets/us/v1.1/reference/deleteorderbyorderid-1)
- Price ticks are checked without silent rounding: two decimals at prices of
  at least USD 1, four below USD 1. Source status timestamps and cumulative fills
  must be normalized and checked against the intended order.
  [Official equity order rules](https://docs.alpaca.markets/us/docs/orders-at-alpaca)
- Actual fills need activity-level reconciliation, not only cumulative averages.
  The activity feed supplies individual fill quantity/price/time and pagination.
  [Official account activities](https://docs.alpaca.markets/us/docs/account-activities)

An intent's modeled stop in `risk.py` is **not** a broker protective order. A
simple limit gateway does not implement stop monitoring or conditional children.
That protection, fill/account reconciliation and failure recovery are blockers
to automated paper execution, not silently completed by this adapter.

Account/position receipts are observations, not proven quote source timestamps.
Risk valuation needs independently fresh market quotes. Only normalized allowlist
fields enter private evidence; account identifiers and raw bodies are discarded.
Public status receives a separate bounded projection without controls or secrets.
