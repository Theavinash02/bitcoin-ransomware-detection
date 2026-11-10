# Data

Everything under `data/raw/` and `data/processed/` is gitignored. `ransomdet download` fetches it again.

## Sources

| Source | File | Version | License |
| --- | --- | --- | --- |
| [BitcoinHeist (UCI #526)](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset) | `raw/BitcoinHeistData.csv` | UCI release, 2,916,697 rows | CC BY 4.0 |
| [Ransomwhere (Zenodo)](https://zenodo.org/records/13999026) | `raw/ransomwhere.json` | v1.1.0, published 2024-10-27, DOI 10.5281/zenodo.13999026, md5 8a7bff74ab0dea4186232bc131c5d40f | CC BY 4.0 |
| [Coin Metrics community API](https://docs.coinmetrics.io/packages/coin-metrics-community-data), `PriceUSD` daily | `raw/btc_usd.csv` | pulled 2026-11-09, 2010-07-18 to 2026-11-08 | CC BY-NC 4.0 |

Credit: Akcora, Li, Gel and Kantarcioglu, "BitcoinHeist: Topological Data Analysis for Ransomware Prediction on the Bitcoin Blockchain", IJCAI 2020 ([arXiv:1906.07852](https://arxiv.org/abs/1906.07852)). Cable, Gregory, Izhikevich and Durumeric, Ransomwhere. Coin Metrics for the price series.

`download` resolves the latest Ransomwhere version on Zenodo and writes its record id, version and checksum to `raw/ransomwhere_meta.json`.

## BitcoinHeist columns

The paper builds one graph per 24-hour window, cut in UTC-6. Edges moving less than 0.3 BTC are dropped first, and `white` addresses are capped at 1,000 per day. A "starter" transaction is one whose inputs come from outside that day's window. The four graph features below follow coins from starter transactions to the address within the same day.

| Column | Meaning |
| --- | --- |
| `address` | Bitcoin address |
| `year`, `day` | Year and day of year (1 = 1 January) of the window |
| `income` | Total satoshi sent to the address in that window (1 BTC = 1e8 satoshi) |
| `neighbors` | Number of transactions that have the address as an output in that window |
| `count` | Number of starter transactions connected to the address through a chain. A merge measure, not a transaction count |
| `weight` | Sum of the fractions of coins that start at starter transactions and reach the address. Grows when many inputs merge |
| `length` | Number of non-starter transactions on the longest chain ending at the address. Measures mixing rounds (capped at 144 by the window) |
| `looped` | Number of starter transactions connected to the address by more than one directed path. Captures split-and-merge patterns |
| `label` | Ransomware family (with a `montreal`, `princeton` or `padua` prefix for the study it came from) or `white` |

The old README called `count` the number of transactions and `neighbors` the number of connected addresses. Both were wrong.

What this means for the outside test: `income` and `neighbors` only need the payments into one address, so they can be rebuilt from Ransomwhere's transaction list. `count`, `weight`, `length` and `looped` need the whole day's graph and cannot.

## Caveats

- Class ratios are not real-world ones. Only up to 1,000 `white` addresses per day were sampled, out of hundreds of thousands active on the network.
- `white` means "not in the three ransomware lists", not "verified benign".
- Coin Metrics prices are UTC end-of-day rates and BitcoinHeist days are cut in UTC-6, a 6-hour mismatch.
