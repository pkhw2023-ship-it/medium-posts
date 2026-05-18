| Strategy | 30-day GPU seconds | 30-day GPU hours | Final hit rate | Lookup p50 (µs) | Correctness |
|---|---|---|---|---|---|
| no cache | 127.84 | 0.0355 | n/a | n/a | yes |
| **SHA256** | **29.27** | **0.0081** | **0.80** | **511.25** | **yes** |
| pHash (alone) | 4.26 | 0.0012 | 1.00 | (see SHA row) | NO — silent false hits |
| SHA + pHash prefilter | 29.27 | 0.0081 | ≥ 0.80 | (see SHA row) | yes (pHash is hint only) |
