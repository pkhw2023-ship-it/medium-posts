| Trigger | Detection mechanism | Re-embed behavior | Latency to detect |
|---|---|---|---|
| Model upgrade (S → B) | Key column `model_version` differs | All keys miss; full re-embed | O(1) per lookup |
| Preprocessing change (224 → 336) | Key column `preproc_version` differs | All keys miss; full re-embed | O(1) per lookup |
| Content edit (1 byte changed) | sha256 differs | Edited file misses; re-embeds | O(1) per lookup |
| Stored vector corruption | **NONE — schema does not checksum the vec column** | Returns corrupt vector silently | Requires explicit re-encode audit |
