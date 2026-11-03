# Verification archive for the manuscript "Perfect state transfer from Hermitian weighing matrices: mixed Cayley graphs over groups with an abelian subgroup of index 2"

`author_scripts_original/` is the author's supplementary archive, byte-for-byte unchanged (its `SHA256SUMS` verifies). Its own README remains the description of those scripts, with two statements superseded by `independent_checks`: "the unrestricted enumeration on Dic_6 did not finish" (an unrestricted exact enumeration finishes in about 30 seconds) and "the 162-triple check of Theorem gd:k2 is not contained" (it is: `independent_checks/t3_k2_exact.py`).

`independent_checks/` is a second set of programs with logs, `run_all.sh`, `requirements.txt` and a README (claim -> script -> log table). Since v12 it also contains the enumeration of the weight-4 weighing sets for non-cyclic A of order at most 16 (`search_noncyclic_k4.py`, `validate_prune.py`) and, since v13, a third implementation of it (`noncyclic_weight4_third_check.py`).

All scripts of the author's archive were re-run by the auditor; the new logs are identical to the author's `expected_output_*` files (timings ignored); see `../audit/rerun_of_author_scripts`. Both sets of programs agree with every number of Tables `tab:nn` and `tab:ab` of the manuscript.
