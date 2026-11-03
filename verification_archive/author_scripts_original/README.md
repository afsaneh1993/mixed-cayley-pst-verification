# Supplementary verification scripts

Companion to the manuscript on perfect state transfer, mixed Cayley graphs and
group-developed Hermitian weighing matrices, and to its supplementary note (labels S1, S2, S3).

Environment used for all logs: Python 3.12.3, numpy 2.4.4, scipy 1.17.1, single core.
No other dependency, except `check_aut_nauty.py`, which needs `pynauty` (`pip install pynauty`; version 2.8.8.1 was used). Automorphism orders in the other scripts come from a backtracking search.

## Which script reproduces which statement

| statement in the paper / note | script | log |
|---|---|---|
| Prop. (D_n family): K^2=4I, spectrum, U(pi/2)=iR_c, abelian obstruction, \|Aut\| (mixed 2n; underlying undirected graph 1440, 256, 640, 1536 for n=6,8,10,12) | `verify_families.py` | `expected_output_verify_families.txt` |
| Prop. S1.1 (weight 2 on Dic_n, SD_{2^k}) and Cor. weight 4 on Dic_n, n=5..30 | `verify_families.py` | same |
| Counts 0, 0, 80 on Dic_3, Dic_4, Dic_5 (all subsets) | `enumerate_dicn_weight4.py --unrestricted` | `expected_output_enumerate_unrestricted_3_4_5.txt` |
| Counts 80, 288, 1176, 2688, 4320 on Dic_5..Dic_9 | `enumerate_dicn_weight4.py --structured` | `expected_output_enumerate_structured_5_to_9.txt` |
| Prop. S1.4 (Dih(Z_4 x Z_4m), m=3..10, and the m=2 remark); Remark S1.2 (\|Aut\| of the Dic_n family for n=3..10 and of SD_{2^k} for k=4,5,6); Prop. S3.3 (MST on M_{2^n}, n=5..8) | `verify_supplement.py` | `expected_output_verify_supplement.txt` |
| Table S1 (normal sets; 19 groups) | `verify_normal_table.py` | `expected_output_verify_normal_table.txt` |
| Table tab:nn (sets, PST sets, non-normal, connected non-normal, undirected) and tab:ab (isomorphism classes, graphs that are not Cayley graphs of an abelian group) for orders <= 16 | `verify_nn_tables.py` | `expected_output_nn_order_*.txt` |
| No weight-4 solution on cyclic A (Remark S1.6(iii)), n = 13..34 | `search_cyclic_k4.py` | `expected_output_search_cyclic_k4.txt` |
| Remark S1.2: \|Aut\| of the Dic_n family (n = 3..12) and of SD_{2^k} (k = 4..7, including 2^38 for SD_128), computed with nauty | `check_aut_nauty.py` | `expected_output_check_aut_nauty.txt` |
| Lemma S3.2 = lem:Cn (spectrum and U(pi/4) of the oriented set C_n on M_{2^n}, n = 5..8) | `check_cn_lemma.py` | `expected_output_check_cn_lemma.txt` |
| Remark on Z_n, n = 6, 10, 14: among all mixed sets with zeta > 0 only the perfect matching has PST (23, 431, 7463 sets) | `check_zn_mixed.py` | `expected_output_check_zn_mixed.txt` |

`verify_families.py` and `verify_supplement.py` return exit status 0 iff every check passes;
`verify_normal_table.py` returns 0 iff all 19 rows agree with Table S1.
`run_all.sh` regenerates the short logs; the long ones are listed at the end.

## Arithmetic

* `K^2 = k^2 I` is tested exactly in Z[i][G] (integer arithmetic).
* Spectra, `U(pi/2)`, `U(pi/4)` and the PST tests in the table scripts are floating point (numpy/scipy,
  tolerance 1e-7 / 1e-8). PST is decided at the times pi*l/(2*gamma), l = 1,2,3, where gamma is the gcd of the
  differences of eigenvalues from zeta (the integrality and time results of the paper).
* Isomorphism tests (isomorphism classes; "not a Cayley graph of any abelian group") are exact searches against
  all mixed Cayley graphs of all abelian groups of the same order.

## What this archive does NOT contain or reproduce

* The "target" columns of Table tab:nn (order 2/4, non-central, other coset) and the per-group \|Aut\| column of
  tab:ab are not reproduced here (only the totals/distribution printed by `verify_nn_tables.py --abelian`).
* The claim "no solution on D_n for n <= 64" (only n <= 34 here), the exhaustive check of Theorem gd:k2 over the 162
  triples (A,y,c) and the programs of Appendix comp.
* The unrestricted enumeration on Dic_6 (all 2^22 subsets): `enumerate_dicn_weight4.py --unrestricted 6` is available
  but did not finish within ten minutes in the environment above, so no log is included; 288 for Dic_6 rests on the
  structured enumeration.
* The structured enumeration filters candidates with the Fourier conditions of the paper, hence is not independent of
  Theorem gd:fourier(b); the unrestricted runs (n <= 5) are.

## Usage

    bash run_all.sh                                  # short logs (about 3 minutes)
    python3 verify_nn_tables.py S3 D4 Q8 D5 D7       # tab:nn rows for the small groups
    python3 verify_nn_tables.py D6 Dic3 --abelian    # order 12, including abelian-Cayley status
    python3 verify_nn_tables.py D8 Q16 SD16 M16 Z4sdZ4 Z2xD4 Z2xQ8 Z4oD4 Z4Z2sdZ2 --abelian   # order 16 (about 15 minutes)
