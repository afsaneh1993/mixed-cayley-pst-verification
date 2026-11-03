#!/usr/bin/env bash
# Regenerates the short logs (about 3 minutes; Python 3.12, numpy, scipy).
set -e
python3 verify_families.py                                 > expected_output_verify_families.txt
python3 verify_supplement.py                               > expected_output_verify_supplement.txt
python3 enumerate_dicn_weight4.py --unrestricted 3 4 5     > expected_output_enumerate_unrestricted_3_4_5.txt
python3 enumerate_dicn_weight4.py --structured 5 6 7 8 9   > expected_output_enumerate_structured_5_to_9.txt
python3 search_cyclic_k4.py 34                             > expected_output_search_cyclic_k4.txt
python3 verify_normal_table.py                             > expected_output_verify_normal_table.txt
python3 check_aut_nauty.py                                   > expected_output_check_aut_nauty.txt   # needs pynauty
python3 check_zn_mixed.py                                   > expected_output_check_zn_mixed.txt
python3 check_cn_lemma.py                                    > expected_output_check_cn_lemma.txt
# Long runs (minutes):
#   python3 verify_nn_tables.py S3 D4 Q8 D5 D7                                  > expected_output_nn_order_le_10_and_14.txt
#   python3 verify_nn_tables.py D6 Dic3 --abelian                               > expected_output_nn_order_12.txt
#   python3 verify_nn_tables.py D8 Q16 SD16 M16 Z4sdZ4 Z2xD4 Z2xQ8 Z4oD4 Z4Z2sdZ2 --abelian > expected_output_nn_order_16.txt
