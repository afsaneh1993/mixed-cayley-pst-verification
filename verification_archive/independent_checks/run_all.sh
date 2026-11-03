#!/bin/bash
# Usage: bash run_all.sh            (all steps; about 25 minutes on one core)
#        bash run_all.sh STEP ...   (only the named steps; each writes logs/STEP.log)
cd "$(dirname "$0")"; mkdir -p logs
ALL="dn k4 block families_extra k2_float k2_exact dic_float dic_full_exact dic_pruned_exact search_exact search_ctrl_Z4xZ8 search_ctrl_Z4xZ12 cor_dicn tabnn_float_small tabnn_float_16a tabnn_float_16b tabnn_exact_small tabnn_exact_16a tabnn_exact_16b ab12_float ab_exact_12 ab_exact_16 znmixed ex32 open1 noncyclic_k4_Z6xZ2_Z4xZ2_Z8 noncyclic_k4_validate_prune noncyclic_k4_Z4xZ4 noncyclic_k4_Z8xZ2 noncyclic_k4_Z4xZ2xZ2 supplement_k4fam_m2_10 regular_abelian_8_12 regular_abelian_16"
for s in ${@:-$ALL}; do
  case $s in
    dn) python3 t1_dn.py ;; k4) python3 t2_k4.py ;; block) python3 t4_block.py ;; families_extra) python3 t9_families_extra.py ;;
    k2_float) python3 t3_k2.py ;; k2_exact) python3 t3_k2_exact.py ;;
    dic_float) python3 wenum.py 3 4 5 6 ;; dic_full_exact) python3 wenum_exact.py full 3 4 5 6 ;; dic_pruned_exact) python3 wenum_exact.py pruned 5 6 7 ;;
    search_exact) python3 search_exact.py cyc 13 34 ;; search_ctrl_Z4xZ8) python3 search_exact.py ab 4 8 ;; search_ctrl_Z4xZ12) python3 search_exact.py ab 4 12 ;;
    cor_dicn) python3 exact_cor_dicn.py ;;
    tabnn_float_small) python3 t5_tabnn.py S3 D4 Q8 D5 D6 Dic3 D7 ;; tabnn_float_16a) python3 t5_tabnn.py D8 Q16 SD16 M16 ;; tabnn_float_16b) python3 t5_tabnn.py Z4sdZ4 Z2xD4 Z2xQ8 Z4oD4 Z4Z2sdZ2 ;;
    tabnn_exact_small) python3 t5_exact.py S3 D4 Q8 D5 D6 Dic3 D7 ;; tabnn_exact_16a) python3 t5_exact.py D8 Q16 SD16 M16 ;; tabnn_exact_16b) python3 t5_exact.py Z4sdZ4 Z2xD4 Z2xQ8 Z4oD4 Z4Z2sdZ2 ;;
    ab12_float) python3 t7.py ;; ab_exact_12) python3 t7_ab16_exact.py 12 D6 Dic3 ;; ab_exact_16) python3 t7_ab16_exact.py 16 D8 Q16 SD16 M16 Z4sdZ4 Z2xD4 Z2xQ8 Z4oD4 Z4Z2sdZ2 ;;
    noncyclic_k4_Z6xZ2_Z4xZ2_Z8) python3 search_noncyclic_k4.py Z6xZ2 Z4xZ2 Z8 ;; noncyclic_k4_validate_prune) python3 validate_prune.py Z6xZ2 ;;
    noncyclic_k4_Z4xZ4) python3 search_noncyclic_k4.py Z4xZ4 ;; noncyclic_k4_Z8xZ2) python3 search_noncyclic_k4.py Z8xZ2 ;; noncyclic_k4_Z4xZ2xZ2) python3 search_noncyclic_k4.py Z4xZ2xZ2 ;;
    supplement_k4fam_m2_10) python3 t11_supplement_k4fam.py ;; regular_abelian_8_12) (python3 t12_regular_abelian.py 8 D4 Q8; python3 t12_regular_abelian.py 12 D6 Dic3) ;;
    regular_abelian_16) python3 t12_regular_abelian.py 16 D8 Q16 SD16 M16 Z4sdZ4 Z2xD4 Z2xQ8 Z4oD4 Z4Z2sdZ2 ;;
    znmixed) python3 t6.py ;; ex32) python3 t8.py ;; open1) python3 t10_open1.py D8 Q16 ;;
    *) echo "unknown step $s"; continue ;;
  esac > logs/$s.log 2>&1
  echo "done $s (exit $?)"
done
