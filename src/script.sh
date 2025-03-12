for i in  ledgar scotus ecthr_a eurlex; do
  for j in  bert-base-uncased microsoft/deberta-base roberta-base nlpaueb/legal-bert-base-uncased ; do
    for k in test validation train ; do
      for l in common_disorder common_hair common_uncommon disorder_common old_young uncommon_common young_old hair_common; do
        python -u main.py lexglue/"$i"/"$j"/seed_1/ lex_glue data/"$l".csv all_occurrences replacement --"$k"=True;
      done;
    done;
  done;
done

for i in  ledgar scotus ecthr_a eurlex; do
  for j in  bert-base-uncased microsoft/deberta-base roberta-base nlpaueb/legal-bert-base-uncased ; do
    for k in test validation train; do
      for l in common_disorder common_hair common_uncommon disorder_common old_young uncommon_common young_old hair_common; do
        for m in african_american african_arab african_asian african_european american_african american_arab american_asian american_european arab_african arab_american arab_asian arab_european asian_african asian_american asian_arab asian_european european_african european_american european_arab european_asian majority_minority majority_mixed minority_majority minority_mixed mixed_majority mixed_minority; do
          python -u main.py /data/lexglue/logs/"$i"/"$j"/seed_1/ lex_glue data/body/"$l".csv --inter_dict_path=data/race/"$m".csv all_occurrences intersectional "$k";
        done;
      done;
    done;
  done;
done
