#!/bin/bash



funs=(process_stvqa.py process_textvqa.py process_tdiuc.py process_gqa.py process_okvqa.py process_vqav2.py)
for f in ${funs[@]}; do
    logf=save/$$_prepare${i}_$(date +'%m-%d').log
    echo "running ${f}" > $logf
    nohup python prepare/$f >> $logf &
done
