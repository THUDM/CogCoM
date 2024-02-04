

i=0
in_dirs=(save/steps_5shot_visual save/steps_absurd_extract)
out_dirs=(save/steps_5shot_wds save/steps_absurd_wds)
names=(comnormal comabsurd)
for f in ${in_dirs[@]}; do
    logf=$$convert_${i}_$(date +'%m-%d').log
    nohup python convert_to_wds.py \
        --data_name ${names[i]} \
        --in_dir ${in_dirs[i]} \
        --out_dir ${out_dirs[i]} \
        >> $logf &
    let i++
done


