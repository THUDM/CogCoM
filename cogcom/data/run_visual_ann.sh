
dino_ckpt=path/to/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
dino_cfg=path/to/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py

# i=0
basedir=save/steps_5shot_extract
files=(${basedir}/0.jsonl  ${basedir}/1.jsonl)
outdir=save/steps_5shot_visual
for fin in ${files[@]}; do
    export CUDA_VISIBLE_DEVICES=${i}
    logf=$$visual_${i}_$(date +'%m-%d').log
    echo "processing ${fin} on device ${i}" > $logf
    nohup python ann_visual.py \
        -c $dino_cfg \
        -p $dino_ckpt \
        -i $fin \
        -o $outdir >> $logf &
    let i++
done

