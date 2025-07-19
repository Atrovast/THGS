###### Description: Run the pipeline for our work.
###### Usage: bash run.sh config_file [optinal: specific scenes to process]
# python launcher.py -f sp_partition.py -cf configs/scannet.yml
# python launcher.py -f graph_weight.py -cf configs/scannet.yml
# python launcher.py -f sp_partition.py -cf configs/scannet.yml -k
# python launcher.py -f merge_proj.py -cf configs/scannet.yml
# if no scenes are provided, all scenes in the dataset will be processed, use no -sc flag
config_file=$1
scenes=${@:2}
echo "Running pipeline for" $config_file
if [ -z "$scenes" ]; then
    python scripts/launcher.py -f sp_partition.py -cf $config_file
    python scripts/launcher.py -f graph_weight.py -cf $config_file
    python scripts/launcher.py -f sp_partition.py -cf $config_file -k
    python scripts/launcher.py -f merge_proj.py -cf $config_file
else
    python scripts/launcher.py -f sp_partition.py -cf $config_file -sc $scenes
    python scripts/launcher.py -f graph_weight.py -cf $config_file -sc $scenes
    python scripts/launcher.py -f sp_partition.py -cf $config_file -sc $scenes -k
    python scripts/launcher.py -f merge_proj.py -cf $config_file -sc $scenes
fi