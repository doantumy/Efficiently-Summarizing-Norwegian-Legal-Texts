#!/usr/bin/env zsh

# conda activate legsum
# IDUN
source env/bin/activate

echo "SHELL TRAINER LAYER START | Input: $1, Output: $2, Model: $3"
mkdir "$2"

python models/train_summarizor.py -i "$1" -o "$2" -m "$3" -w "$4" -e "$5" -r "$6" -g "$7" -wm "$8" -lr "$9" -bs "$10" -st "$11" -ls "$12" -ml "$13" -mnl "$14" -mnt "$15" -tem "$16" -tk "$17" -tp "$18" -nrng "$19" -nb "$20"

echo "SHELL TRAINER LAYER END  | Input: $1, Output: $2, Model: $3"