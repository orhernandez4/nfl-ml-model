#!/bin/bash
set -e
export PYTHONPATH=$(pwd)

# add some style
fancy_echo() {
    local input="$1"
    local flair="─ ⋆⋅☆⋅⋆ ─"
    local border_length=$(( ${#input} - ${#flair}))
    local bar=$(printf '─%.0s' $(seq 1 $border_length))
    echo "┌${bar}${flair}┐"
    echo " $input "
    echo "└${flair}${bar}┘"
}

# paths
SOURCE="src"
MODELING="$SOURCE/model"


# build data
echo -n "Build data? [y/N]: "
read answer
if [[ $answer =~ ^[Yy]$ ]]; then
    fancy_echo "Building data"
    uv run $SOURCE/data/build.py
fi


# train models
echo -n "Train models? [y/N]: "
read answer
if [[ $answer =~ ^[Yy]$ ]]; then
    fancy_echo "Training models"
    uv run $MODELING/train.py
fi


# generate predictions
echo -n "Generate predictions? [y/N]: "
read answer
if [[ $answer =~ ^[Yy]$ ]]; then
    fancy_echo "Generating predictions"
    uv run $SOURCE/data/predict/predict.py
fi
