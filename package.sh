#!/bin/bash

rm -f mannequin.tgz
rm -rf mannequin/
mkdir -p submission/
rsync * -r --exclude='submission' submission/

( cd submission
    rm -f run.sh
    rm -f package.sh
    rm -rf dist/
    rm -rf .git/
)

mv submission mannequin
tar -cvzf mannequin.tgz mannequin/*
rm -rf mannequin
