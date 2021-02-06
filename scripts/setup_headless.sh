#!/bin/bash

# start nvidia-xconfig (might have to run this twice)
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

# start X on DISPLAY 0
sudo python ~/alfred/scripts/startx.py $1 & # if this throws errors e.g "(EE) Server terminated with error (1)" or "(EE) already running ..." try a display > 0

# source env
source ~/alfred_env/bin/activate
export DISPLAY=:$1

# check THOR
#cd $ALFRED_ROOT
#python scripts/check_thor.py
