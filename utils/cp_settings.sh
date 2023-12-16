#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p ~/Documents/AirSim/
cp $SCRIPT_DIR/../settings/settings.json ~/Documents/AirSim/settings.json