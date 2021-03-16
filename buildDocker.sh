#!/bin/bash
python3 ./scripts/buildModel.py
MODELSIZE=$(cat MAXMODELSIZE)
docker build --no-cache --build-arg buildTime_MODELSIZE=$MODELSIZE -t uvIndex:latest -f dockerfile