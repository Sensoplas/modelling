python ./scripts/buildModel.py
$MODELSIZE = Get-Content -Path ./MAXMODELSIZE
docker build --no-cache --build-arg buildTime_MODELSIZE=$MODELSIZE -t gcr.io/sensoplas/uvindex:latest .