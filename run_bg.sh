#!/bin/bash
cd /home/openbot/workspace/projects/factor-pipeline
python3 run_alpha191_batch.py > /tmp/alpha191_run.log 2>&1
echo "DONE at $(date)" >> /tmp/alpha191_run.log
