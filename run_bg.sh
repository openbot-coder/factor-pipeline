#!/bin/bash
cd "$(dirname "$0")"
mkdir -p logs

# Create factors list file for batch processing
cat > /tmp/alpha191_factors.txt <<'EOF'
$close
Mean($close, 5)
Mean($close, 20)
Std($close, 20)
Delta($close, 1)
Rank($close)
Log($volume)
Corr($close, $volume, 20)
EOF

fp factor batch /tmp/alpha191_factors.txt --output results/alpha191 > logs/alpha191_run.log 2>&1
echo "DONE at $(date)" >> logs/alpha191_run.log
