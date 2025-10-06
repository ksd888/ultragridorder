#!/bin/bash
# Backup critical files

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup state and logs
cp -r state/ "$BACKUP_DIR/"
cp -r logs/ "$BACKUP_DIR/"
cp grid_plan.csv "$BACKUP_DIR/"
cp config.yaml "$BACKUP_DIR/"

echo "✅ Backup created: $BACKUP_DIR"

# Keep only last 30 backups
ls -dt backups/*/ | tail -n +31 | xargs rm -rf
EOF

chmod +x backup.sh