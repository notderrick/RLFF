#!/bin/bash
# RLFF Dashboard Launcher

echo "🏈 RLFF Training Dashboard"
echo "=========================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install Flask if needed
echo "📦 Checking dependencies..."
pip install flask flask-cors -q

echo ""
echo "✓ Starting dashboard..."
echo ""

# Run the app
python webapp/app.py
