# 🖥️ RLFF Retro Terminal Dashboard

A CRT-inspired web interface for monitoring and controlling your RLFF training pipeline.

## Features

### 🎮 Control Panel
- One-click training execution (Test, SFT, GRPO, Tournament)
- Real-time progress tracking
- Stop/start controls
- Training time elapsed

### 📊 Live Metrics
- **System Status**: Current training state
- **Checkpoints**: SFT/GRPO model availability
- **Player Pool**: Cache status
- **Performance**: Average reward & confidence scores

### 📺 Terminal Output
- Real-time log streaming
- Color-coded messages (success/error/info)
- Auto-scrolling terminal
- CRT phosphor effects

### 🔬 Experiment Browser
- View all saved checkpoints
- File sizes and timestamps
- Quick access to models

### 📈 Player Statistics
- VOR breakdown by position
- Player pool composition
- Average projections

## Design Philosophy

**Retro Computing Aesthetic**:
- CRT monitor simulation with scanlines
- Phosphor green monochrome color scheme
- VT323 monospace font
- Terminal cursor animation
- Glitch effects on headers
- Screen curvature overlay

Inspired by:
- 1980s mainframe terminals
- Fallout Pip-Boy interfaces
- Cyberpunk hacker screens
- Classic BIOS interfaces

## Quick Start

```bash
# Start the dashboard
./start_dashboard.sh

# Or use Make
make dashboard

# Or manually
source venv/bin/activate
python webapp/app.py
```

Then open: **http://localhost:5001**

## API Endpoints

### `GET /api/status`
Returns current system status and metrics

```json
{
  "status": "running",
  "progress": 45,
  "current_step": 450,
  "total_steps": 1000,
  "elapsed_time": "00:15:32",
  "checkpoints": {
    "sft": true,
    "grpo": false
  },
  "metrics": {
    "reward": [0.5, 0.6, 0.7, ...],
    "confidence": [0.8, 0.82, 0.85, ...]
  }
}
```

### `GET /api/logs`
Returns recent training logs

```json
{
  "logs": [
    {"time": "14:32:01", "message": "Starting SFT..."},
    {"time": "14:32:05", "message": "✓ Model loaded"}
  ]
}
```

### `POST /api/start_training`
Start a training job

```json
{
  "mode": "sft"  // Options: test, sft, grpo, tournament
}
```

### `POST /api/stop_training`
Stop current training

### `GET /api/experiments`
List all experiment checkpoints

### `GET /api/player_stats`
Get player pool statistics

## Architecture

```
webapp/
├── app.py              # Flask server
├── templates/
│   └── dashboard.html  # Main UI
├── static/
│   ├── css/
│   │   └── retro.css   # CRT styling
│   └── js/
│       └── dashboard.js # Client logic
└── README.md
```

## Customization

### Colors
Edit `webapp/static/css/retro.css`:

```css
:root {
    --crt-green: #33ff33;      /* Primary color */
    --crt-amber: #ffb000;      /* Warning color */
    --crt-red: #ff3333;        /* Error color */
    --crt-blue: #3399ff;       /* Info color */
}
```

### Fonts
Change to amber monochrome:

```css
body {
    color: var(--crt-amber);
}

.btn {
    border-color: var(--crt-amber);
    color: var(--crt-amber);
}
```

### Refresh Rates
Edit `webapp/static/js/dashboard.js`:

```javascript
setInterval(updateStatus, 1000);  // Status: 1 second
setInterval(updateLogs, 2000);    // Logs: 2 seconds
```

## Browser Support

Tested on:
- ✓ Chrome 90+
- ✓ Firefox 88+
- ✓ Safari 14+
- ✓ Edge 90+

## Tips

1. **Full Screen**: Press F11 for immersive CRT experience
2. **Dark Room**: Best viewed in low light for authentic feel
3. **Monitor Tilt**: Slight upward angle enhances scanline effect
4. **Sound**: Play ambient synthwave for maximum vibes

## Troubleshooting

**Dashboard won't start**:
```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill existing process
kill $(lsof -t -i :5000)
```

**Flask not found**:
```bash
source venv/bin/activate
pip install flask flask-cors
```

**Can't access from network**:
The server binds to `0.0.0.0:5000` so it should be accessible from other devices on your network at `http://YOUR_IP:5000`

---

**Built with ❤️ and nostalgia for the golden age of computing**
