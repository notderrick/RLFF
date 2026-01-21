# 🖥️ RLFF Retro Terminal Dashboard

## Overview

A fully-functional, retro-styled web interface for monitoring and controlling your RLFF training pipeline. Inspired by 1980s mainframe terminals, cyberpunk aesthetics, and classic CRT monitors.

**Live at**: http://localhost:5001

---

## 🎨 Visual Design

### CRT Monitor Simulation
- **Scanline Overlay**: Authentic horizontal scanning lines across the entire interface
- **Phosphor Glow**: Green text with subtle glow/bloom effect
- **Screen Curvature**: Radial gradient overlay simulating curved glass
- **Glitch Effects**: Animated RGB split on header text
- **Blinking Cursor**: Classic terminal cursor in output window

### Color Palette
```
Primary (Green):  #33ff33 - Main UI elements
Dim Green:        #1a8c1a - Secondary text
Background:       #0a0a0a - Deep black
Amber (Warning):  #ffb000 - Attention states
Red (Error):      #ff3333 - Error states
Blue (Info):      #3399ff - Information
```

### Typography
- **Primary Font**: Share Tech Mono (clean monospace)
- **Display Font**: VT323 (authentic terminal feel)
- **Letter Spacing**: Wide for readability
- **Text Shadow**: Phosphor glow on important elements

---

## 🎮 Features Breakdown

### 1. Control Panel (Top Left)

**Training Controls**:
- ▶ **RUN TESTS**: Execute environment test suite
- 🧠 **START SFT**: Begin supervised fine-tuning (quick mode: 100 examples, 1 epoch)
- ⚡ **START GRPO**: Start RL training (quick mode: 10 episodes)
- 🏆 **RUN TOURNAMENT**: Simulate leagues (quick mode: 10 leagues)
- ■ **STOP**: Terminate current training process

**Progress Tracking**:
- Animated progress bar with phosphor glow
- Current step / total steps counter
- Elapsed time (HH:MM:SS format)
- Progress percentage display

### 2. System Metrics (Top Right)

**Checkpoint Status**:
- SFT checkpoint availability (✓/✗)
- GRPO checkpoint availability (✓/✗)

**Player Pool**:
- Cache status indicator
- Shows if player data is pre-generated

**Performance Metrics**:
- Average reward (last 10 values)
- Model confidence (last 10 values)
- Updates in real-time during training

### 3. Terminal Output (Center, Full Width)

**Real-time Log Streaming**:
- Auto-scrolling terminal window
- Color-coded messages:
  - **Green**: Success messages (✓)
  - **Red**: Errors (✗)
  - **Amber**: Warnings (⚠)
- Timestamp for each log entry
- Blinking cursor at bottom
- Clear button to reset logs

### 4. Experiments Browser (Bottom Left)

**Checkpoint Explorer**:
- Lists all saved model checkpoints
- Shows mode (SFT/GRPO/Tournament)
- File size in MB
- Last modified timestamp
- Hover effects for interactivity

### 5. Player Statistics (Bottom Right)

**Position Breakdown**:
- Count of players per position
- Average VOR (Value Over Replacement)
- Average projected points
- Top player for each position

---

## 🚀 Usage Guide

### Starting the Dashboard

**Method 1: Quick Start Script**
```bash
./start_dashboard.sh
```

**Method 2: Makefile**
```bash
make dashboard
```

**Method 3: Manual**
```bash
source venv/bin/activate
python webapp/app.py
```

Then open **http://localhost:5001** in your browser.

### Running Training

1. **Click a training button** (e.g., "RUN TESTS")
2. **Watch the terminal** for real-time output
3. **Monitor progress bar** for completion status
4. **View metrics** updating in real-time

### Stopping Training

1. **Click STOP button** (becomes enabled when training is running)
2. **Process terminates** gracefully
3. **Status returns to IDLE**

---

## 📊 API Documentation

### Status Endpoint
```http
GET /api/status
```

**Response**:
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
  "player_pool_cached": true,
  "metrics": {
    "reward": [0.5, 0.6, 0.7],
    "confidence": [0.8, 0.82, 0.85]
  }
}
```

### Start Training
```http
POST /api/start_training
Content-Type: application/json

{
  "mode": "sft"
}
```

**Modes**: `test`, `sft`, `grpo`, `tournament`

### Logs Endpoint
```http
GET /api/logs
```

**Response**:
```json
{
  "logs": [
    {"time": "14:32:01", "message": "Starting SFT..."},
    {"time": "14:32:05", "message": "✓ Model loaded"}
  ]
}
```

---

## 🎯 Technical Details

### Frontend Stack
- **HTML5**: Semantic markup
- **CSS3**: Advanced animations, grid layout
- **Vanilla JavaScript**: No framework dependencies
- **Google Fonts**: VT323, Share Tech Mono

### Backend Stack
- **Flask 3.0**: Web framework
- **Flask-CORS**: Cross-origin support
- **Python 3.9+**: Server runtime
- **Threading**: Background job execution

### Real-time Updates
- **Polling**: Status updates every 1 second
- **Log streaming**: Updates every 2 seconds
- **No WebSockets**: Simpler deployment

### Browser Compatibility
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

## 🛠️ Customization

### Change Color Scheme to Amber

Edit `webapp/static/css/retro.css`:

```css
:root {
    --crt-green: #ffb000;  /* Change to amber */
    --crt-green-dim: #8c6400;
}
```

### Adjust Scanline Intensity

```css
.scanlines {
    background: linear-gradient(
        to bottom,
        transparent 50%,
        rgba(0, 0, 0, 0.5) 51%  /* Increase for darker scanlines */
    );
}
```

### Modify Refresh Rates

Edit `webapp/static/js/dashboard.js`:

```javascript
setInterval(updateStatus, 500);   // Faster updates (0.5s)
setInterval(updateLogs, 1000);    // Faster logs (1s)
```

---

## 🎭 Design Inspiration

### Visual References
- **Fallout 4**: Pip-Boy interface design
- **Alien (1979)**: Mother computer terminal
- **WarGames (1983)**: WOPR terminal
- **Blade Runner**: Voight-Kampff machine
- **The Matrix**: Falling code aesthetic

### CSS Techniques Used
- `linear-gradient` for scanlines
- `radial-gradient` for screen curvature
- `text-shadow` for phosphor glow
- `clip-path` for glitch effects
- `@keyframes` for animations
- `::-webkit-scrollbar` for custom scrollbars

---

## 📸 Screenshot Guide

### Header Area
- Large glitched title "RLFF // TRAINING TERMINAL"
- System time (updates every second)
- Status indicator (IDLE/RUNNING/COMPLETED/ERROR)

### Control Panel
- 5 colorful training buttons
- Progress bar with animated fill
- Step counter and elapsed time

### Terminal Window
- Black background with green text
- Scrolling log output
- Blinking cursor at bottom
- Clear button in header

### Lower Panels
- Left: Experiment list with file details
- Right: Player stats table with VOR data

---

## 🚧 Future Enhancements

### Planned Features
- [ ] Real-time training charts (Plotly.js integration)
- [ ] Win rate progression graph
- [ ] Draft heatmap visualization
- [ ] Model comparison tool
- [ ] Export training logs to file
- [ ] Dark/Light theme toggle
- [ ] Sound effects (optional retro bleeps)
- [ ] Full-screen mode toggle
- [ ] Keyboard shortcuts

### Advanced Ideas
- [ ] WebSocket support for lower latency
- [ ] Multi-user support (team training)
- [ ] Training queue system
- [ ] Email/Slack notifications on completion
- [ ] Integration with MLflow/WandB
- [ ] Mobile-responsive version
- [ ] PWA (Progressive Web App) support

---

## 🐛 Troubleshooting

### Dashboard Won't Start

**Issue**: Port 5000 already in use

**Solution**:
```bash
# Find and kill the process
lsof -i :5000
kill $(lsof -t -i :5000)

# Or change port in webapp/app.py:
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Training Button Not Working

**Issue**: Python not in PATH or venv not activated

**Solution**:
```bash
# Check venv activation
which python  # Should show /path/to/RLFF/venv/bin/python

# Manually activate
source venv/bin/activate
```

### Logs Not Updating

**Issue**: JavaScript errors in console

**Solution**:
- Open browser DevTools (F12)
- Check Console tab for errors
- Verify API endpoints are responding:
  - http://localhost:5001/api/status
  - http://localhost:5001/api/logs

### Blank Screen

**Issue**: CSS/JS files not loading

**Solution**:
- Check Flask static file serving
- Clear browser cache (Ctrl+Shift+R)
- Verify file paths in HTML

---

## 📝 Credits

**Design**: Inspired by retro computing, cyberpunk, and 1980s terminals

**Fonts**:
- VT323 by Peter Hull (Google Fonts)
- Share Tech Mono (Google Fonts)

**Frameworks**:
- Flask (BSD License)
- Vanilla JavaScript

**Built with**: ❤️, nostalgia, and way too much coffee

---

**Repository**: https://github.com/notderrick/RLFF

**Live Dashboard**: http://localhost:5001 (when running)

**Questions?**: Open an issue on GitHub
