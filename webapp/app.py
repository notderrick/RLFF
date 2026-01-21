"""
RLFF Web Dashboard
Retro terminal-style interface for monitoring and controlling training
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import json
import os
import sys
from pathlib import Path
import subprocess
import threading
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
CORS(app)

# Global state
training_process = None
training_logs = []
max_log_lines = 1000

class TrainingMonitor:
    """Monitor training process and collect metrics"""

    def __init__(self):
        self.status = "idle"  # idle, running, completed, error
        self.current_step = 0
        self.total_steps = 0
        self.metrics = {
            'loss': [],
            'reward': [],
            'confidence': [],
            'win_rate': []
        }
        self.start_time = None
        self.end_time = None

    def start(self, mode, steps):
        self.status = "running"
        self.current_step = 0
        self.total_steps = steps
        self.start_time = datetime.now()
        self.end_time = None

    def update(self, step, metrics_dict):
        self.current_step = step
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def complete(self):
        self.status = "completed"
        self.end_time = datetime.now()

    def error(self):
        self.status = "error"
        self.end_time = datetime.now()

    def get_progress(self):
        if self.total_steps == 0:
            return 0
        return int((self.current_step / self.total_steps) * 100)

    def get_elapsed_time(self):
        if self.start_time is None:
            return "00:00:00"

        end = self.end_time or datetime.now()
        elapsed = end - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

monitor = TrainingMonitor()


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """Get current system status"""

    # Check if experiments exist
    experiments_dir = Path(__file__).parent.parent / "experiments"

    sft_exists = (experiments_dir / "sft" / "final").exists()
    grpo_exists = (experiments_dir / "grpo" / "checkpoint_final").exists()

    # Get player pool info
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    player_cache = list(data_dir.glob("players_*.json"))

    return jsonify({
        'status': monitor.status,
        'progress': monitor.get_progress(),
        'current_step': monitor.current_step,
        'total_steps': monitor.total_steps,
        'elapsed_time': monitor.get_elapsed_time(),
        'checkpoints': {
            'sft': sft_exists,
            'grpo': grpo_exists
        },
        'player_pool_cached': len(player_cache) > 0,
        'metrics': {
            key: values[-50:] if values else []  # Last 50 points
            for key, values in monitor.metrics.items()
        }
    })


@app.route('/api/logs')
def api_logs():
    """Get recent training logs"""
    return jsonify({
        'logs': training_logs[-100:]  # Last 100 lines
    })


@app.route('/api/experiments')
def api_experiments():
    """List all experiment checkpoints"""
    experiments_dir = Path(__file__).parent.parent / "experiments"

    experiments = []

    if experiments_dir.exists():
        for mode_dir in experiments_dir.iterdir():
            if mode_dir.is_dir():
                for checkpoint in mode_dir.iterdir():
                    if checkpoint.is_dir():
                        # Get checkpoint info
                        stat = checkpoint.stat()
                        experiments.append({
                            'mode': mode_dir.name,
                            'name': checkpoint.name,
                            'path': str(checkpoint),
                            'size_mb': sum(f.stat().st_size for f in checkpoint.rglob('*') if f.is_file()) / (1024 * 1024),
                            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        })

    return jsonify({'experiments': experiments})


@app.route('/api/start_training', methods=['POST'])
def api_start_training():
    """Start a training job"""
    global training_process, training_logs

    data = request.json
    mode = data.get('mode')  # 'test', 'sft', 'grpo', 'tournament'

    if training_process and training_process.poll() is None:
        return jsonify({'error': 'Training already running'}), 400

    # Prepare command
    project_root = Path(__file__).parent.parent
    venv_python = project_root / "venv" / "bin" / "python"

    commands = {
        'test': [str(venv_python), 'test_env.py'],
        'sft': [str(venv_python), 'train_sft.py', '--num-examples', '100', '--epochs', '1'],  # Quick test
        'grpo': [str(venv_python), 'train_grpo.py', '--episodes', '10'],  # Quick test
        'tournament': [str(venv_python), 'tournament.py', '--leagues', '10']  # Quick test
    }

    if mode not in commands:
        return jsonify({'error': f'Invalid mode: {mode}'}), 400

    # Clear logs
    training_logs = []

    # Start training
    monitor.start(mode, 100)  # Placeholder steps

    def run_training():
        global training_process
        try:
            training_process = subprocess.Popen(
                commands[mode],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output
            for line in training_process.stdout:
                line = line.strip()
                if line:
                    training_logs.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'message': line
                    })

                    # Keep only last N lines
                    if len(training_logs) > max_log_lines:
                        training_logs.pop(0)

            training_process.wait()

            if training_process.returncode == 0:
                monitor.complete()
                training_logs.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'message': f'✓ {mode.upper()} completed successfully'
                })
            else:
                monitor.error()
                training_logs.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'message': f'✗ {mode.upper()} failed with code {training_process.returncode}'
                })

        except Exception as e:
            monitor.error()
            training_logs.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'message': f'✗ Error: {str(e)}'
            })

    # Run in background thread
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'mode': mode})


@app.route('/api/stop_training', methods=['POST'])
def api_stop_training():
    """Stop current training"""
    global training_process

    if training_process and training_process.poll() is None:
        training_process.terminate()
        monitor.status = "idle"
        training_logs.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'message': '⚠ Training stopped by user'
        })
        return jsonify({'success': True})

    return jsonify({'error': 'No training running'}), 400


@app.route('/api/player_stats')
def api_player_stats():
    """Get player pool statistics"""
    from src.data import PlayerLoader

    try:
        loader = PlayerLoader()
        players = loader.load_players(num_players=300, use_cache=True)

        # Calculate stats by position
        stats = {}
        for player in players:
            pos = player.position.value
            if pos not in stats:
                stats[pos] = {
                    'count': 0,
                    'avg_vor': 0,
                    'avg_proj': 0,
                    'top_player': None
                }

            stats[pos]['count'] += 1
            stats[pos]['avg_vor'] += player.vor
            stats[pos]['avg_proj'] += player.projected_points

            if stats[pos]['top_player'] is None or player.vor > stats[pos]['top_player']['vor']:
                stats[pos]['top_player'] = {
                    'name': player.name,
                    'vor': player.vor,
                    'proj': player.projected_points
                }

        # Average
        for pos in stats:
            count = stats[pos]['count']
            stats[pos]['avg_vor'] = round(stats[pos]['avg_vor'] / count, 1)
            stats[pos]['avg_proj'] = round(stats[pos]['avg_proj'] / count, 1)

        return jsonify({'stats': stats})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏈 RLFF Dashboard Starting...")
    print("="*60)
    print("\n🌐 Open in browser: http://localhost:5000")
    print("📊 Retro terminal interface loaded")
    print("\nPress Ctrl+C to stop\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
