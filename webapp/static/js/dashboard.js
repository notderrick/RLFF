// RLFF Dashboard JavaScript

// Update system time
function updateTime() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
    document.getElementById('system-time').textContent = timeStr;
}

// Update status
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        // Update status indicator
        const statusEl = document.getElementById('system-status');
        statusEl.textContent = data.status.toUpperCase();
        statusEl.className = `status-${data.status}`;

        // Update progress
        document.getElementById('progress-text').textContent =
            data.status === 'running' ? 'TRAINING IN PROGRESS' :
            data.status === 'completed' ? 'COMPLETED' :
            data.status === 'error' ? 'ERROR' : 'READY';

        document.getElementById('progress-percent').textContent = `${data.progress}%`;
        document.getElementById('progress-fill').style.width = `${data.progress}%`;
        document.getElementById('step-info').textContent = `${data.current_step} / ${data.total_steps}`;
        document.getElementById('time-info').textContent = data.elapsed_time;

        // Update checkpoints
        document.getElementById('sft-checkpoint').textContent =
            `SFT: ${data.checkpoints.sft ? '✓' : '✗'}`;
        document.getElementById('grpo-checkpoint').textContent =
            `GRPO: ${data.checkpoints.grpo ? '✓' : '✗'}`;

        // Update player pool status
        document.getElementById('player-pool-status').textContent =
            data.player_pool_cached ? '✓ CACHED' : '✗ NOT CACHED';

        // Update metrics
        if (data.metrics.reward.length > 0) {
            const avgReward = data.metrics.reward.slice(-10).reduce((a, b) => a + b, 0) /
                             Math.min(10, data.metrics.reward.length);
            document.getElementById('avg-reward').textContent = avgReward.toFixed(2);
        }

        if (data.metrics.confidence.length > 0) {
            const avgConf = data.metrics.confidence.slice(-10).reduce((a, b) => a + b, 0) /
                           Math.min(10, data.metrics.confidence.length);
            document.getElementById('confidence').textContent = avgConf.toFixed(2);
        }

        // Enable/disable stop button
        const stopBtn = document.getElementById('stop-btn');
        stopBtn.disabled = data.status !== 'running';

    } catch (error) {
        console.error('Error updating status:', error);
    }
}

// Update logs
async function updateLogs() {
    try {
        const response = await fetch('/api/logs');
        const data = await response.json();

        const terminal = document.getElementById('terminal');

        // Clear old logs but keep welcome message
        const welcomeMsg = terminal.querySelector('.welcome');
        terminal.innerHTML = '';
        if (welcomeMsg) {
            terminal.appendChild(welcomeMsg);
        }

        // Add new logs
        data.logs.forEach(log => {
            const line = document.createElement('div');
            line.className = 'terminal-line';

            // Colorize based on content
            if (log.message.includes('✓') || log.message.toLowerCase().includes('success')) {
                line.classList.add('success');
            } else if (log.message.includes('✗') || log.message.toLowerCase().includes('error')) {
                line.classList.add('error');
            }

            line.textContent = `[${log.time}] ${log.message}`;
            terminal.appendChild(line);
        });

        // Add cursor
        const cursor = document.createElement('div');
        cursor.className = 'terminal-cursor';
        cursor.textContent = '█';
        terminal.appendChild(cursor);

        // Scroll to bottom
        terminal.scrollTop = terminal.scrollHeight;

    } catch (error) {
        console.error('Error updating logs:', error);
    }
}

// Start training
async function startTraining(mode) {
    try {
        const response = await fetch('/api/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ mode })
        });

        const data = await response.json();

        if (data.error) {
            addLog(`ERROR: ${data.error}`, 'error');
        } else {
            addLog(`Starting ${mode.toUpperCase()} training...`, 'success');
        }

    } catch (error) {
        console.error('Error starting training:', error);
        addLog(`ERROR: Failed to start training`, 'error');
    }
}

// Stop training
async function stopTraining() {
    try {
        const response = await fetch('/api/stop_training', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            addLog(`ERROR: ${data.error}`, 'error');
        } else {
            addLog('Training stopped', 'success');
        }

    } catch (error) {
        console.error('Error stopping training:', error);
    }
}

// Clear logs
function clearLogs() {
    const terminal = document.getElementById('terminal');
    terminal.innerHTML = `
        <div class="terminal-line welcome">> RLFF TRAINING SYSTEM v1.0</div>
        <div class="terminal-line">> LOGS CLEARED</div>
        <div class="terminal-cursor">█</div>
    `;
}

// Add log manually
function addLog(message, type = '') {
    const terminal = document.getElementById('terminal');
    const cursor = terminal.querySelector('.terminal-cursor');

    const line = document.createElement('div');
    line.className = `terminal-line ${type}`;
    line.textContent = `[${new Date().toLocaleTimeString('en-US', { hour12: false })}] ${message}`;

    terminal.insertBefore(line, cursor);
    terminal.scrollTop = terminal.scrollHeight;
}

// Load experiments
async function loadExperiments() {
    try {
        const response = await fetch('/api/experiments');
        const data = await response.json();

        const listEl = document.getElementById('experiments-list');

        if (data.experiments.length === 0) {
            listEl.innerHTML = '<div class="loading">NO EXPERIMENTS FOUND</div>';
            return;
        }

        listEl.innerHTML = '';
        data.experiments.forEach(exp => {
            const item = document.createElement('div');
            item.className = 'experiment-item';
            item.innerHTML = `
                <div class="experiment-mode">${exp.mode.toUpperCase()} / ${exp.name}</div>
                <div class="experiment-info">
                    ${exp.size_mb.toFixed(1)} MB | ${exp.modified}
                </div>
            `;
            listEl.appendChild(item);
        });

    } catch (error) {
        console.error('Error loading experiments:', error);
    }
}

// Load player stats
async function loadPlayerStats() {
    try {
        const response = await fetch('/api/player_stats');
        const data = await response.json();

        const statsEl = document.getElementById('player-stats');

        if (data.error) {
            statsEl.innerHTML = `<div class="loading">ERROR: ${data.error}</div>`;
            return;
        }

        statsEl.innerHTML = '';

        Object.entries(data.stats).forEach(([pos, stats]) => {
            const row = document.createElement('div');
            row.className = 'stat-row';
            row.innerHTML = `
                <span class="stat-label">${pos}</span>
                <span class="stat-value">${stats.count} players | VOR: ${stats.avg_vor}</span>
            `;
            statsEl.appendChild(row);
        });

    } catch (error) {
        console.error('Error loading player stats:', error);
        const statsEl = document.getElementById('player-stats');
        statsEl.innerHTML = '<div class="loading">ERROR LOADING STATS</div>';
    }
}

// Initialize
function init() {
    updateTime();
    setInterval(updateTime, 1000);

    updateStatus();
    setInterval(updateStatus, 1000);

    updateLogs();
    setInterval(updateLogs, 2000);

    loadExperiments();
    loadPlayerStats();

    console.log('🏈 RLFF Dashboard initialized');
}

// Run on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
