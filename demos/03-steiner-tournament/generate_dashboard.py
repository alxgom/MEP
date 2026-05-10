"""
Dashboard Generator
===================
Reads heavy_results.json and produces a standalone dashboard.html.
"""

import json
import os

def generate_dashboard(results_file="heavy_results.json", output_file="dashboard.html"):
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Run run_heavy_benchmark.py first.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Steiner Heavy Benchmark Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .canvas-container { position: relative; width: 100%; height: 500px; background: #1a1a2e; border-radius: 8px; overflow: hidden; }
        canvas { display: block; width: 100%; height: 100%; }
        .card { background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); padding: 20px; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen p-8">
    <div class="max-w-7xl mx-auto">
        <header class="mb-8 flex justify-between items-center">
            <div>
                <h1 class="text-3xl font-bold text-gray-800">Steiner Heavy Benchmark Dashboard</h1>
                <p class="text-gray-600">Explore results for N=70 Terminals across 20 Seeds</p>
            </div>
            <div class="flex gap-4">
                <select id="caseSelect" class="p-2 border rounded shadow-sm bg-white"></select>
                <select id="algoSelect" class="p-2 border rounded shadow-sm bg-white"></select>
            </div>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            <!-- Stats Card -->
            <div class="card col-span-1">
                <h2 class="text-xl font-semibold mb-4 border-b pb-2">Metrics</h2>
                <div id="metrics" class="space-y-4">
                    <div class="flex justify-between"><span>Length:</span> <b id="statLength">-</b></div>
                    <div class="flex justify-between"><span>Gap %:</span> <b id="statGap">-</b></div>
                    <div class="flex justify-between"><span>Time:</span> <b id="statTime">-</b></div>
                    <div class="flex justify-between"><span>Steiner Points:</span> <b id="statSteiner">-</b></div>
                    <div class="flex justify-between"><span>Max 120° Dev:</span> <b id="statDev">-</b></div>
                </div>
            </div>

            <!-- Visualization Card -->
            <div class="card lg:col-span-2">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-semibold">Topology Visualizer</h2>
                    <span class="text-sm text-gray-500">Terminals (Blue), Steiner (Orange)</span>
                </div>
                <div class="canvas-container">
                    <canvas id="steinerCanvas"></canvas>
                </div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
             <div class="card">
                <h2 class="text-xl font-semibold mb-4">Optimality Gap per Algorithm</h2>
                <canvas id="gapChart"></canvas>
            </div>
            <div class="card">
                <h2 class="text-xl font-semibold mb-4">Execution Time (log scale)</h2>
                <canvas id="timeChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const rawData = DATA_PLACEHOLDER;
        
        const caseSelect = document.getElementById('caseSelect');
        const algoSelect = document.getElementById('algoSelect');
        const canvas = document.getElementById('steinerCanvas');
        const ctx = canvas.getContext('2d');

        // Extract unique cases and algos
        const cases = [...new Set(rawData.map(r => r.case))].sort((a,b) => parseInt(a.split('_')[1]) - parseInt(b.split('_')[1]));
        const algos = [...new Set(rawData.map(r => r.algorithm))];

        cases.forEach(c => { const opt = new Option(c, c); caseSelect.add(opt); });
        algos.forEach(a => { const opt = new Option(a, a); algoSelect.add(opt); });

        function updateUI() {
            const selectedCase = caseSelect.value;
            const selectedAlgo = algoSelect.value;
            const result = rawData.find(r => r.case === selectedCase && r.algorithm === selectedAlgo);

            if (result) {
                document.getElementById('statLength').textContent = result.length.toFixed(4);
                document.getElementById('statGap').textContent = result.gap_pct.toFixed(2) + '%';
                document.getElementById('statTime').textContent = result.time.toFixed(4) + 's';
                document.getElementById('statSteiner').textContent = result.steiner_count;
                document.getElementById('statDev').textContent = result.max_120_dev.toExponential(2) + '°';
                drawTree(result);
            }
        }

        function drawTree(data) {
            const points = data.points;
            const edges = data.edges;
            const n_terminals = points.length - data.steiner_count;

            // Set internal resolution
            canvas.width = canvas.offsetWidth * window.devicePixelRatio;
            canvas.height = canvas.offsetHeight * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            const w = canvas.offsetWidth;
            const h = canvas.offsetHeight;
            const margin = 40;

            // Normalize points
            const xs = points.map(p => p[0]);
            const ys = points.map(p => p[1]);
            const minX = Math.min(...xs), maxX = Math.max(...xs);
            const minY = Math.min(...ys), maxY = Math.max(...ys);
            
            const scale = Math.min((w - 2 * margin) / (maxX - minX || 1), (h - 2 * margin) / (maxY - minY || 1));
            const offsetX = (w - (maxX - minX) * scale) / 2 - minX * scale;
            const offsetY = (h - (maxY - minY) * scale) / 2 - minY * scale;

            const mapX = (x) => x * scale + offsetX;
            const mapY = (y) => h - (y * scale + offsetY); // Flip Y for screen coords

            ctx.clearRect(0, 0, w, h);

            // Draw Edges
            ctx.strokeStyle = '#4ade80'; // Green
            ctx.lineWidth = 2;
            edges.forEach(([u, v]) => {
                ctx.beginPath();
                ctx.moveTo(mapX(points[u][0]), mapY(points[u][1]));
                ctx.lineTo(mapX(points[v][0]), mapY(points[v][1]));
                ctx.stroke();
            });

            // Draw Points
            points.forEach((p, i) => {
                const isSteiner = i >= n_terminals;
                ctx.fillStyle = isSteiner ? '#fb923c' : '#3b82f6';
                ctx.beginPath();
                ctx.arc(mapX(p[0]), mapY(p[1]), isSteiner ? 4 : 5, 0, Math.PI * 2);
                ctx.fill();
                if (!isSteiner) {
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            });
        }

        function initCharts() {
            // Summary Data
            const summary = algos.map(a => {
                const filtered = rawData.filter(r => r.algorithm === a);
                return {
                    name: a,
                    avgGap: filtered.reduce((acc, r) => acc + r.gap_pct, 0) / filtered.length,
                    avgTime: filtered.reduce((acc, r) => acc + r.time, 0) / filtered.length
                };
            }).sort((a,b) => a.avgGap - b.avgGap);

            new Chart(document.getElementById('gapChart'), {
                type: 'bar',
                data: {
                    labels: summary.map(s => s.name),
                    datasets: [{
                        label: 'Average Optimality Gap %',
                        data: summary.map(s => s.avgGap),
                        backgroundColor: '#3b82f6'
                    }]
                },
                options: { indexAxis: 'y' }
            });

            new Chart(document.getElementById('timeChart'), {
                type: 'bar',
                data: {
                    labels: summary.map(s => s.name),
                    datasets: [{
                        label: 'Average Time (s)',
                        data: summary.map(s => s.avgTime),
                        backgroundColor: '#fb923c'
                    }]
                },
                options: { 
                    scales: { y: { type: 'logarithmic' } }
                }
            });
        }

        caseSelect.addEventListener('change', updateUI);
        algoSelect.addEventListener('change', updateUI);
        window.addEventListener('resize', updateUI);

        updateUI();
        initCharts();
    </script>
</body>
</html>
    """
    
    final_html = html_template.replace("DATA_PLACEHOLDER", json.dumps(data))
    
    with open(output_file, "w") as f:
        f.write(final_html)
        
    print(f"Dashboard generated: {output_file}")

if __name__ == "__main__":
    generate_dashboard()
