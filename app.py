import os
from pathlib import Path
from flask import (
    Flask, render_template, render_template_string,
    request, redirect, url_for, flash
)
from collections import defaultdict

# --- Robust, explicit folders (Option B) + diagnostics ---
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)

# Helpful diagnostics on startup (shows in Render logs)
print(f"[BOOT] CWD            = {Path.cwd()}")
print(f"[BOOT] BASE_DIR       = {BASE_DIR}")
print(f"[BOOT] TEMPLATES_DIR  = {TEMPLATES_DIR} (exists={TEMPLATES_DIR.exists()})")
print(f"[BOOT] STATIC_DIR     = {STATIC_DIR} (exists={STATIC_DIR.exists()})")
try:
    print(f"[BOOT] BASE_DIR list  = {sorted(p.name for p in BASE_DIR.iterdir())}")
except Exception as e:
    print(f"[BOOT] Could not list BASE_DIR: {e}")
try:
    if TEMPLATES_DIR.exists():
        print(f"[BOOT] templates/ list = {sorted(p.name for p in TEMPLATES_DIR.iterdir())}")
    else:
        print("[BOOT] templates/ does not exist")
except Exception as e:
    print(f"[BOOT] Could not list templates/: {e}")
# ---------------------------------------------------------

# Set a secret key for flashing messages (change to something secure for production)
app.secret_key = "change-this-to-a-secure-random-string"

# This dictionary will hold the entire state of our game
game_state = {
    'players': [], 'scores': {}, 'round_history': [], 'current_round': 1,
    'current_player_index': 0, 'phase': 'setup', 'winner': None, 'undo_history': [],
    # Updated state for the advanced playoff system
    'pending_playoffs': [], 'playoff_group': [], 'playoff_round': 1,
    'playoff_round_scores': {}, 'playoff_history': [], 'playoff_base_score': 0,
    'final_standings': [], 'final_playoff_scores': {}, 'all_playoff_history': {},
    'max_playoff_rounds': 0
}

def reset_game():
    """Resets the game state to its initial values."""
    global game_state
    game_state = {
        'players': [], 'scores': {}, 'round_history': [], 'current_round': 1,
        'current_player_index': 0, 'phase': 'setup', 'winner': None,
        'undo_history': [], 'pending_playoffs': [], 'playoff_group': [],
        'playoff_round': 1, 'playoff_round_scores': {}, 'playoff_history': [],
        'playoff_base_score': 0, 'final_standings': [], 'final_playoff_scores': {},
        'all_playoff_history': {}, 'max_playoff_rounds': 0
    }

@app.route('/_ls')
def _ls():
    """Debug: list base & templates dirs in JSON-ish text."""
    from json import dumps
    def listdir(p: Path):
        try:
            return sorted(x.name for x in p.iterdir())
        except Exception:
            return []
    return dumps({
        "cwd": str(Path.cwd()),
        "BASE_DIR": str(BASE_DIR),
        "BASE_DIR_exists": BASE_DIR.exists(),
        "BASE_DIR_list": listdir(BASE_DIR),
        "TEMPLATES_DIR": str(TEMPLATES_DIR),
        "TEMPLATES_DIR_exists": TEMPLATES_DIR.exists(),
        "TEMPLATES_DIR_list": listdir(TEMPLATES_DIR) if TEMPLATES_DIR.exists() else [],
        "STATIC_DIR": str(STATIC_DIR),
        "STATIC_DIR_exists": STATIC_DIR.exists(),
        "STATIC_DIR_list": listdir(STATIC_DIR) if STATIC_DIR.exists() else [],
    }, indent=2), 200, {"Content-Type": "application/json"}

@app.route('/_env')
def _env():
    """Debug: show key env vars Render sets."""
    keys = ["PORT", "RENDER", "PYTHON_VERSION", "PYTHONHOME", "PYTHONPATH"]
    return "<pre>" + "\n".join(f"{k}={os.environ.get(k)}" for k in keys) + "</pre>"

@app.route('/')
def index():
    """Renders the main game page."""
    if game_state['phase'] == 'final_ranking' and not game_state['final_standings']:
        # This logic runs once to calculate the final ranks and display scores
        player_scores = sorted([(score, player) for player, score in game_state['scores'].items()])
        standings = []
        last_score = -999
        current_rank = 0
        for score, player in player_scores:
            if score != last_score:
                current_rank = len(standings) + 1
            base_score = int(score)
            tie_breaker_score = game_state['final_playoff_scores'].get(player, 0)
            display_score = base_score + tie_breaker_score
            standings.append({'rank': current_rank, 'name': player, 'score': display_score})
            last_score = score
        game_state['final_standings'] = standings
        max_rounds = 0
        if game_state['all_playoff_history']:
            max_rounds = max(len(h) for h in game_state['all_playoff_history'].values())
        game_state['max_playoff_rounds'] = max_rounds

    # Try to render the template; if missing, show an inline fallback so we can keep the service up.
    try:
        return render_template('index.html', game=game_state)
    except Exception as e:
        return render_template_string(
            """<!doctype html>
            <html><head><meta charset="utf-8"><title>Missing template</title></head>
            <body style="font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; line-height:1.5">
              <h1>Template not found</h1>
              <p>Flask tried to load <code>templates/index.html</code> but couldn't find it.</p>
              <p><strong>Fix:</strong> Ensure your repo contains <code>templates/index.html</code> at the same level as <code>app.py</code>, with exact lowercase casing, and that it's committed (not ignored).</p>
              <h2>Details</h2>
              <pre style="background:#111;color:#eee;padding:12px;border-radius:8px;overflow:auto;">{{ details }}</pre>
              <p>Also see <a href="/_ls">/_ls</a> for live directory listing.</p>
            </body></html>""",
            details={
                "cwd": str(Path.cwd()),
                "BASE_DIR": str(BASE_DIR),
                "TEMPLATES_DIR": str(TEMPLATES_DIR),
                "templates_exists": TEMPLATES_DIR.exists(),
                "templates_contents": sorted([p.name for p in TEMPLATES_DIR.iterdir()]) if TEMPLATES_DIR.exists() else [],
                "error": repr(e),
            }
        ), 500

@app.route('/start', methods=['POST'])
def start_game():
    player_names = request.form.get('players', '')
    # Split, trim, and filter empties
    players = [name.strip() for name in player_names.split(',') if name and name.strip()]
    if not players:
        flash("Please enter at least one player name.", "error")
        return redirect(url_for('index'))

    # CHECK 1: Ensure no duplicates (case-insensitive)
    lowered = [p.lower() for p in players]
    if len(set(lowered)) != len(lowered):
        dups = sorted({name for name in players if lowered.count(name.lower()) > 1})
        flash(f"Duplicate player name(s) not allowed: {', '.join(dups)}. Please enter unique names.", "error")
        return redirect(url_for('index'))

    # If all good, reset and start
    reset_game()
    game_state['players'] = players
    game_state['scores'] = {player: 0 for player in players}
    game_state['round_history'] = [{} for _ in range(20)]
    game_state['phase'] = 'playing'
    return redirect(url_for('index'))

@app.route('/score', methods=['POST'])
def record_score():
    score_change = int(request.form.get('score'))
    if game_state['phase'] == 'playing':
        player = game_state['players'][game_state['current_player_index']]
        game_state['scores'][player] += score_change
        game_state['round_history'][game_state['current_round'] - 1][player] = score_change
        game_state['undo_history'].append({'player_index': game_state['current_player_index'], 'score_change': score_change})
        game_state['current_player_index'] += 1
        if game_state['current_player_index'] >= len(game_state['players']):
            game_state['current_player_index'] = 0
            game_state['current_round'] += 1
        if game_state['current_round'] > 20:
            initiate_playoffs()
    elif game_state['phase'] == 'playoff':
        player = game_state['playoff_group'][game_state['current_player_index']]
        game_state['playoff_round_scores'][player] = score_change
        game_state['current_player_index'] += 1
        if game_state['current_player_index'] >= len(game_state['playoff_group']):
            resolve_playoff_round()
    return redirect(url_for('index'))

def initiate_playoffs():
    scores_to_players = defaultdict(list)
    for player, score in game_state['scores'].items():
        scores_to_players[int(score)].append(player)
    game_state['pending_playoffs'] = []
    for score, players in scores_to_players.items():
        if len(players) > 1:
            game_state['pending_playoffs'].append({'score': score, 'players': players})
    game_state['pending_playoffs'].sort(key=lambda p: p['score'])
    start_next_playoff()

def start_next_playoff():
    if game_state['pending_playoffs']:
        next_playoff = game_state['pending_playoffs'].pop(0)
        game_state['phase'] = 'playoff'
        game_state['playoff_group'] = next_playoff['players']
        game_state['playoff_base_score'] = next_playoff['score']
        game_state['current_player_index'] = 0
        game_state['playoff_round'] = 1
        game_state['playoff_round_scores'] = {}
        game_state['playoff_history'] = []
    else:
        game_state['phase'] = 'final_ranking'
        if game_state['scores']:
            min_score = min(game_state['scores'].values())
            winners = [p for p, s in game_state['scores'].items() if s == min_score]
            game_state['winner'] = winners[0] if winners else None

def resolve_playoff_round():
    scores = game_state['playoff_round_scores']
    game_state['playoff_history'].append(scores.copy())
    for player, score in scores.items():
        if player not in game_state['all_playoff_history']:
            game_state['all_playoff_history'][player] = []
        game_state['all_playoff_history'][player].append(score)

    # If any tie remains (i.e., duplicate values), keep playoff going
    if len(set(scores.values())) < len(scores):
        game_state['playoff_round'] += 1
        game_state['current_player_index'] = 0
        game_state['playoff_round_scores'] = {}
        return

    # Lowest playoff score is better. Sort players by score ascending.
    sorted_players = sorted(scores.keys(), key=lambda p: scores[p])

    base_score = game_state['playoff_base_score']
    for i, player in enumerate(sorted_players):
        game_state['scores'][player] = base_score + (i + 1) * 0.01
        game_state['final_playoff_scores'][player] = scores[player]
    start_next_playoff()

@app.route('/undo', methods=['POST'])
def undo_last_move():
    if game_state['phase'] != 'playing' or not game_state['undo_history']:
        return redirect(url_for('index'))
    last_move = game_state['undo_history'].pop()
    prev_player_index = (game_state['current_player_index'] - 1 + len(game_state['players'])) % len(game_state['players'])
    game_state['current_player_index'] = prev_player_index
    if game_state['current_player_index'] == len(game_state['players']) - 1:
        game_state['current_round'] -= 1
    player_to_undo = game_state['players'][prev_player_index]
    game_state['scores'][player_to_undo] -= last_move['score_change']
    game_state['round_history'][game_state['current_round'] - 1].pop(player_to_undo, None)
    return redirect(url_for('index'))

@app.route('/restart', methods=['POST'])
def restart():
    reset_game()
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Local dev server (Gunicorn will run this app in production)
    app.run(debug=True, host='0.0.0.0', port=5000)
