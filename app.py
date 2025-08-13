import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from collections import defaultdict

# --- Robust, explicit folders ---
BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
# --------------------------------

# Set a secret key for flashing messages (change to something secure for production)
app.secret_key = "change-this-to-a-secure-random-string"

# Entire game state
game_state = {
    'players': [],
    'scores': {},                    # base totals (ints), tie-breakers NOT added
    'round_history': [],             # list of 20 dicts: round -> {player: score_change}
    'current_round': 1,
    'current_player_index': 0,
    'phase': 'setup',
    'winner': None,
    'undo_history': [],

    # Playoff / tie-break state
    'pending_playoffs': [],          # [{score: base_score, players: [..]}] per tied base score
    'playoff_group': [],             # players currently in a playoff
    'playoff_round': 1,
    'playoff_round_scores': {},      # current TB round {player: score}
    'playoff_history': [],           # list of dicts per TB round for current playoff group
    'playoff_base_score': 0,

    # Final displays
    'final_standings': [],           # [{rank, name, score}] with score = base total only
    'final_playoff_scores': {},      # last TB round per player (optional; kept for reference)
    'all_playoff_history': {},       # {player: [tb1, tb2, ...]}
    'max_playoff_rounds': 0          # for table rendering
}

def reset_game():
    """Reset game state to initial values."""
    global game_state
    game_state = {
        'players': [],
        'scores': {},
        'round_history': [],
        'current_round': 1,
        'current_player_index': 0,
        'phase': 'setup',
        'winner': None,
        'undo_history': [],

        'pending_playoffs': [],
        'playoff_group': [],
        'playoff_round': 1,
        'playoff_round_scores': {},
        'playoff_history': [],
        'playoff_base_score': 0,

        'final_standings': [],
        'final_playoff_scores': {},
        'all_playoff_history': {},
        'max_playoff_rounds': 0
    }

def _final_order_players():
    """
    Compute final ordering without changing displayed totals.
    Sort key is:
      (base_total, tb_round1, tb_round2, ...)
    Lower is better at each position.
    """
    def key(player: str):
        base = int(game_state['scores'].get(player, 0))
        tb_seq = game_state['all_playoff_history'].get(player, [])
        return (base, *tb_seq)

    return sorted(game_state['players'], key=key)

def _serialize_game():
    """Return a JSON-serializable snapshot of the game state (minimal but sufficient)."""
    return {
        'players': list(game_state['players']),
        'scores': dict(game_state['scores']),
        'round_history': game_state['round_history'],
        'current_round': game_state['current_round'],
        'current_player_index': game_state['current_player_index'],
        'phase': game_state['phase'],
        'winner': game_state['winner'],

        # playoff data
        'playoff_group': list(game_state['playoff_group']),
        'playoff_base_score': game_state['playoff_base_score'],
        'playoff_round': game_state['playoff_round'],
        'playoff_round_scores': dict(game_state['playoff_round_scores']),
        'playoff_history': list(game_state['playoff_history']),

        # final display helpers
        'final_standings': list(game_state['final_standings']),
        'all_playoff_history': dict(game_state['all_playoff_history']),
        'max_playoff_rounds': game_state['max_playoff_rounds'],
    }

@app.route('/')
def index():
    """Render the main page; build final standings once when needed."""
    if game_state['phase'] == 'final_ranking' and not game_state['final_standings']:
        ordered_players = _final_order_players()

        standings = []
        last_key = None
        current_rank = 0

        for player in ordered_players:
            base_total = int(game_state['scores'].get(player, 0))
            tb_seq = tuple(game_state['all_playoff_history'].get(player, []))
            this_key = (base_total, tb_seq)

            if this_key != last_key:
                current_rank = len(standings) + 1  # new rank when the tuple differs

            standings.append({
                'rank': current_rank,
                'name': player,
                'score': base_total  # show BASE total only (no tie-break added)
            })
            last_key = this_key

        game_state['final_standings'] = standings

        max_rounds = 0
        if game_state['all_playoff_history']:
            max_rounds = max(len(h) for h in game_state['all_playoff_history'].values())
        game_state['max_playoff_rounds'] = max_rounds

    return render_template('index.html', game=game_state)

# ---------- Classic POST endpoints (still work if JS is disabled) ----------
@app.route('/start', methods=['POST'])
def start_game():
    player_names = request.form.get('players', '')
    players = [name.strip() for name in player_names.split(',') if name and name.strip()]
    if not players:
        flash("Please enter at least one player name.", "error")
        return redirect(url_for('index'))

    lowered = [p.lower() for p in players]
    if len(set(lowered)) != len(lowered):
        dups = sorted({name for name in players if lowered.count(name.lower()) > 1})
        flash(f"Duplicate player name(s) not allowed: {', '.join(dups)}. Please enter unique names.", "error")
        return redirect(url_for('index'))

    reset_game()
    game_state['players'] = players
    game_state['scores'] = {player: 0 for player in players}
    game_state['round_history'] = [{} for _ in range(20)]
    game_state['phase'] = 'playing'
    return redirect(url_for('index'))

@app.route('/score', methods=['POST'])
def record_score():
    score_change = int(request.form.get('score'))
    _apply_score(score_change)
    return redirect(url_for('index'))

@app.route('/undo', methods=['POST'])
def undo_last_move():
    _apply_undo()
    return redirect(url_for('index'))

@app.route('/restart', methods=['POST'])
def restart():
    reset_game()
    return redirect(url_for('index'))
# ---------------------------------------------------------------------------

# ---------------------- JSON API (no full-page reload) ---------------------
@app.post('/api/score')
def api_score():
    data = request.get_json(force=True, silent=True) or {}
    score_change = int(data.get('score', 0))
    _apply_score(score_change)
    return jsonify({'ok': True, 'game': _serialize_game()})

@app.post('/api/undo')
def api_undo():
    _apply_undo()
    return jsonify({'ok': True, 'game': _serialize_game()})
# ---------------------------------------------------------------------------

# --------------------------- Internal helpers ------------------------------
def _apply_score(score_change: int):
    if game_state['phase'] == 'playing':
        player = game_state['players'][game_state['current_player_index']]
        game_state['scores'][player] += score_change
        game_state['round_history'][game_state['current_round'] - 1][player] = score_change
        game_state['undo_history'].append({
            'player_index': game_state['current_player_index'],
            'score_change': score_change
        })

        game_state['current_player_index'] += 1
        if game_state['current_player_index'] >= len(game_state['players']):
            game_state['current_player_index'] = 0
            game_state['current_round'] += 1

        # After 20 rounds, move to playoffs if needed
        if game_state['current_round'] > 20:
            initiate_playoffs()

    elif game_state['phase'] == 'playoff':
        player = game_state['playoff_group'][game_state['current_player_index']]
        game_state['playoff_round_scores'][player] = score_change
        game_state['current_player_index'] += 1
        if game_state['current_player_index'] >= len(game_state['playoff_group']):
            resolve_playoff_round()

def _apply_undo():
    if game_state['phase'] != 'playing' or not game_state['undo_history']:
        return
    last_move = game_state['undo_history'].pop()
    prev_player_index = (game_state['current_player_index'] - 1 + len(game_state['players'])) % len(game_state['players'])
    game_state['current_player_index'] = prev_player_index
    if game_state['current_player_index'] == len(game_state['players']) - 1:
        game_state['current_round'] -= 1
    player_to_undo = game_state['players'][prev_player_index]
    game_state['scores'][player_to_undo] -= last_move['score_change']
    game_state['round_history'][game_state['current_round'] - 1].pop(player_to_undo, None)

def initiate_playoffs():
    # Group players by their BASE totals (int)
    scores_to_players = defaultdict(list)
    for player, score in game_state['scores'].items():
        scores_to_players[int(score)].append(player)

    game_state['pending_playoffs'] = []
    for base_score, players in scores_to_players.items():
        if len(players) > 1:
            game_state['pending_playoffs'].append({'score': base_score, 'players': players})

    # Sort by base score (ascending)
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
        ordered_players = _final_order_players()
        game_state['winner'] = ordered_players[0] if ordered_players else None

def resolve_playoff_round():
    scores = game_state['playoff_round_scores']  # {player: tb_score}

    # Record this TB round
    game_state['playoff_history'].append(scores.copy())
    for player, tb_score in scores.items():
        game_state['all_playoff_history'].setdefault(player, []).append(tb_score)

    # If any tie remains (i.e., duplicate TB values), continue playoff
    if len(set(scores.values())) < len(scores):
        game_state['playoff_round'] += 1
        game_state['current_player_index'] = 0
        game_state['playoff_round_scores'] = {}
        return

    # Tie broken: DO NOT modify base totals. Just remember the last TB scores (optional).
    for player, tb_score in scores.items():
        game_state['final_playoff_scores'][player] = tb_score

    # Move to next pending tie or final ranking
    start_next_playoff()
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Local dev server (Gunicorn runs the app in production)
    app.run(debug=True, host='0.0.0.0', port=5000)
