import os
from pathlib import Path
from uuid import uuid4
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from collections import defaultdict

# --- Robust, explicit folders ---
BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
# --------------------------------

# Secret key for session cookies & flashing messages
app.secret_key = os.environ.get("SECRET_KEY", "change-this-to-a-secure-random-string")

# Each browser gets a unique session id cookie. We'll keep per-session game states here.
# NOTE: This is in-memory. If the server restarts, all games reset.
# For persistence or multiple instances, use Redis/Flask-Session (I can wire that if you want).
_games: dict[str, dict] = {}

def _fresh_state() -> dict:
    """Return a new, empty game state dict."""
    return {
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

def _get_state() -> dict:
    """Fetch or create this browser's game state, keyed by a session id cookie."""
    sid = session.get("sid")
    if not sid:
        sid = uuid4().hex
        session["sid"] = sid
        session.permanent = True  # make cookie last longer (default 31 days)
    if sid not in _games:
        _games[sid] = _fresh_state()
    return _games[sid]

def _reset_state():
    """Reset ONLY this browser's game, not others."""
    sid = session.get("sid")
    if not sid:
        return
    _games[sid] = _fresh_state()

def _final_order_players(gs: dict) -> list[str]:
    """
    Compute final ordering without changing displayed totals.
    Sort key is:
      (base_total, tb_round1, tb_round2, ...)
    Lower is better at each position.
    """
    def key(player: str):
        base = int(gs['scores'].get(player, 0))
        tb_seq = gs['all_playoff_history'].get(player, [])
        return (base, *tb_seq)

    return sorted(gs['players'], key=key)

def _serialize_game(gs: dict) -> dict:
    """Return a JSON-serializable snapshot of the game state (minimal but sufficient)."""
    return {
        'players': list(gs['players']),
        'scores': dict(gs['scores']),
        'round_history': gs['round_history'],
        'current_round': gs['current_round'],
        'current_player_index': gs['current_player_index'],
        'phase': gs['phase'],
        'winner': gs['winner'],

        # playoff data
        'playoff_group': list(gs['playoff_group']),
        'playoff_base_score': gs['playoff_base_score'],
        'playoff_round': gs['playoff_round'],
        'playoff_round_scores': dict(gs['playoff_round_scores']),
        'playoff_history': list(gs['playoff_history']),

        # final display helpers
        'final_standings': list(gs['final_standings']),
        'all_playoff_history': dict(gs['all_playoff_history']),
        'max_playoff_rounds': gs['max_playoff_rounds'],
    }

@app.route('/')
def index():
    """Render the main page; build final standings once when needed."""
    gs = _get_state()

    if gs['phase'] == 'final_ranking' and not gs['final_standings']:
        ordered_players = _final_order_players(gs)

        standings = []
        last_key = None
        current_rank = 0

        for player in ordered_players:
            base_total = int(gs['scores'].get(player, 0))
            tb_seq = tuple(gs['all_playoff_history'].get(player, []))
            this_key = (base_total, tb_seq)

            if this_key != last_key:
                current_rank = len(standings) + 1  # new rank when the tuple differs

            standings.append({
                'rank': current_rank,
                'name': player,
                'score': base_total  # show BASE total only (no tie-break added)
            })
            last_key = this_key

        gs['final_standings'] = standings

        max_rounds = 0
        if gs['all_playoff_history']:
            max_rounds = max(len(h) for h in gs['all_playoff_history'].values())
        gs['max_playoff_rounds'] = max_rounds

    return render_template('index.html', game=gs)

# ---------- Classic POST endpoints (fallback if JS disabled) ----------
@app.route('/start', methods=['POST'])
def start_game():
    gs = _get_state()
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

    _reset_state()
    gs = _get_state()
    gs['players'] = players
    gs['scores'] = {player: 0 for player in players}
    gs['round_history'] = [{} for _ in range(20)]
    gs['phase'] = 'playing'
    return redirect(url_for('index'))

@app.route('/score', methods=['POST'])
def record_score():
    gs = _get_state()
    score_change = int(request.form.get('score'))
    _apply_score(gs, score_change)
    return redirect(url_for('index'))

@app.route('/undo', methods=['POST'])
def undo_last_move():
    gs = _get_state()
    _apply_undo(gs)
    return redirect(url_for('index'))

@app.route('/restart', methods=['POST'])
def restart():
    _reset_state()
    return redirect(url_for('index'))
# ---------------------------------------------------------------------

# ---------------------- JSON API (no full-page reload) ----------------
@app.post('/api/score')
def api_score():
    gs = _get_state()
    data = request.get_json(force=True, silent=True) or {}
    score_change = int(data.get('score', 0))
    _apply_score(gs, score_change)
    return jsonify({'ok': True, 'game': _serialize_game(gs)})

@app.post('/api/undo')
def api_undo():
    gs = _get_state()
    _apply_undo(gs)
    return jsonify({'ok': True, 'game': _serialize_game(gs)})
# ---------------------------------------------------------------------

# --------------------------- Internal helpers ------------------------
def _apply_score(gs: dict, score_change: int):
    if gs['phase'] == 'playing':
        player = gs['players'][gs['current_player_index']]
        gs['scores'][player] += score_change
        gs['round_history'][gs['current_round'] - 1][player] = score_change
        gs['undo_history'].append({
            'player_index': gs['current_player_index'],
            'score_change': score_change
        })

        gs['current_player_index'] += 1
        if gs['current_player_index'] >= len(gs['players']):
            gs['current_player_index'] = 0
            gs['current_round'] += 1

        # After 20 rounds, move to playoffs if needed
        if gs['current_round'] > 20:
            initiate_playoffs(gs)

    elif gs['phase'] == 'playoff':
        player = gs['playoff_group'][gs['current_player_index']]
        gs['playoff_round_scores'][player] = score_change
        gs['current_player_index'] += 1
        if gs['current_player_index'] >= len(gs['playoff_group']):
            resolve_playoff_round(gs)

def _apply_undo(gs: dict):
    if gs['phase'] != 'playing' or not gs['undo_history']:
        return
    last_move = gs['undo_history'].pop()
    prev_player_index = (gs['current_player_index'] - 1 + len(gs['players'])) % len(gs['players'])
    gs['current_player_index'] = prev_player_index
    if gs['current_player_index'] == len(gs['players']) - 1:
        gs['current_round'] -= 1
    player_to_undo = gs['players'][prev_player_index]
    gs['scores'][player_to_undo] -= last_move['score_change']
    gs['round_history'][gs['current_round'] - 1].pop(player_to_undo, None)

def initiate_playoffs(gs: dict):
    # Group players by their BASE totals (int)
    scores_to_players = defaultdict(list)
    for player, score in gs['scores'].items():
        scores_to_players[int(score)].append(player)

    gs['pending_playoffs'] = []
    for base_score, players in scores_to_players.items():
        if len(players) > 1:
            gs['pending_playoffs'].append({'score': base_score, 'players': players})

    # Sort by base score (ascending)
    gs['pending_playoffs'].sort(key=lambda p: p['score'])
    start_next_playoff(gs)

def start_next_playoff(gs: dict):
    if gs['pending_playoffs']:
        next_playoff = gs['pending_playoffs'].pop(0)
        gs['phase'] = 'playoff'
        gs['playoff_group'] = next_playoff['players']
        gs['playoff_base_score'] = next_playoff['score']
        gs['current_player_index'] = 0
        gs['playoff_round'] = 1
        gs['playoff_round_scores'] = {}
        gs['playoff_history'] = []
    else:
        gs['phase'] = 'final_ranking'
        ordered_players = _final_order_players(gs)
        gs['winner'] = ordered_players[0] if ordered_players else None

def resolve_playoff_round(gs: dict):
    scores = gs['playoff_round_scores']  # {player: tb_score}

    # Record this TB round
    gs['playoff_history'].append(scores.copy())
    for player, tb_score in scores.items():
        gs['all_playoff_history'].setdefault(player, []).append(tb_score)

    # If any tie remains (i.e., duplicate TB values), continue playoff
    if len(set(scores.values())) < len(scores):
        gs['playoff_round'] += 1
        gs['current_player_index'] = 0
        gs['playoff_round_scores'] = {}
        return

    # Tie broken: DO NOT modify base totals. Just remember the last TB scores (optional).
    for player, tb_score in scores.items():
        gs['final_playoff_scores'][player] = tb_score

    # Move to next pending tie or final ranking
    start_next_playoff(gs)
# ---------------------------------------------------------------------

if __name__ == '__main__':
    # Local dev server (Gunicorn runs the app in production)
    app.run(debug=True, host='0.0.0.0', port=5000)
