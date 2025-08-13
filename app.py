import os
from pathlib import Path
from uuid import uuid4
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from collections import defaultdict

# --- Folders ---
BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
# ---------------

app.secret_key = os.environ.get("SECRET_KEY", "change-this-to-a-secure-random-string")

# In-memory, per-browser games (resets on server restart)
_games: dict[str, dict] = {}

def _fresh_state() -> dict:
    return {
        'players': [],
        'scores': {},                    # base totals only
        'round_history': [],             # 20 dicts: round -> {player: delta}
        'current_round': 1,
        'current_player_index': 0,
        'phase': 'setup',
        'winner': None,
        'undo_history': [],

        # Tie-break / playoff
        'pending_playoffs': [],
        'playoff_group': [],
        'playoff_round': 1,
        'playoff_round_scores': {},
        'playoff_history': [],
        'playoff_base_score': 0,

        # Final display
        'final_standings': [],
        'final_playoff_scores': {},
        'all_playoff_history': {},
        'max_playoff_rounds': 0,

        # Per-browser convenience
        'recent_names': [],              # most-recent first, unique (case-insensitive)
    }

def _get_state() -> dict:
    sid = session.get("sid")
    if not sid:
        sid = uuid4().hex
        session["sid"] = sid
        session.permanent = True
    if sid not in _games:
        _games[sid] = _fresh_state()
    return _games[sid]

def _reset_state():
    sid = session.get("sid")
    if not sid:
        return
    _games[sid] = _fresh_state()

def _final_order_players(gs: dict) -> list[str]:
    # Sort by (base_total, tb1, tb2, ...) — lower is better
    def key(player: str):
        base = int(gs['scores'].get(player, 0))
        tb_seq = gs['all_playoff_history'].get(player, [])
        return (base, *tb_seq)
    return sorted(gs['players'], key=key)

def _serialize_game(gs: dict) -> dict:
    return {
        'players': list(gs['players']),
        'scores': dict(gs['scores']),
        'round_history': gs['round_history'],
        'current_round': gs['current_round'],
        'current_player_index': gs['current_player_index'],
        'phase': gs['phase'],
        'winner': gs['winner'],
        'playoff_group': list(gs['playoff_group']),
        'playoff_base_score': gs['playoff_base_score'],
        'playoff_round': gs['playoff_round'],
        'playoff_round_scores': dict(gs['playoff_round_scores']),
        'playoff_history': list(gs['playoff_history']),
        'final_standings': list(gs['final_standings']),
        'all_playoff_history': dict(gs['all_playoff_history']),
        'max_playoff_rounds': gs['max_playoff_rounds'],
        'recent_names': list(gs.get('recent_names', [])),
    }

def _merge_recent(existing: list[str], new_names: list[str], cap: int = 24) -> list[str]:
    # Case-insensitive dedupe, preserve first casing encountered
    out: list[str] = []
    seen_lower = set()
    for name in new_names + existing:
        k = name.lower()
        if k in seen_lower:
            continue
        out.append(name)
        seen_lower.add(k)
        if len(out) >= cap:
            break
    return out

@app.route('/')
def index():
    gs = _get_state()

    if gs['phase'] == 'final_ranking' and not gs['final_standings']:
        ordered = _final_order_players(gs)
        standings = []
        last_key = None
        current_rank = 0
        for p in ordered:
            base_total = int(gs['scores'].get(p, 0))
            tb_seq = tuple(gs['all_playoff_history'].get(p, []))
            this_key = (base_total, tb_seq)
            if this_key != last_key:
                current_rank = len(standings) + 1
            standings.append({'rank': current_rank, 'name': p, 'score': base_total})
            last_key = this_key

        gs['final_standings'] = standings
        gs['max_playoff_rounds'] = max((len(h) for h in gs['all_playoff_history'].values()), default=0)

    return render_template('index.html', game=gs)

# ---------- Fallback form posts ----------
@app.route('/start', methods=['POST'])
def start_game():
    gs = _get_state()
    players_raw = request.form.get('players', '')
    players = [n.strip() for n in players_raw.split(',') if n and n.strip()]
    if not players:
        flash("Please enter at least one player name.", "error")
        return redirect(url_for('index'))

    lowered = [p.lower() for p in players]
    if len(set(lowered)) != len(lowered):
        dups = sorted({name for name in players if lowered.count(name.lower()) > 1})
        flash(f"Duplicate player name(s) not allowed: {', '.join(dups)}. Please enter unique names.", "error")
        return redirect(url_for('index'))

    # Update recent (persist through reset)
    updated_recent = _merge_recent(gs.get('recent_names', []), players, cap=24)

    _reset_state()
    gs = _get_state()
    gs['players'] = players
    gs['scores'] = {p: 0 for p in players}
    gs['round_history'] = [{} for _ in range(20)]
    gs['phase'] = 'playing'
    gs['recent_names'] = updated_recent
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
    gs = _get_state()
    prev_recent = gs.get('recent_names', [])
    _reset_state()
    gs = _get_state()
    gs['recent_names'] = prev_recent
    return redirect(url_for('index'))
# -----------------------------------------

# ---------- JSON API (no full-page reload) ----------
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
# ----------------------------------------------------

# ---------------- Helpers ----------------
def _apply_score(gs: dict, score_change: int):
    if gs['phase'] == 'playing':
        player = gs['players'][gs['current_player_index']]
        gs['scores'][player] += score_change
        gs['round_history'][gs['current_round'] - 1][player] = score_change
        gs['undo_history'].append({'player_index': gs['current_player_index'], 'score_change': score_change})

        gs['current_player_index'] += 1
        if gs['current_player_index'] >= len(gs['players']):
            gs['current_player_index'] = 0
            gs['current_round'] += 1

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
    prev_idx = (gs['current_player_index'] - 1 + len(gs['players'])) % len(gs['players'])
    gs['current_player_index'] = prev_idx
    if gs['current_player_index'] == len(gs['players']) - 1:
        gs['current_round'] -= 1
    player_to_undo = gs['players'][prev_idx]
    gs['scores'][player_to_undo] -= last_move['score_change']
    gs['round_history'][gs['current_round'] - 1].pop(player_to_undo, None)

def initiate_playoffs(gs: dict):
    scores_to_players = defaultdict(list)
    for p, s in gs['scores'].items():
        scores_to_players[int(s)].append(p)

    gs['pending_playoffs'] = []
    for base_score, players in scores_to_players.items():
        if len(players) > 1:
            gs['pending_playoffs'].append({'score': base_score, 'players': players})

    gs['pending_playoffs'].sort(key=lambda p: p['score'])
    start_next_playoff(gs)

def start_next_playoff(gs: dict):
    if gs['pending_playoffs']:
        nxt = gs['pending_playoffs'].pop(0)
        gs['phase'] = 'playoff'
        gs['playoff_group'] = nxt['players']
        gs['playoff_base_score'] = nxt['score']
        gs['current_player_index'] = 0
        gs['playoff_round'] = 1
        gs['playoff_round_scores'] = {}
        gs['playoff_history'] = []
    else:
        gs['phase'] = 'final_ranking'
        ordered = _final_order_players(gs)
        gs['winner'] = ordered[0] if ordered else None

def resolve_playoff_round(gs: dict):
    scores = gs['playoff_round_scores']  # {player: tb_score}
    gs['playoff_history'].append(scores.copy())
    for p, tb in scores.items():
        gs['all_playoff_history'].setdefault(p, []).append(tb)

    if len(set(scores.values())) < len(scores):
        gs['playoff_round'] += 1
        gs['current_player_index'] = 0
        gs['playoff_round_scores'] = {}
        return

    for p, tb in scores.items():
        gs['final_playoff_scores'][p] = tb

    start_next_playoff(gs)
# ----------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
