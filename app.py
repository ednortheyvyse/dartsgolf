import os
from pathlib import Path
from uuid import uuid4
from collections import defaultdict
import statistics

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session

# --- Folders ---
BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
# ---------------

app.secret_key = os.environ.get("SECRET_KEY", "change-this-to-a-secure-random-string")

_games: dict[str, dict] = {}

def _fresh_state() -> dict:
    return {
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
        'max_playoff_rounds': 0,
        'recent_names': [],
        'end_after_round': False,
        'rounds_played': 0,
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
    def key(player: str):
        base = int(gs['scores'].get(player, 0))
        tb_seq = gs['all_playoff_history'].get(player, [])
        return (base, *tb_seq)
    return sorted(gs['players'], key=key)

def _compute_rounds_played(gs: dict) -> int:
    return sum(1 for r in gs.get('round_history', []) if r)

def _merge_recent(existing: list[str], new_names: list[str], cap: int = 24) -> list[str]:
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
        gs['rounds_played'] = _compute_rounds_played(gs)
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

@app.post('/api/score')
def api_score():
    gs = _get_state()
    data = request.get_json(force=True, silent=True) or {}
    score_change = int(data.get('score', 0))
    _apply_score(gs, score_change)
    return jsonify({'ok': True, 'game': gs})

@app.post('/api/undo')
def api_undo():
    gs = _get_state()
    _apply_undo(gs)
    return jsonify({'ok': True, 'game': gs})

@app.post('/api/end')
def api_end_after_round():
    gs = _get_state()
    if gs['phase'] == 'playing':
        gs['end_after_round'] = not bool(gs.get('end_after_round', False))
    return jsonify({'ok': True, 'game': gs})

def _apply_score(gs: dict, score_change: int):
    if gs['phase'] == 'playing':
        player = gs['players'][gs['current_player_index']]
        gs['scores'][player] += score_change
        gs['round_history'][gs['current_round'] - 1][player] = score_change
        gs['undo_history'].append({'player_index': gs['current_player_index'], 'score_change': score_change})

        last_index = len(gs['players']) - 1
        was_last_in_round = (gs['current_player_index'] == last_index)

        if was_last_in_round and gs.get('end_after_round'):
            initiate_playoffs(gs)
            return

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

    if prev_idx == len(gs['players']) - 1:
        gs['current_round'] -= 1

    player_to_undo = gs['players'][prev_idx]
    gs['scores'][player_to_undo] -= last_move['score_change']
    gs['round_history'][gs['current_round'] - 1].pop(player_to_undo, None)

def initiate_playoffs(gs: dict):
    gs['end_after_round'] = False

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
        gs['rounds_played'] = _compute_rounds_played(gs)
        ordered = _final_order_players(gs)
        gs['winner'] = ordered[0] if ordered else None

def resolve_playoff_round(gs: dict):
    scores = gs['playoff_round_scores']
    gs['playoff_history'].append(scores.copy())
    for p, tb in scores.items():
        gs['all_playoff_history'].setdefault(p, []).append(tb)

    # If any tie remains (duplicate TB values), continue another TB round
    if len(set(scores.values())) < len(scores):
        gs['playoff_round'] += 1
        gs['current_player_index'] = 0
        gs['playoff_round_scores'] = {}
        return

    for p, tb in scores.items():
        gs['final_playoff_scores'][p] = tb

    start_next_playoff(gs)

# -----------------------------
# Stats helpers and /stats page
# -----------------------------

def _extract_per_player_sequences(gs: dict) -> dict[str, list[int]]:
    """Return per-player list of per-round scores for rounds actually played."""
    rp = _compute_rounds_played(gs)
    rh = gs.get('round_history', [])[:rp]
    per: dict[str, list[int]] = {}
    for p in gs.get('players', []):
        per[p] = [ (rh[i].get(p, 0) if i < len(rh) else 0) for i in range(rp) ]
    return per

def _longest_streak(seq: list[int], predicate) -> int:
    best = cur = 0
    for v in seq:
        if predicate(v):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best

def _best_comebacks_by_player(per: dict[str, list[int]]):
    """
    For each player, compute their single biggest improvement (strokes saved)
    between any two rounds i<j (seq[i] - seq[j], positive only).
    Returns dict[player] -> detail dict.
    """
    out = {}
    for p, seq in per.items():
        n = len(seq)
        best = None
        for i in range(n - 1):
            for j in range(i + 1, n):
                improvement = seq[i] - seq[j]
                if improvement > 0:
                    cand = {
                        'player': p,
                        'improvement': improvement,
                        'from_score': seq[i],
                        'to_score': seq[j],
                        'from_round': i + 1,
                        'to_round': j + 1,
                    }
                    if (best is None) or (cand['improvement'] > best['improvement']):
                        best = cand
        if best:
            out[p] = best
    return out

@app.route('/stats')
def stats():
    gs = _get_state()
    per = _extract_per_player_sequences(gs)

    # Accumulators
    birdie_streaks = {}
    bogey_streaks = {}
    birdie_counts = {}
    bogey_counts = {}
    avgs = {}
    rounds_counts = {}

    for p, seq in per.items():
        birdie_streaks[p] = _longest_streak(seq, lambda x: x < 0)
        bogey_streaks[p] = _longest_streak(seq, lambda x: x > 0)
        birdie_counts[p] = sum(1 for v in seq if v < 0)
        bogey_counts[p] = sum(1 for v in seq if v > 0)
        if seq:
            avgs[p] = statistics.mean(seq)
            rounds_counts[p] = len(seq)
        else:
            avgs[p] = 0.0
            rounds_counts[p] = 0

    # Only rank players who have at least one scored round
    players_with_data = [p for p in gs.get('players', []) if rounds_counts.get(p, 0) > 0]

    # Rankings
    birdie_streak_order = sorted(
        players_with_data,
        key=lambda p: (-birdie_streaks[p], -birdie_counts[p], avgs[p])
    )
    bogey_streak_order = sorted(
        players_with_data,
        key=lambda p: (-bogey_streaks[p], -bogey_counts[p], -avgs[p])
    )
    average_order = sorted(
        players_with_data,
        key=lambda p: (avgs[p], -rounds_counts[p])
    )
    most_birdies_order = sorted(
        players_with_data,
        key=lambda p: (-birdie_counts[p], avgs[p])
    )
    most_bogeys_order = sorted(
        players_with_data,
        key=lambda p: (-bogey_counts[p], -avgs[p])
    )

    birdie_streak_ranking = [
        {'name': p, 'streak': birdie_streaks[p], 'count': birdie_counts[p], 'average': avgs[p]}
        for p in birdie_streak_order
    ]
    bogey_streak_ranking = [
        {'name': p, 'streak': bogey_streaks[p], 'count': bogey_counts[p], 'average': avgs[p]}
        for p in bogey_streak_order
    ]
    average_ranking = [
        {'name': p, 'average': avgs[p], 'rounds': rounds_counts[p]}
        for p in average_order
    ]
    most_birdies_ranking = [
        {'name': p, 'count': birdie_counts[p], 'average': avgs[p]}
        for p in most_birdies_order
    ]
    most_bogeys_ranking = [
        {'name': p, 'count': bogey_counts[p], 'average': avgs[p]}
        for p in most_bogeys_order
    ]

    # Comeback ranking (biggest stroke savings)
    best_comebacks = _best_comebacks_by_player(per)
    comeback_ranking = sorted(best_comebacks.values(), key=lambda d: d['improvement'], reverse=True)

    return render_template(
        'stats.html',
        birdie_streak_ranking=birdie_streak_ranking,
        bogey_streak_ranking=bogey_streak_ranking,
        average_ranking=average_ranking,
        most_birdies_ranking=most_birdies_ranking,
        most_bogeys_ranking=most_bogeys_ranking,
        comeback_ranking=comeback_ranking,
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
