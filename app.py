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
        'playoff_group': [],            # CURRENT active tie subgroup playing a TB round
        'playoff_pool': [],             # All unresolved players for current base-score tie
        'playoff_round': 1,
        'playoff_round_scores': {},
        'playoff_history': [],          # list[ dict[player]->tb_score ] for this tie only
        'playoff_base_score': 0,
        'final_standings': [],
        'final_playoff_scores': {},
        'all_playoff_history': {},      # player -> list of ALL TB results across ties (in order)
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
    # Lower base score is better; for ties, compare the sequence of tie-breaker results lexicographically (lower is better).
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
        # In playoff mode, current_player_index iterates over the ACTIVE tie subgroup only
        player = gs['playoff_group'][gs['current_player_index']]
        gs['playoff_round_scores'][player] = score_change
        gs['current_player_index'] += 1
        if gs['current_player_index'] >= len(gs['playoff_group']):
            resolve_playoff_round(gs)

def _apply_undo(gs: dict):
    # Undo applies only during main play (not in playoff phase)
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

    # Group players by base total score; only ties (len > 1) need playoffs
    scores_to_players = defaultdict(list)
    for p, s in gs['scores'].items():
        scores_to_players[int(s)].append(p)

    gs['pending_playoffs'] = []
    for base_score, players in scores_to_players.items():
        if len(players) > 1:
            gs['pending_playoffs'].append({'score': base_score, 'players': players})

    # Resolve from worst totals to best totals (higher total is worse)
    gs['pending_playoffs'].sort(key=lambda p: p['score'], reverse=True)
    start_next_playoff(gs)

def start_next_playoff(gs: dict):
    if gs['pending_playoffs']:
        nxt = gs['pending_playoffs'].pop(0)
        gs['phase'] = 'playoff'
        # Active tie pool = everyone tied at this base score. The active subgroup = everyone initially.
        gs['playoff_pool'] = list(nxt['players'])
        gs['playoff_group'] = list(nxt['players'])
        gs['playoff_base_score'] = nxt['score']
        gs['current_player_index'] = 0
        gs['playoff_round'] = 1
        gs['playoff_round_scores'] = {}
        gs['playoff_history'] = []
    else:
        # No more ties to resolve; produce final ranking
        gs['phase'] = 'final_ranking'
        gs['rounds_played'] = _compute_rounds_played(gs)
        ordered = _final_order_players(gs)
        gs['winner'] = ordered[0] if ordered else None

def _tb_sequence_for_player_in_current_tie(gs: dict, player: str) -> list[int]:
    """
    For the CURRENT tie only (gs['playoff_history']), return the list of TB scores
    that 'player' has recorded so far in this tie.
    """
    seq: list[int] = []
    for rnd in gs['playoff_history']:
        if player in rnd:
            seq.append(int(rnd[player]))
    return seq

def _finalize_player_from_current_tie(gs: dict, player: str):
    """
    Mark 'player' as resolved in the current tie. Store their last TB score
    (if any) for convenience (display not required).
    """
    seq = _tb_sequence_for_player_in_current_tie(gs, player)
    if seq:
        gs['final_playoff_scores'][player] = seq[-1]
    else:
        gs['final_playoff_scores'][player] = 0  # never needed, but keep consistent
    if player in gs['playoff_pool']:
        gs['playoff_pool'].remove(player)

def resolve_playoff_round(gs: dict):
    """
    Resolve one TB round for the ACTIVE subgroup (gs['playoff_group']).
    - Append scores to history and all_playoff_history.
    - Compute worst-first elimination:
        * If a unique worst sequence exists among the unresolved pool, finalize it (possibly multiple times in one pass).
        * If the worst is tied, set playoff_group to only those tied-at-worst players for the next round.
        * Continue until the entire tie is resolved, then move to the next pending tie or final ranking.
    """
    scores = gs['playoff_round_scores']
    # Record this round into the tie's history
    gs['playoff_history'].append(scores.copy())

    # Append to global per-player TB history (used for final ordering & display)
    for p, tb in scores.items():
        gs['all_playoff_history'].setdefault(p, []).append(int(tb))

    # Reset per-round collectors for the next step
    gs['playoff_round_scores'] = {}
    gs['current_player_index'] = 0

    # Helper to compute worst-at-this-point among the unresolved pool using lexicographic compare of TB sequences
    def worst_subgroup_in_pool() -> list[str]:
        pool = list(gs['playoff_pool'])
        if not pool:
            return []

        # Build sequences for current tie
        seqs = {p: tuple(_tb_sequence_for_player_in_current_tie(gs, p)) for p in pool}
        # Worst means lexicographically largest (because higher numbers are worse)
        worst_seq = max(seqs.values()) if seqs else tuple()
        worst_players = [p for p, s in seqs.items() if s == worst_seq]
        return worst_players

    # After adding this round, try to resolve as much as possible without forcing new throws
    while True:
        worst_players = worst_subgroup_in_pool()

        if not worst_players:
            # All resolved for this tie
            start_next_playoff(gs)
            return

        if len(worst_players) == 1:
            # Unique worst: finalize immediately and loop again (maybe multiple eliminations per round)
            loser = worst_players[0]
            _finalize_player_from_current_tie(gs, loser)
            # If we just finalized the last remaining, finish this tie
            if not gs['playoff_pool']:
                start_next_playoff(gs)
                return
            # Continue loop to see if the next worst is also unique (no extra throws needed)
            continue
        else:
            # There is a tie for worst among some players; only they continue to the next TB round
            gs['playoff_group'] = worst_players
            gs['playoff_round'] += 1
            # Next client interactions will collect just these players' scores
            return

@app.route('/stats')
def stats():
    gs = _get_state()

    # Build per-player sequence of per-round scores (main game, not TBs)
    def _extract_per_player_sequences(gs: dict) -> dict[str, list[int]]:
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
        out = {}
        for p, seq in per.items():
            n = len(seq)
            best = None
            for i in range(n - 1):
                for j in range(i + 1, n):
                    improvement = seq[i] - seq[j]  # positive if j is lower/better than i
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

    per = _extract_per_player_sequences(gs)

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

    players_with_data = [p for p in gs.get('players', []) if rounds_counts.get(p, 0) > 0]

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
