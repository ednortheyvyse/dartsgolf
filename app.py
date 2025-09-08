import os
import json
from pathlib import Path
from uuid import uuid4
from collections import defaultdict
import statistics

from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify,
    session, send_from_directory, abort
)

# ---------------- App & Folders ----------------
BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Security / cookie hardening (10)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-to-a-secure-random-string")
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(int(os.environ.get("SESSION_COOKIE_SECURE", "0"))),  # set 1 in prod
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30,  # 30 days
)

# Optional Redis persistence (10)
# If REDIS_URL is set and redis library available, persist games there.
# Fallback: in-memory dict (dev).
_redis = None
try:
    import redis  # type: ignore
    REDIS_URL = os.environ.get("REDIS_URL")
    if REDIS_URL:
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
except Exception:
    _redis = None

_games: dict[str, dict] = {}  # in-memory fallback


def _fresh_state() -> dict:
    # Settings defaults (7)
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
        'playoff_pool': [],
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

        # ---- Settings (7) ----
        'holes': 20,
        'score_buttons': [-3, -2, -1, 0, 1, 2],
    }


# ------------ Storage helpers (10) ------------
def _storage_key(sid: str) -> str:
    return f"game:{sid}"


def _storage_get(sid: str) -> dict | None:
    if _redis:
        data = _redis.get(_storage_key(sid))
        return json.loads(data) if data else None
    return _games.get(sid)


def _storage_set(sid: str, gs: dict) -> None:
    if _redis:
        _redis.set(_storage_key(sid), json.dumps(gs))
    else:
        _games[sid] = gs


def _storage_reset(sid: str) -> None:
    if _redis:
        _redis.delete(_storage_key(sid))
    _games.pop(sid, None)


# -------------- Game accessors --------------
def _get_sid() -> str:
    sid = session.get("sid")
    if not sid:
        sid = uuid4().hex
        session["sid"] = sid
        session.permanent = True
    return sid


def _get_state() -> dict:
    sid = _get_sid()
    gs = _storage_get(sid)
    if not gs:
        gs = _fresh_state()
        _storage_set(sid, gs)
    return gs


def _reset_state():
    sid = _get_sid()
    _storage_set(sid, _fresh_state())


def _persist():
    sid = _get_sid()
    gs = _get_state()
    _storage_set(sid, gs)


def _final_order_players(gs: dict) -> list[str]:
    # Lower base score is better; for ties, compare TB sequence lexicographically (lower is better).
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


# --------------------- Routes ---------------------
@app.route('/')
def index():
    gs = _get_state()

    # Build final standings if we just entered final_ranking
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
        _persist()

    return render_template('index.html', game=gs, show_stats=False)


# NEW: tolerate GET /start (prefetches / SW / crawlers) by redirecting home
@app.route('/start', methods=['GET'])
def start_game_get():
    return redirect(url_for('index'))


@app.route('/start', methods=['POST'])
def start_game():
    gs_prev = _get_state()
    # allow carrying settings chosen in setup (holes/buttons)
    preserved_holes = int(gs_prev.get('holes', 20))
    preserved_buttons = list(gs_prev.get('score_buttons', [-3, -2, -1, 0, 1, 2]))

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

    updated_recent = _merge_recent(gs_prev.get('recent_names', []), players, cap=24)

    _reset_state()
    gs = _get_state()
    gs['players'] = players
    gs['scores'] = {p: 0 for p in players}
    gs['holes'] = max(1, int(preserved_holes))
    gs['score_buttons'] = preserved_buttons[:]
    gs['round_history'] = [{} for _ in range(gs['holes'])]
    gs['phase'] = 'playing'
    gs['recent_names'] = updated_recent
    _persist()
    return redirect(url_for('index'))


@app.route('/score', methods=['POST'])
def record_score():
    gs = _get_state()
    score_change = int(request.form.get('score'))
    _apply_score(gs, score_change)
    _persist()
    return redirect(url_for('index'))


@app.route('/undo', methods=['POST'])
def undo_last_move():
    gs = _get_state()
    _apply_undo(gs)
    _persist()
    return redirect(url_for('index'))


@app.route('/restart', methods=['POST'])
def restart():
    gs = _get_state()
    prev_recent = gs.get('recent_names', [])
    prev_holes = gs.get('holes', 20)
    prev_buttons = gs.get('score_buttons', [-3, -2, -1, 0, 1, 2])
    _reset_state()
    gs = _get_state()
    gs['recent_names'] = prev_recent
    gs['holes'] = prev_holes
    gs['score_buttons'] = prev_buttons
    _persist()
    return redirect(url_for('index'))


# ---------- JSON APIs ----------
@app.post('/api/score')
def api_score():
    gs = _get_state()
    data = request.get_json(force=True, silent=True) or {}
    score_change = int(data.get('score', 0))
    _apply_score(gs, score_change)
    _persist()
    return jsonify({'ok': True, 'game': gs})


@app.post('/api/undo')
def api_undo():
    gs = _get_state()
    _apply_undo(gs)
    _persist()
    return jsonify({'ok': True, 'game': gs})


@app.post('/api/end')
def api_end_after_round():
    gs = _get_state()
    if gs['phase'] == 'playing':
        gs['end_after_round'] = not bool(gs.get('end_after_round', False))
        _persist()
    return jsonify({'ok': True, 'game': gs})


@app.post('/api/settings')
def api_settings():
    """
    Update game settings (7).
    - holes: only allowed in setup phase (to prevent corruption mid-game)
    - score_buttons: allowed anytime
    """
    gs = _get_state()
    data = request.get_json(force=True, silent=True) or {}
    changed = False

    # Score buttons
    if 'score_buttons' in data:
        try:
            raw = data['score_buttons']
            if isinstance(raw, list):
                btns = []
                for v in raw:
                    iv = int(v)
                    if -10 <= iv <= 10:
                        btns.append(iv)
                # normalize & de-dup, keep order
                seen = set()
                norm = []
                for b in btns:
                    if b not in seen:
                        seen.add(b)
                        norm.append(b)
                if 2 <= len(norm) <= 11:
                    gs['score_buttons'] = norm
                    changed = True
        except Exception:
            pass

    # Holes (only pre-game)
    if gs['phase'] == 'setup' and 'holes' in data:
        try:
            holes = int(data['holes'])
            if 1 <= holes <= 50:
                gs['holes'] = holes
                gs['round_history'] = [{} for _ in range(holes)]
                changed = True
        except Exception:
            pass

    if changed:
        _persist()
    return jsonify({'ok': True, 'game': gs, 'changed': changed})


@app.post('/api/load_saved')
def api_load_saved():
    """
    Replace server state with a previously saved client copy (4).
    Accept only whitelisted keys to avoid injection of arbitrary data.
    """
    client = request.get_json(force=True, silent=True) or {}
    allowed = set(_fresh_state().keys())
    filtered = {k: v for k, v in client.items() if k in allowed}
    if not isinstance(filtered.get('players', []), list):
        return jsonify({'ok': False, 'error': 'invalid payload'}), 400
    sid = _get_sid()
    _storage_set(sid, filtered)
    return jsonify({'ok': True})


# ---------- Service worker at root (explicit, no-cache) ----------
@app.route('/sw.js')
def sw():
    """
    Serve the service worker at the origin scope with no caching.
    Prefer a root-level sw.js (same dir as app.py) if present;
    otherwise fall back to /static/sw.js.
    """
    root_sw = BASE_DIR / 'sw.js'
    if root_sw.exists():
        return send_from_directory(
            BASE_DIR, 'sw.js',
            mimetype='application/javascript',
            max_age=0
        )
    return send_from_directory(
        app.static_folder, 'sw.js',
        mimetype='application/javascript',
        max_age=0
    )


# ---------- Small health/ops and icon convenience routes ----------
@app.route('/healthz', methods=['GET', 'HEAD'])
def healthz():
    return ("ok", 200)


@app.route('/favicon.ico')
def favicon():
    fav = BASE_DIR / 'static' / 'favicon.ico'
    if fav.exists():
        return send_from_directory(app.static_folder, 'favicon.ico')
    abort(404)


# iOS will probe these even if you don't link them. Point them at the 180px icon.
@app.route('/apple-touch-icon.png')
@app.route('/apple-touch-icon-precomposed.png')
def apple_touch_icon():
    png_180 = BASE_DIR / 'static' / 'icons' / 'dartboard-180.png'
    if png_180.exists():
        return send_from_directory(app.static_folder, 'icons/dartboard-180.png')
    abort(404)


# ----------------- Game logic -----------------
def _apply_score(gs: dict, score_change: int):
    holes = int(gs.get('holes', 20))
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

        if gs['current_round'] > holes:
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
        if gs['current_round'] < 1:
            gs['current_round'] = 1

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
    seq: list[int] = []
    for rnd in gs['playoff_history']:
        if player in rnd:
            seq.append(int(rnd[player]))
    return seq


def _finalize_player_from_current_tie(gs: dict, player: str):
    seq = _tb_sequence_for_player_in_current_tie(gs, player)
    if seq:
        gs['final_playoff_scores'][player] = seq[-1]
    else:
        gs['final_playoff_scores'][player] = 0
    if player in gs['playoff_pool']:
        gs['playoff_pool'].remove(player)


def resolve_playoff_round(gs: dict):
    scores = gs['playoff_round_scores']
    gs['playoff_history'].append(scores.copy())

    for p, tb in scores.items():
        gs['all_playoff_history'].setdefault(p, []).append(int(tb))

    gs['playoff_round_scores'] = {}
    gs['current_player_index'] = 0

    def worst_subgroup_in_pool() -> list[str]:
        pool = list(gs['playoff_pool'])
        if not pool:
            return []
        seqs = {p: tuple(_tb_sequence_for_player_in_current_tie(gs, p)) for p in pool}
        worst_seq = max(seqs.values()) if seqs else tuple()
        worst_players = [p for p, s in seqs.items() if s == worst_seq]
        return worst_players

    while True:
        worst_players = worst_subgroup_in_pool()
        if not worst_players:
            start_next_playoff(gs)
            return

        if len(worst_players) == 1:
            loser = worst_players[0]
            _finalize_player_from_current_tie(gs, loser)
            if not gs['playoff_pool']:
                start_next_playoff(gs)
                return
            continue
        else:
            gs['playoff_group'] = worst_players
            gs['playoff_round'] += 1
            return


# ---------------- Stats view (unchanged) ----------------
@app.route('/stats')
def stats():
    gs = _get_state()

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
        'index.html',
        show_stats=True,
        game=gs,
        birdie_streak_ranking=birdie_streak_ranking,
        bogey_streak_ranking=bogey_streak_ranking,
        average_ranking=average_ranking,
        most_birdies_ranking=most_birdies_ranking,
        most_bogeys_ranking=most_bogeys_ranking,
        comeback_ranking=comeback_ranking,
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
