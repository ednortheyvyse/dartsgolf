import os
import json
from pathlib import Path
from uuid import uuid4
from collections import defaultdict
import statistics
import numpy as np

from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify,
    session, send_from_directory, abort
)
import logging

# Configure logging to show DEBUG messages from all loggers
logging.basicConfig(level=logging.DEBUG)

# ---------------- App & Folders ----------------
BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Security / cookie hardening (10)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-to-a-secure-random-string-in-production")
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
        logging.info("REDIS_URL found, attempting to connect to Redis...")
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
        _redis.ping() # Check connection
        logging.info("Redis connection successful.")
    else:
        logging.warning("REDIS_URL not set. Falling back to in-memory storage.")
except Exception as e:
    _redis = None
    logging.error(f"Redis connection failed: {e}. Falling back to in-memory storage.")

_games: dict[str, dict] = {}  # in-memory fallback


# --- Constants ---
DEFAULT_HOLES = 20
DEFAULT_SCORE_BUTTONS = [-3, -2, -1, 0, 1, 2]
MIN_HOLES = 1
MAX_HOLES = 50
MIN_SCORE_BUTTON_VALUE = -10
MAX_SCORE_BUTTON_VALUE = 10
RECENT_NAMES_CAP = 24
RUDENESS_LABELS = [
    # Level 0: Serious
    {-3: "Albatross", -2: "Eagle", -1: "Birdie", 0: "Par", 1: "Bogey", 2: "Double Bogey"},
    # Level 1: Medium
    {-3: "Amazing!", -2: "Incredible!", -1: "Nice one!", 0: "Hell Yeah Brother", 1: "Oof", 2: "Yikes"},
    # Level 2: Rude
    {
        -3: "Fucking Magical", -2: "Holy Shit!", -1: "Fuck Yeah!", 0: "Meh",
        1: "Fucksake", 2: "CUNT!"
    },
]




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
        'holes': DEFAULT_HOLES,
        'score_buttons': DEFAULT_SCORE_BUTTONS,
        'rudeness_level': 0,
    }


# ------------ Storage helpers (10) ------------
def _storage_key(sid: str) -> str:
    """Generates a storage key for a given session ID."""
    return f"game:{sid}"


def _storage_get(sid: str) -> dict | None:
    """
    Retrieves game state from storage for a given session ID.
    Returns None if not found.
    """
    if _redis:
        data = _redis.get(_storage_key(sid))
        gs = json.loads(data) if data else None # type: ignore
        logging.debug(f"[_storage_get] (Redis) Loaded state for SID {sid}. Playoff round scores: {gs.get('playoff_round_scores') if gs else 'N/A'}")
        return gs
    else:
        gs = _games.get(sid)
        logging.debug(f"[_storage_get] (In-memory) Loaded state for SID {sid}. Playoff round scores: {gs.get('playoff_round_scores') if gs else 'N/A'}")
        return gs


def _storage_set(sid: str, gs: dict) -> None:
    """
    Stores game state for a given session ID.
    
    Args:
        sid: The session ID.
        gs: The game state dictionary to store.
    """
    if _redis:
        logging.debug(f"[_storage_set] (Redis) Saving state for SID {sid}. Playoff round scores: {gs.get('playoff_round_scores')}")
        _redis.set(_storage_key(sid), json.dumps(gs))
    else:
        logging.debug(f"[_storage_set] (In-memory) Saving state for SID {sid}. Playoff round scores: {gs.get('playoff_round_scores')}")
        _games[sid] = gs


def _storage_reset(sid: str) -> None:
    """Resets (deletes) game state for a given session ID."""
    logging.debug(f"[_storage_reset] Resetting state for SID {sid}")
    if _redis:
        _redis.delete(_storage_key(sid))
    _games.pop(sid, None)


# -------------- Game accessors --------------
def _get_sid() -> str:
    """Retrieves or generates a unique session ID for the current user."""
    sid = session.get("sid")
    if not sid:
        sid = uuid4().hex
        logging.info(f"New session created with SID: {sid}")
        session["sid"] = sid
        session.permanent = True
    return sid


def _get_state() -> dict:
    """Retrieves the current game state, initializing it if not found."""
    sid = _get_sid()
    gs = _storage_get(sid)
    if not gs:
        gs: dict = _fresh_state() # type: ignore
        logging.info(f"No state found for SID {sid}. Creating fresh state.")
        _storage_set(sid, gs)
    # Defensive: ensure playoff_round_scores is always a dictionary
    if not isinstance(gs.get('playoff_round_scores'), dict):
        gs['playoff_round_scores'] = {}
        _storage_set(sid, gs) # Persist this correction
    return gs


def _reset_state():
    sid = _get_sid()
    _storage_set(sid, _fresh_state())


def _persist(gs: dict) -> None:
    """Persists the current game state to storage."""
    # This function now correctly receives the modified gs object
    # and saves it, instead of re-fetching from storage.
    sid = _get_sid()
    _storage_set(sid, gs) # This line is correct, the issue was in the calling functions.


def _compute_rounds_played(gs: dict) -> int:
    """Calculates the number of rounds actually played (i.e., with scores)."""
    return sum(1 for r in gs.get('round_history', []) if r)


def _merge_recent(existing: list[str], new_names: list[str], cap: int = RECENT_NAMES_CAP) -> list[str]:
    """Merges new player names into a list of recent names, handling duplicates and capping the list."""
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


def _final_order_players(gs: dict) -> list[str]:
    # Lower base score is better; for ties, compare TB sequence lexicographically (lower is better).
    def key(player: str):
        base = int(gs['scores'].get(player, 0))
        tb_seq = gs['all_playoff_history'].get(player, [])
        return (base, *tb_seq)
    return sorted(gs['players'], key=key)


# --------------------- Routes ---------------------
@app.route('/')
def index():
    logging.debug(f"Route: GET / - Rendering main page.")
    gs = _get_state()
    
    # Retrieve and clear any player list from a failed form submission
    previous_players = session.pop('previous_players_input', None)

    # Inject the correct score labels based on the current rudeness level
    level = gs.get('rudeness_level', 0)
    gs['score_labels'] = RUDENESS_LABELS[level]

    # Build final standings if we just entered final_ranking
    if gs['phase'] == 'final_ranking' and not gs['final_standings']:
        logging.info("Phase is 'final_ranking'. Computing and persisting final standings.")
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
        _persist(gs)

    return render_template('index.html', game=gs, show_stats=False, previous_players=previous_players)


# NEW: tolerate GET /start (prefetches / SW / crawlers) by redirecting home
@app.route('/start', methods=['GET'])
def start_game_get():
    logging.debug("Route: GET /start - Redirecting to index.")
    return redirect(url_for('index'))


@app.route('/start', methods=['POST'])
def start_game():
    logging.info("Route: POST /start - Starting new game.")
    gs_prev = _get_state()
    # allow carrying settings chosen in setup (holes/buttons)
    preserved_holes = int(gs_prev.get('holes', DEFAULT_HOLES))
    preserved_buttons = list(gs_prev.get('score_buttons', DEFAULT_SCORE_BUTTONS))
    preserved_rudeness = int(gs_prev.get('rudeness_level', 0))

    players_raw = request.form.get('players', '')
    players = [n.strip() for n in players_raw.split(',') if n and n.strip()]
    if not players:
        session['previous_players_input'] = players_raw
        logging.warning("Start game failed: No player names entered.")
        flash("Please enter at least one player name.", "warning")
        return redirect(url_for('index'))

    lowered = [p.lower() for p in players]
    if len(set(lowered)) != len(lowered):
        dups = sorted({name for name in players if lowered.count(name.lower()) > 1})
        logging.warning(f"Start game failed: Duplicate player names detected: {dups}")
        session['previous_players_input'] = ','.join([p for p in players if p.lower() not in [d.lower() for d in dups]])
        flash(f"Duplicate player name(s) not allowed: {', '.join(dups)}. Please enter unique names.", "warning")
        return redirect(url_for('index'))

    # Check for names longer than 14 characters
    long_names = [p for p in players if len(p) > 14]
    if long_names:
        session['previous_players_input'] = players_raw
        logging.warning(f"Start game failed: Player name too long: '{long_names[0]}'")
        flash(f"Player names cannot exceed 14 characters. Offending name: '{long_names[0]}'", "warning")
        return redirect(url_for('index'))

    updated_recent = _merge_recent(gs_prev.get('recent_names', []), players)

    _reset_state()
    gs = _get_state()
    gs['players'] = players
    gs['scores'] = {p: 0 for p in players}
    gs['holes'] = max(1, int(preserved_holes))
    gs['score_buttons'] = preserved_buttons[:]
    gs['rudeness_level'] = preserved_rudeness
    gs['round_history'] = [{} for _ in range(gs['holes'])]
    gs['phase'] = 'playing'
    gs['recent_names'] = updated_recent
    _persist(gs)
    logging.info(f"New game started with players: {players}, Holes: {gs['holes']}")
    return redirect(url_for('index'))


@app.route('/score', methods=['POST'])
def record_score():
    """
    Records a score change via a form submission (non-API).
    """
    logging.debug("Route: POST /score - Recording score.")
    gs = _get_state()
    score_change = int(request.form.get('score'))
    _apply_score(gs, score_change)
    _persist(gs)
    return redirect(url_for('index'))


@app.route('/undo', methods=['POST'])
def undo_last_move():
    """
    Undoes the last score move via a form submission (non-API).
    """
    logging.debug("Route: POST /undo - Undoing last move.")
    gs = _get_state()
    _apply_undo(gs)
    _persist(gs)
    return redirect(url_for('index')) # _apply_undo modifies gs, so this is correct.


@app.route('/restart', methods=['POST'])
def restart():
    logging.info("Route: POST /restart - Restarting game, preserving settings.")
    gs = _get_state()
    # Preserve recent names, holes, and score buttons across game restarts
    # This ensures a smoother UX when starting a new game.
    # (7)
    prev_recent = gs.get('recent_names', [])
    prev_holes = gs.get('holes', DEFAULT_HOLES)
    prev_buttons = gs.get('score_buttons', DEFAULT_SCORE_BUTTONS)
    prev_rudeness = gs.get('rudeness_level', 0)
    _reset_state()
    gs = _get_state()
    gs['recent_names'] = prev_recent
    gs['holes'] = prev_holes
    gs['score_buttons'] = prev_buttons # This is a fresh state, so gs is the one we just created.
    gs['rudeness_level'] = prev_rudeness
    _persist(gs)
    return redirect(url_for('index'))


# ---------- JSON APIs ----------
@app.post('/api/score')
def api_score():
    """
    API endpoint to record a score change.
    """
    logging.debug("Route: POST /api/score - API score request.")
    # 1. Get the current state
    gs = _get_state()
    data = request.get_json(force=True, silent=True) or {}
    logging.debug(f"API score data: {data}")
    score_change = int(data.get('score', 0))
    # 2. Modify the state in-place
    _apply_score(gs, score_change)
    # 3. Persist the MODIFIED state
    _persist(gs)
    return jsonify({'ok': True, 'game': gs})


@app.post('/api/undo')
def api_undo():
    """
    API endpoint to undo the last score move.
    """
    logging.debug("Route: POST /api/undo - API undo request.")
    # 1. Get the current state
    gs = _get_state()
    # 2. Modify the state in-place
    _apply_undo(gs)
    # 3. Persist the MODIFIED state
    _persist(gs)
    return jsonify({'ok': True, 'game': gs})


@app.post('/api/end')
def api_end_after_round():
    gs = _get_state()
    """
    API endpoint to toggle the 'end_after_round' flag.
    """
    if gs['phase'] == 'playing':
        new_flag_state = not bool(gs.get('end_after_round', False))
        gs['end_after_round'] = new_flag_state
        logging.info(f"API: Toggled 'end_after_round' to {new_flag_state}")
        _persist(gs)
    return jsonify({'ok': True, 'game': gs})


@app.post('/api/settings')
def api_settings():
    """
    Update game settings (7).
    - holes: only allowed in setup phase (to prevent corruption mid-game)
    - score_buttons: allowed anytime
    """
    logging.debug("Route: POST /api/settings - API settings update request.")
    gs = _get_state()
    data = request.get_json(force=True, silent=True) or {}
    logging.debug(f"API settings data: {data}")
    changed = False

    # Score buttons
    if 'score_buttons' in data:
        try:
            raw = data['score_buttons']
            if isinstance(raw, list):
                btns: list[int] = []
                for v in raw:
                    iv = int(v)
                    if MIN_SCORE_BUTTON_VALUE <= iv <= MAX_SCORE_BUTTON_VALUE:
                        btns.append(iv)
                # Normalize & de-dup, keep order
                seen = set()
                norm = []
                for b in btns:
                    if b not in seen:
                        seen.add(b)
                        norm.append(b)
                if 2 <= len(norm) <= 11:
                    gs['score_buttons'] = norm
                    logging.info(f"Settings updated: score_buttons set to {norm}")
                    changed = True
        except Exception:
            pass

    # Rudeness Level
    if 'rudeness_level' in data:
        try:
            level = int(data['rudeness_level'])
            if 0 <= level < len(RUDENESS_LABELS):
                gs['rudeness_level'] = level
                logging.info(f"Settings updated: rudeness_level set to {level}")
                changed = True
        except (ValueError, TypeError):
            pass

    # Holes (only pre-game)
    if gs['phase'] == 'setup' and 'holes' in data:
        try:
            holes = int(data['holes'])
            if MIN_HOLES <= holes <= MAX_HOLES:
                gs['holes'] = holes
                gs['round_history'] = [{} for _ in range(holes)]
                logging.info(f"Settings updated: holes set to {holes}")
                changed = True
        except Exception:
            pass

    if changed:
        _persist(gs)
    return jsonify({'ok': True, 'game': gs, 'changed': changed})


@app.post('/api/load_saved')
def api_load_saved():
    """
    Replace server state with a previously saved client copy (4).
    Accept only whitelisted keys to avoid injection of arbitrary data.
    """
    logging.info("Route: POST /api/load_saved - Loading state from client.")
    client = request.get_json(force=True, silent=True) or {}
    allowed = set(_fresh_state().keys())
    filtered = {k: v for k, v in client.items() if k in allowed}
    if not isinstance(filtered.get('players', []), list):
        logging.error("API load_saved failed: invalid payload.")
        return jsonify({'ok': False, 'error': 'invalid payload'}), 400
    sid = _get_sid()
    logging.info(f"Successfully loaded and replaced state for SID {sid}.")
    _storage_set(sid, filtered)
    return jsonify({'ok': True})


@app.post('/api/clear_recents')
def api_clear_recents():
    """
    Clears the list of recent player names.
    """
    logging.info("Route: POST /api/clear_recents - Clearing recent names.")
    gs = _get_state()
    gs['recent_names'] = []
    _persist(gs)
    return jsonify({'ok': True, 'game': gs})


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
    """
    Applies a score change to the current game state, handling player turns,
    round progression, and initiating playoffs if conditions are met.
    """
    logging.debug(f"[_apply_score] Phase: {gs['phase']}. Current player index: {gs.get('current_player_index')}")
    holes = int(gs.get('holes', 20))
    if gs['phase'] == 'playing':
        player = gs['players'][gs['current_player_index']]
        gs['scores'][player] += score_change
        gs['round_history'][gs['current_round'] - 1][player] = score_change
        gs['undo_history'].append({'player_index': gs['current_player_index'], 'score_change': score_change})

        last_index = len(gs['players']) - 1
        was_last_in_round = (gs['current_player_index'] == last_index)

        if was_last_in_round and gs.get('end_after_round'):
            logging.info("End of round and 'end_after_round' is set. Initiating playoffs.")
            initiate_playoffs(gs)
            return

        gs['current_player_index'] += 1
        if gs['current_player_index'] >= len(gs['players']):
            gs['current_player_index'] = 0
            gs['current_round'] += 1

        if gs['current_round'] > holes:
            logging.info(f"Final round ({gs['current_round']-1}) complete. Initiating playoffs.")
            initiate_playoffs(gs)

    elif gs['phase'] == 'playoff':
        # In playoff mode, current_player_index iterates over the ACTIVE tie subgroup only
        player = gs['playoff_group'][gs['current_player_index']]
        logging.debug(f"[_apply_score] Playoff score for {player}: {score_change}")
        gs['playoff_round_scores'][player] = score_change
        logging.debug(f"[_apply_score] After adding score for {player}: {gs.get('playoff_round_scores')}")
        gs['current_player_index'] += 1
        if gs['current_player_index'] >= len(gs['playoff_group']):
            logging.debug(f"[_apply_score] Playoff round complete, resolving. Scores: {gs.get('playoff_round_scores')}")
            resolve_playoff_round(gs)


def _apply_undo(gs: dict):
    """
    Undoes the last score move, reverting game state to the previous turn.
    Only applicable during the 'playing' phase.
    """
    # Undo applies only during main play (not in playoff phase)
    if gs['phase'] != 'playing' or not gs['undo_history']:
        logging.warning(f"[_apply_undo] Undo skipped. Phase: {gs['phase']}, History empty: {not gs['undo_history']}")
        return

    last_move = gs['undo_history'].pop()
    prev_idx = (gs['current_player_index'] - 1 + len(gs['players'])) % len(gs['players'])
    logging.info(f"[_apply_undo] Reverting move for player index {prev_idx}. Score change: {last_move['score_change']}")
    gs['current_player_index'] = prev_idx

    if prev_idx == len(gs['players']) - 1:
        gs['current_round'] -= 1
        if gs['current_round'] < 1:
            gs['current_round'] = 1

    player_to_undo = gs['players'][prev_idx]
    gs['scores'][player_to_undo] -= last_move['score_change']
    gs['round_history'][gs['current_round'] - 1].pop(player_to_undo, None)


def initiate_playoffs(gs: dict):
    """
    Initiates the playoff phase by identifying tied players and setting up
    the initial playoff groups.
    """
    logging.info("Initiating playoffs: identifying tied players.")
    gs['end_after_round'] = False

    # Group players by base total score; only ties (len > 1) need playoffs
    scores_to_players = defaultdict(list)
    for p, s in gs['scores'].items():
        scores_to_players[int(s)].append(p)

    gs['pending_playoffs'] = []
    for base_score, players in scores_to_players.items():
        if len(players) > 1:
            logging.info(f"Tie detected at score {base_score} for players: {players}")
            gs['pending_playoffs'].append({'score': base_score, 'players': players})

    # Resolve from worst totals to best totals (higher total is worse)
    logging.debug(f"Pending playoffs sorted by score (desc): {gs['pending_playoffs']}")
    gs['pending_playoffs'].sort(key=lambda p: p['score'], reverse=True)
    start_next_playoff(gs)


def start_next_playoff(gs: dict):
    """
    Starts the next playoff round or transitions to final ranking if no
    more ties are pending.
    """
    if gs['pending_playoffs']:
        nxt = gs['pending_playoffs'].pop(0)
        logging.info(f"Starting next playoff for score {nxt['score']} with players {nxt['players']}")
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
        logging.info("All playoffs resolved. Transitioning to final ranking.")
        gs['phase'] = 'final_ranking'
        gs['rounds_played'] = _compute_rounds_played(gs)
        ordered = _final_order_players(gs)
        gs['winner'] = ordered[0] if ordered else None


def _tb_sequence_for_player_in_current_tie(gs: dict, player: str) -> list[int]:
    """
    Retrieves the tie-breaker score sequence for a given player within the
    current playoff context.
    """
    seq: list[int] = []
    for rnd in gs['playoff_history']:
        if player in rnd:
            seq.append(int(rnd[player]))
    return seq


def _finalize_player_from_current_tie(gs: dict, player: str):
    """
    Finalizes a player's standing in the current tie-breaker, removing them from the active playoff pool."""
    seq = _tb_sequence_for_player_in_current_tie(gs, player)
    if seq:
        gs['final_playoff_scores'][player] = seq[-1]
    else:
        gs['final_playoff_scores'][player] = 0
    if player in gs['playoff_pool']:
        gs['playoff_pool'].remove(player)


def resolve_playoff_round(gs: dict):
    """
    Resolves the current playoff round, updates player standings, and
    determines the next playoff group or transitions to final ranking.
    """
    logging.debug(f"[resolve_playoff_round] Starting resolution. Current playoff_round_scores: {gs.get('playoff_round_scores')}")
    scores = gs['playoff_round_scores']
    gs['playoff_history'].append(scores.copy())

    for p, tb in scores.items():
        gs['all_playoff_history'].setdefault(p, []).append(int(tb))

    gs['playoff_round_scores'] = {}
    logging.debug(f"[resolve_playoff_round] playoff_round_scores reset. New state: {gs.get('playoff_round_scores')}")
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
        logging.debug(f"Worst subgroup in current pool {gs['playoff_pool']}: {worst_players}")
        if not worst_players:
            start_next_playoff(gs)
            return

        if len(worst_players) == 1:
            loser = worst_players[0]
            _finalize_player_from_current_tie(gs, loser)
            logging.info(f"Player '{loser}' has been finalized and removed from the playoff pool.")
            if not gs['playoff_pool']:
                start_next_playoff(gs)
                return
            continue
        else:
            gs['playoff_group'] = worst_players
            logging.info(f"Tie continues between {worst_players}. Starting new playoff round {gs['playoff_round'] + 1}.")
            gs['playoff_round'] += 1
            return


# ---------------- Stats view (unchanged) ----------------
@app.route('/stats')
def stats():
    """
    Renders the statistics page, calculating various player metrics.
    """
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

    def _best_comebacks_by_player(cumulative_per: dict[str, list[int]]) -> dict:
        out = {}
        for p, seq in cumulative_per.items():
            if len(seq) < 2: continue
            s = np.array(seq)
            cummin_from_end = np.minimum.accumulate(s[::-1])[::-1]
            improvements = s[:-1] - cummin_from_end[1:]
            if np.any(improvements > 0):
                max_improvement = np.max(improvements)
                if max_improvement > 0:
                    from_idx = np.argmax(improvements)
                    # Find the index of the minimum score in the rest of the array
                    to_idx = from_idx + 1 + np.argmin(s[from_idx+1:])
                    out[p] = {
                        'player': p,
                        'improvement': int(max_improvement),
                        'from_score': seq[from_idx],
                        'to_score': seq[to_idx],
                        'from_round': from_idx + 1,
                        'to_round': to_idx + 1,
                    }
        return out

    def _biggest_falls_by_player(cumulative_per: dict[str, list[int]]) -> dict:
        out = {}
        for p, seq in cumulative_per.items():
            if len(seq) < 2: continue
            s = np.array(seq)
            # Find the running maximum from the end of the sequence backwards
            cummax_from_end = np.maximum.accumulate(s[::-1])[::-1]
            # Calculate potential worsenings: score at a later point minus score now
            worsenings = cummax_from_end[1:] - s[:-1]
            if np.any(worsenings > 0):
                max_worsening = np.max(worsenings)
                if max_worsening > 0:
                    from_idx = np.argmax(worsenings)
                    # Find the index of the maximum score in the rest of the array
                    to_idx = from_idx + 1 + np.argmax(s[from_idx+1:])
                    out[p] = {
                        'player': p,
                        'worsening': int(max_worsening),
                        'from_score': seq[from_idx],
                        'to_score': seq[to_idx],
                        'from_round': from_idx + 1,
                        'to_round': to_idx + 1,
                    }
        return out

    per = _extract_per_player_sequences(gs)

    birdie_streaks = {}
    bogey_streaks = {}
    birdie_counts = {}
    bogey_counts = {}
    # Calculate cumulative scores for comeback/fall stats
    cumulative_per = {p: np.cumsum(seq).tolist() for p, seq in per.items()}
    on_target_percentages = {}

    avgs = {}
    rounds_counts = {}

    for p, seq in per.items():
        birdie_streaks[p] = _longest_streak(seq, lambda x: x < 0)
        bogey_streaks[p] = _longest_streak(seq, lambda x: x > 0)
        birdie_counts[p] = sum(1 for v in seq if v < 0)
        bogey_counts[p] = sum(1 for v in seq if v > 0)
        if seq:
            avg = statistics.mean(seq)
            # Calculate on target percentage
            on_target_rounds = sum(1 for v in seq if v <= 0)
            on_target_percentages[p] = (on_target_rounds / len(seq)) * 100 if len(seq) > 0 else 0
            avgs[p] = statistics.mean(seq)
            rounds_counts[p] = len(seq)
        else:
            on_target_percentages[p] = 0
            avgs[p] = 0.0
            rounds_counts[p] = 0
    players_with_data = [p for p in gs.get('players', []) if rounds_counts.get(p, 0) > 0] # type: ignore

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

    on_target_order = sorted(
        players_with_data,
        key=lambda p: (-on_target_percentages.get(p, 0), avgs.get(p, 0)) # Higher percentage is better
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
    on_target_ranking = [
        {'name': p, 'percentage': on_target_percentages.get(p, 0)} for p in on_target_order
    ]
    best_comebacks = _best_comebacks_by_player(cumulative_per)
    comeback_ranking = sorted(best_comebacks.values(), key=lambda d: d['improvement'], reverse=True)

    biggest_falls = _biggest_falls_by_player(cumulative_per)
    fall_ranking = sorted(biggest_falls.values(), key=lambda d: d['worsening'], reverse=True)

    # Calculate max values for bar scaling in the template
    max_birdie_streak = max(birdie_streaks.values(), default=0)
    max_bogey_streak = max(bogey_streaks.values(), default=0)
    max_on_target_percentage = max(on_target_percentages.values(), default=0)
    max_birdie_count = max(birdie_counts.values(), default=0)
    max_bogey_count = max(bogey_counts.values(), default=0)
    max_average_abs = max((abs(a) for a in avgs.values()), default=0)
    max_comeback_improvement = max((cb['improvement'] for cb in best_comebacks.values()), default=0)
    max_fall_worsening = max((f['worsening'] for f in biggest_falls.values()), default=0)
    return render_template(
        'index.html',
        show_stats=True,
        game=gs,
        birdie_streak_ranking=birdie_streak_ranking,
        bogey_streak_ranking=bogey_streak_ranking,
        average_ranking=average_ranking,
        most_birdies_ranking=most_birdies_ranking,
        most_bogeys_ranking=most_bogeys_ranking,
        on_target_ranking=on_target_ranking,
        comeback_ranking=comeback_ranking,
        fall_ranking=fall_ranking,
        # Max values for mini-graph scaling
        max_birdie_streak=max_birdie_streak,
        max_bogey_streak=max_bogey_streak,
        max_birdie_count=max_birdie_count,
        max_on_target_percentage=max_on_target_percentage,
        max_bogey_count=max_bogey_count,
        max_average_abs=max_average_abs,
        max_comeback_improvement=max_comeback_improvement,
        max_fall_worsening=max_fall_worsening,
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # For production, debug mode should be off.
    # It can be enabled for local development by setting the DEBUG environment variable.
    is_debug = os.environ.get("DEBUG", "0").lower() in ("1", "true", "yes")
    app.logger.info(f"Starting Flask server on 0.0.0.0:{port} (Debug: {is_debug})")
    app.run(debug=is_debug, host='0.0.0.0', port=port)
