import os
import json
import time
from pathlib import Path
import random
from collections import defaultdict
import statistics
import numpy as np
import uuid
from fuzzywuzzy import process

from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify,
    session, send_from_directory, abort, g
)
from werkzeug.middleware.proxy_fix import ProxyFix
import logging

# Get app start time for cache-busting static assets
app_start_time = int(time.time())

# Configure logging to show DEBUG messages from all loggers
logging.basicConfig(level=logging.DEBUG)

# ---------------- App & Folders ----------------
BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Tell Flask it's behind a proxy (e.g., Cloudflare) and to trust X-Forwarded-Proto
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Make the app_start_time available to all templates
@app.context_processor
def inject_app_start_time():
    return dict(app_start_time=app_start_time)

# Security / cookie hardening (10)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-to-a-secure-random-string-in-production")
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(int(os.environ.get("SESSION_COOKIE_SECURE", "0"))),  # set 1 in prod
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30,  # 30 days
    # Explicitly set the server name for cookie domain security
    SERVER_NAME=os.environ.get("SERVER_NAME", None),
)

app.before_request(
    lambda: setattr(g, 'url_scheme', 'https' if app.config['SESSION_COOKIE_SECURE'] else 'http')
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
_player_stats_fallback: dict[str, dict] = {} # in-memory fallback for player stats


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

# Word list for generating human-readable SIDs
EASY_WORDS = [
    # Original 146 Words
    "apple", "baker", "candy", "delta", "eagle", "fancy", "giant", "happy",
    "igloo", "joker", "kilo", "lemon", "magic", "noble", "ocean", "piano",
    "queen", "robot", "salsa", "tiger", "ultra", "viper", "wagon", "xenon",
    "yacht", "zebra", "arrow", "beach", "cloud", "dream", "ember", "frost",
    "globe", "honey", "ivory", "juice", "koala", "lunar", "mango", "noodle",
    "olive", "pearl", "quest", "river", "sunny", "tulip", "unity", "velvet",
    "watch", "x-ray", "yield", "zesty", "amber", "bravo", "coral", "dusk",
    "echo", "flame", "grape", "haven", "index", "jade", "karma", "laser",
    "melon", "nexus", "orbit", "prism", "quark", "relic", "shadow", "torch",
    "umbra", "vortex", "windy", "year", "zenith", "aqua", "bliss", "comet",
    "dew", "elf", "fable", "gem", "halo", "iris", "jolt", "kite", "lava",
    "mist", "neon", "opal", "pulse", "rain", "spark", "tide", "union",
    "vivid", "wave", "yarn", "zone", "ace", "blue", "cat", "dog", "egg",
    "fox", "gem", "hat", "ink", "jet", "key", "log", "map", "net", "owl",
    "pin", "quiz", "rat", "sun", "top", "urn", "van", "web", "zip",

    # Added 146 Unique Words (Total: 292)
    "abyss", "blaze", "chime", "dance", "elate", "ferry", "groove", "hatch",
    "ideal", "jewel", "knoll", "light", "morph", "notch", "oaken", "pixel",
    "quake", "ridge", "slate", "thorn", "unite", "vault", "whirl", "yodel",
    "zest", "alert", "braid", "crush", "diner", "enact", "flake", "gloom",
    "heave", "iron", "joust", "kiosk", "loft", "marsh", "novel", "outer",
    "panel", "quill", "roast", "snare", "train", "usher", "value", "vibe",
    "widow", "yawn", "zodiac", "angle", "blimp", "crave", "donor", "essay",
    "foyer", "grill", "heron", "inlet", "jelly", "knack", "lodge", "merit",
    "nomad", "offend", "patio", "quarry", "rally", "sniff", "tramp", "utter",
    "venue", "vowel", "wager", "wrist", "zillion", "apron", "bacon", "chase",
    "drive", "epoch", "fiber", "gorge", "hasty", "insert", "jungle", "kettle",
    "lasso", "minor", "neigh", "opera", "pouch", "quiver", "radio", "scoop",
    "tinsel", "uncle", "vixen", "wisp", "xylitol", "yummy", "zircon", "almond",
    "bronze", "carpet", "daisy", "eject", "forage", "glider", "heir", "infant",
    "jockey", "kindle", "liner", "muffin", "ninth", "organ", "parade", "quinoa",
    "reign", "sprint", "tundra", "upward", "vacuum", "wallet", "yearly",
    "zipper", "agile", "breeze", "canyon", "deluxe", "endorse", "ferret",
    "giggle", "humor", "imprint", "jigsaw", "krypton", "lattice", "mercury",
    "nostril", "oyster", "pelican", "quantum", "resume", "sapphire", "tether",
    "upgrade", "venom", "wander", "youth", "zither"
]


def _generate_readable_sid(k: int = 3, sep: str = '-') -> str:
    """
    Generates a human-readable session ID from a list of simple words.
    Example: 'happy-tiger-beach'

    Args:
        k: The number of words to include in the ID.
        sep: The separator to use between words.

    Returns:
        A human-readable string to be used as a session ID.
    """
    if not EASY_WORDS:
        # Fallback to a simple random hex string if word list is empty
        return ''.join(random.choices('0123456789abcdef', k=12))
    
    return sep.join(random.choices(EASY_WORDS, k=k))


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
        logging.debug(f"[_storage_get] (Redis) Loaded state for SID {sid}. Phase: {gs.get('phase') if gs else 'N/A'}, Round: {gs.get('current_round') if gs else 'N/A'}")
        return gs
    else:
        gs = _games.get(sid)
        logging.debug(f"[_storage_get] (In-memory) Loaded state for SID {sid}. Phase: {gs.get('phase') if gs else 'N/A'}, Round: {gs.get('current_round') if gs else 'N/A'}")
        return gs


def _storage_set(sid: str, gs: dict) -> None:
    """
    Stores game state for a given session ID.
    
    Args:
        sid: The session ID.
        gs: The game state dictionary to store.
    """
    if _redis:
        logging.debug(f"[_storage_set] (Redis) Saving state for SID {sid}. Phase: {gs.get('phase')}, Round: {gs.get('current_round')}")
        _redis.set(_storage_key(sid), json.dumps(gs))
    else:
        logging.debug(f"[_storage_set] (In-memory) Saving state for SID {sid}. Phase: {gs.get('phase')}, Round: {gs.get('current_round')}")
        _games[sid] = gs


# -------------- Game accessors --------------
def _get_sid() -> str:
    """Retrieves or generates a unique session ID for the current user."""
    sid = session.get("sid")
    # If there's no SID in the session cookie, create one.
    # This should only happen once for a new visitor.
    if not sid: 
        sid = _generate_readable_sid()
        logging.info(f"New session created with SID: {sid}")
        session["sid"] = sid
        session.permanent = True
    return sid # Return the existing or newly created SID.


def _get_state() -> dict:
    """Retrieves the current game state, initializing it if not found."""
    sid = _get_sid()
    gs = _storage_get(sid)
    if not gs:
        # If no game state is found in storage for this session,
        # create a fresh one and save it immediately.
        gs = _fresh_state()
        logging.info(f"No state found for SID {sid}. Creating fresh state.")
        _storage_set(sid, gs)
    elif not isinstance(gs.get('playoff_round_scores'), dict):
        gs['playoff_round_scores'] = {}
        _storage_set(sid, gs) # Persist this correction

    # Always inject the current SID for display purposes
    gs['sid'] = sid
    return gs


def _reset_state():
    """Resets the game state for the current session to a fresh state."""
    sid = _get_sid()
    _storage_set(sid, _fresh_state()) # Overwrite with a fresh state, don't delete the key


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


def _update_recent_names_after_rename(old_name: str, new_name: str):
    """
    Updates a player's name in the 'recent_names' list for all game sessions.
    This is a more complex operation as it requires iterating through all games.
    """
    if not _redis:
        # For in-memory, we can only affect the current session's recent names
        gs = _get_state()
        if old_name in gs.get('recent_names', []):
            gs['recent_names'] = [new_name if n.lower() == old_name.lower() else n for n in gs['recent_names']]
            _persist(gs)
        return
    
    # Find all game sessions that contain the old name in their recent_names list
    game_keys = _redis.keys("game:*")
    for key in game_keys:
        gs_raw = _redis.get(key)
        if gs_raw:
            gs = json.loads(gs_raw)
            if old_name in gs.get('recent_names', []):
                gs['recent_names'] = [new_name if n.lower() == old_name.lower() else n for n in gs['recent_names']]
                _redis.set(key, json.dumps(gs))

    # For Redis, we can do a more global update if needed, but for now,
    # let's stick to the current session to avoid complexity.
    # A global update would require iterating all game:* keys.
    # The current implementation will handle it for the active session.
    pass

def _final_order_players(gs: dict) -> list[str]:
    # Lower base score is better; for ties, compare TB sequence lexicographically (lower is better).
    def key(player: str):
        # The key function now receives a player ID
        base = int(gs['scores'].get(player, 0))
        tb_seq = gs['all_playoff_history'].get(player, [])
        return (base, *tb_seq)
    return sorted(gs['players'], key=key)


def _inject_template_data(gs: dict) -> None:
    """Injects computed values needed for rendering into the game state."""
    # Inject the correct score labels based on the current rudeness level
    level = gs.get('rudeness_level', 0)
    gs['score_labels'] = RUDENESS_LABELS[level]


def _update_persistent_player_stats(gs: dict):
    """
    At the end of a game, update the persistent, long-term stats for each player in Redis.
    """
    stat_deltas = {} # To store the changes for each player

    standings = gs.get('final_standings', [])
    if not standings:
        return stat_deltas

    logging.info(f"Updating persistent stats for {len(standings)} players.")
    player_map = gs.get('player_map', {})
    for standing in standings:
        player_name = standing['name']
        player_id = next((pid for pid, pdata in player_map.items() if pdata['name'] == player_name), None)
        if not player_id: continue

        birdies = sum(1 for r in gs['round_history'] if r.get(player_id, 1) < 0)
        bogeys = sum(1 for r in gs['round_history'] if r.get(player_id, -1) > 0)
        # New stats for overall average and on-target percentage
        game_score = standing.get('score', 0)
        game_rounds_played = sum(1 for r in gs['round_history'] if player_id in r)
        game_on_target_rounds = sum(1 for r in gs['round_history'] if r.get(player_id, 1) <= 0)

        # Store deltas for animation
        stat_deltas[player_name] = {
            'games_played': 1,
            'wins': 1 if standing['rank'] == 1 else 0,
            'total_birdies': birdies,
            'total_bogeys': bogeys,
            'total_score_all_games': game_score,
            'total_rounds_all_games': game_rounds_played,
            'total_on_target_rounds_all_games': game_on_target_rounds,
        }

        if _redis:
            player_key = f"player_stats:{player_id}"
            # Use a pipeline for atomic updates
            pipe = _redis.pipeline()
            pipe.hincrby(player_key, "games_played", 1)
            if standing['rank'] == 1:
                pipe.hincrby(player_key, "wins", 1)
            pipe.hincrby(player_key, "total_birdies", birdies)
            pipe.hincrby(player_key, "total_bogeys", bogeys)
            pipe.hincrby(player_key, "total_score_all_games", game_score)
            pipe.hincrby(player_key, "total_rounds_all_games", game_rounds_played)
            pipe.hincrby(player_key, "total_on_target_rounds_all_games", game_on_target_rounds)
            # Store the deltas temporarily for frontend animation
            delta_key = f"player_stats_delta:{player_id}"
            pipe.set(delta_key, json.dumps(stat_deltas[player_name]), ex=300) # Expire after 5 minutes

            pipe.execute()
        else:
            # Fallback to in-memory dictionary for local development
            if player_id not in _player_stats_fallback:
                _player_stats_fallback[player_id] = {
                    "games_played": 0,
                    "wins": 0,
                    "total_birdies": 0,
                    "total_bogeys": 0,
                    "total_score_all_games": 0,
                    "total_rounds_all_games": 0,
                    "total_on_target_rounds_all_games": 0,
                    # In-memory fallback for deltas
                    "last_game_deltas": {},
                }
            
            stats = _player_stats_fallback[player_id]
            stats["games_played"] += 1
            if standing['rank'] == 1:
                stats["wins"] += 1
            stats["total_birdies"] += birdies
            stats["total_bogeys"] += bogeys
            stats["total_score_all_games"] += game_score
            stats["total_rounds_all_games"] += game_rounds_played
            stats["total_on_target_rounds_all_games"] += game_on_target_rounds

            # Update best/worst game scores
            if 'best_game_score' not in stats:
                stats['best_game_score'] = game_score
            else:
                stats['best_game_score'] = min(stats['best_game_score'], game_score)

            if 'worst_game_score' not in stats:
                stats['worst_game_score'] = game_score
            else:
                stats['worst_game_score'] = max(stats['worst_game_score'], game_score)

            # Calculate deltas for best/worst scores
            # Note: For min/max, a delta is only meaningful if a new record is set.
            # We'll calculate the change from the *previous* record.
            old_best = stats.get('best_game_score_before_this_game', game_score)
            old_worst = stats.get('worst_game_score_before_this_game', game_score)
            if game_score < old_best:
                stat_deltas[player_name]['best_game_score_delta'] = game_score - old_best
            if game_score > old_worst:
                stat_deltas[player_name]['worst_game_score_delta'] = game_score - old_worst

            stats["last_game_deltas"] = stat_deltas[player_name]
    return stat_deltas

def _get_all_player_stats():
    """
    Fetches all player stats from Redis or the in-memory fallback.
    Returns a list of player stat dictionaries.
    """
    all_stats = []
    if _redis:
        # Use SCAN to iterate over player stat keys without blocking the server
        for key in _redis.scan_iter("player_stats:*"):
            player_id = key.split(":", 1)[1]
            stats_raw = _redis.hgetall(key)
            if stats_raw:
                stats = {k: int(v) for k, v in stats_raw.items()}
                stats['id'] = player_id
                all_stats.append(stats)
    else:
        # Use the in-memory fallback
        for player_id, stats_raw in _player_stats_fallback.items():
            stats = stats_raw.copy()
            stats['id'] = player_id
            all_stats.append(stats)
    return all_stats
# --------------------- Routes ---------------------
@app.route('/')
def index():
    logging.debug(f"Route: GET / - Rendering main page.")
    gs = _get_state()
    
    # Retrieve and clear any player list from a failed form submission
    previous_players = session.pop('previous_players_input', None)

    _inject_template_data(gs)
    # Build final standings if we just entered final_ranking
    if gs['phase'] == 'final_ranking' and not gs['final_standings']:
        logging.info("Phase is 'final_ranking'. Computing and persisting final standings.")
        gs['rounds_played'] = _compute_rounds_played(gs)
        ordered_ids = _final_order_players(gs)
        standings = []
        last_key = None
        current_rank = 0
        for pid in ordered_ids:
            base_total = int(gs['scores'].get(pid, 0))
            tb_seq = tuple(gs['all_playoff_history'].get(pid, []))
            this_key = (base_total, tb_seq)
            if this_key != last_key:
                current_rank = len(standings) + 1
            player_name = gs.get('player_map', {}).get(pid, {}).get('name', 'Unknown')
            standings.append({'rank': current_rank, 'name': player_name, 'id': pid, 'score': base_total})
            last_key = this_key

        gs['final_standings'] = standings
        gs['max_playoff_rounds'] = max((len(h) for h in gs['all_playoff_history'].values()), default=0)
        gs['stat_deltas'] = _update_persistent_player_stats(gs) # Update persistent stats at game end
        _persist(gs)

    return render_template('index.html', game=gs, show_stats=False, previous_players=previous_players)


# NEW: tolerate GET /start (prefetches / SW / crawlers) by redirecting home
@app.route('/start', methods=['GET'])
def start_game_get():
    logging.debug("Route: GET /start - Redirecting to index.")
    return redirect(url_for('index', _scheme=g.get('url_scheme', 'http')))


@app.route('/start', methods=['POST'])
def start_game():
    logging.info("Route: POST /start - Starting new game.")
    gs_prev = _get_state()
    # allow carrying settings chosen in setup (holes/buttons)
    preserved_holes = int(gs_prev.get('holes', DEFAULT_HOLES))
    preserved_buttons = list(gs_prev.get('score_buttons', DEFAULT_SCORE_BUTTONS))
    preserved_rudeness = int(gs_prev.get('rudeness_level', 0))

    player_ids_raw = request.form.get('players', '')
    player_ids = [pid.strip() for pid in player_ids_raw.split(',') if pid and pid.strip()]
    if not player_ids:
        session['previous_players_input'] = player_ids_raw
        logging.warning("Start game failed: No player names entered.")
        flash("Please enter at least one player name.", "warning")
        return redirect(url_for('index', _scheme=g.get('url_scheme', 'http')))

    # Look up player objects from the database using the received IDs
    player_db = get_player_db()
    player_objects = [player_db.get(pid) for pid in player_ids if pid in player_db]
    
    # Create a map of ID -> {id, name} for this game
    player_map = {p['id']: p for p in player_objects}

    # Update recent names list
    updated_recent = _merge_recent(gs_prev.get('recent_names', []), [p['name'] for p in player_objects])

    gs = _fresh_state() # Start with a fresh dictionary
    gs['players'] = player_ids # Store IDs
    gs['player_map'] = player_map # Store the lookup map
    gs['player_names'] = [p['name'] for p in player_objects] # Keep names for display order

    gs['scores'] = {pid: 0 for pid in player_ids}
    gs['holes'] = max(1, int(preserved_holes))
    gs['score_buttons'] = preserved_buttons[:]
    gs['rudeness_level'] = preserved_rudeness
    gs['round_history'] = [{} for _ in range(gs['holes'])]
    gs['phase'] = 'playing'
    gs['recent_names'] = updated_recent
    _storage_set(_get_sid(), gs) # Overwrite the state for the current session with the new game
    logging.info(f"New game started with players: {[p['name'] for p in player_objects]}, Holes: {gs['holes']}")
    return redirect(url_for('index', _scheme=g.get('url_scheme', 'http')))


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
    return redirect(url_for('index', _scheme=g.get('url_scheme', 'http')))


@app.route('/undo', methods=['POST'])
def undo_last_move():
    """
    Undoes the last score move via a form submission (non-API).
    """
    logging.debug("Route: POST /undo - Undoing last move.")
    gs = _get_state()
    _apply_undo(gs)
    _persist(gs)
    return redirect(url_for('index', _scheme=g.get('url_scheme', 'http')))


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
    _inject_template_data(gs)
    _persist(gs)
    return redirect(url_for('index', _scheme=g.get('url_scheme', 'http')))


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
    _inject_template_data(gs)
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
    _inject_template_data(gs)
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
    _inject_template_data(gs)
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
    _inject_template_data(gs)
    return jsonify({'ok': True, 'game': gs, 'changed': changed, 'app_start_time': app_start_time})


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
    _inject_template_data(gs)
    return jsonify({'ok': True, 'game': gs})

def get_player_db():
    """
    Gets the player database from the session.
    For local testing, this simulates a simple DB.
    In production, you would replace this with your Redis calls.
    """
    if 'players' not in session:
        # Initialize with a few example players if desired
        session['players'] = {} # Stored as {player_id: {'id': ..., 'name': ...}}
    return session['players']

@app.route('/api/resolve-player/<string:name>')
def resolve_player(name):
    """
    Finds players with exact or similar names.
    This is used for the "Did you mean...?" feature.
    """
    player_db = get_player_db()
    all_players = list(player_db.values())
    all_player_names = [p['name'] for p in all_players]
    
    # 1. Find an exact (case-insensitive) match
    exact_match_obj = next((p for p in all_players if p['name'].lower() == name.lower()), None)

    # 2. Find fuzzy suggestions, excluding the exact match if found
    suggestions = []
    # Only search for suggestions if there isn't an exact match, to avoid ambiguity.
    if not exact_match_obj:
        similar_matches = process.extract(name, all_player_names, limit=5)
        for found_name, score in similar_matches:
            # We use a higher threshold here to only suggest likely typos
            if score > 80:
                player = next((p for p in all_players if p['name'] == found_name), None)
                if player:
                    suggestions.append(player)

    return jsonify(ok=True, exact_match=exact_match_obj, suggestions=suggestions)

@app.route('/api/create-player', methods=['POST'])
def create_player():
    """
    Creates a new player and adds them to the database.
    """
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify(ok=False, error="Player name is required."), 400

    name = data['name'].strip()
    if not name or len(name) > 14:
        return jsonify(ok=False, error="Invalid player name."), 400

    player_db = get_player_db()

    new_player = {
        'id': str(uuid.uuid4()),
        'name': name
    }
    player_db[new_player['id']] = new_player
    session.modified = True # Make sure the session is saved
    
    # Also add the new player to the list of recent names to ensure they can be added to a game
    gs = _get_state()
    gs['recent_names'] = _merge_recent(gs.get('recent_names', []), [name])
    _persist(gs)

    return jsonify(ok=True, player=new_player)

@app.route('/api/players')
def get_all_players():
    """
    Returns a list of all players in the database.
    """
    player_db = get_player_db()
    all_players = sorted(list(player_db.values()), key=lambda p: p['name'].lower())
    return jsonify(ok=True, players=all_players)

@app.route('/leaderboard')
def leaderboard():
    """
    Renders a global leaderboard page, ranking all players by games played.
    """
    gs = _get_state() # Get session state for context
    player_db = get_player_db()
    all_player_stats = _get_all_player_stats()

    # Enrich stats with current names from the player_db
    for stats in all_player_stats:
        stats['name'] = player_db.get(stats['id'], {}).get('name', 'Unknown Player')

    # Sort players by games_played (desc), then by name (asc) as a tie-breaker
    leaderboard_data = sorted(
        all_player_stats,
        key=lambda p: (-p.get('games_played', 0), p.get('name', '').lower())
    )

    return render_template('index.html', game=gs, show_leaderboard=True, leaderboard=leaderboard_data)

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


@app.route('/logout')
def logout():
    """
    Clears the user's session cookie and redirects to the homepage.
    This effectively logs them out and starts a new session.
    """
    session.clear()
    return redirect(url_for('index', _scheme=g.get('url_scheme', 'http')))


# ----------------- Game logic -----------------
def _apply_score(gs: dict, score_change: int):
    """
    Applies a score change to the current game state, handling player turns,
    round progression, and initiating playoffs if conditions are met.
    """
    holes = int(gs.get('holes', 20))
    if gs['phase'] == 'playing':
        player_id = gs['players'][gs['current_player_index']]
        player_name = gs['player_map'].get(player_id, {}).get('name', 'Unknown')
        logging.debug(f"[_apply_score] Applying score of {score_change} for player '{player_name}' (ID: {player_id}) in round {gs['current_round']}.")

        gs['scores'][player_id] += score_change
        gs['round_history'][gs['current_round'] - 1][player_id] = score_change
        gs['undo_history'].append({'player_index': gs['current_player_index'], 'score_change': score_change})

        last_index = len(gs['players']) - 1
        was_last_in_round = (gs['current_player_index'] == last_index)

        # Check if the game should end
        is_game_over = was_last_in_round and (gs['current_round'] >= holes or gs.get('end_after_round'))

        if is_game_over:
            logging.info(f"Final round ({gs['current_round']}) complete. Initiating playoffs.")
            initiate_playoffs(gs)
            # If playoffs immediately resolve to final_ranking, include stat deltas
            if gs.get('phase') == 'final_ranking':
                gs['stat_deltas'] = _update_persistent_player_stats(gs)
        else:
            # Advance to the next player/round
            gs['current_player_index'] = (gs['current_player_index'] + 1) % len(gs['players'])
            if was_last_in_round:
                gs['current_round'] += 1

    elif gs['phase'] == 'playoff':
        # In playoff mode, current_player_index iterates over the ACTIVE tie subgroup only
        player_id = gs['playoff_group'][gs['current_player_index']]
        player_name = gs['player_map'].get(player_id, {}).get('name', 'Unknown')
        logging.debug(f"[_apply_score] Playoff score for {player_name}: {score_change}")
        gs['playoff_round_scores'][player_id] = score_change
        logging.debug(f"[_apply_score] After adding score for {player_name}: {gs.get('playoff_round_scores')}")
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
    prev_idx = last_move['player_index']
    player_id_to_undo = gs['players'][prev_idx]
    player_name_to_undo = gs['player_map'].get(player_id_to_undo, {}).get('name', 'Unknown')
    logging.info(f"[_apply_undo] Reverting score of {last_move['score_change']} for player '{player_name_to_undo}'.")
    gs['current_player_index'] = prev_idx

    if prev_idx == len(gs['players']) - 1:
        gs['current_round'] -= 1
        if gs['current_round'] < 1:
            gs['current_round'] = 1
    
    gs['scores'][player_id_to_undo] -= last_move['score_change']
    gs['round_history'][gs['current_round'] - 1].pop(player_id_to_undo, None)


def initiate_playoffs(gs: dict):
    """
    Initiates the playoff phase by identifying tied players and setting up
    the initial playoff groups.
    """
    logging.info("Initiating playoffs: identifying tied players.")
    gs['end_after_round'] = False

    # Group players by base total score; only ties (len > 1) need playoffs
    scores_to_players = defaultdict(list)
    for pid, s in gs['scores'].items():
        scores_to_players[int(s)].append(pid)

    gs['pending_playoffs'] = []
    for base_score, player_ids in scores_to_players.items():
        if len(player_ids) > 1:
            logging.info(f"Tie detected at score {base_score} for players: {player_ids}")
            gs['pending_playoffs'].append({'score': base_score, 'players': player_ids})

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
        logging.info(f"Starting next playoff for score {nxt['score']} with player IDs {nxt['players']}")
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
        gs['stat_deltas'] = _update_persistent_player_stats(gs)


def _tb_sequence_for_player_in_current_tie(gs: dict, player_id: str) -> list[int]:
    """
    Retrieves the tie-breaker score sequence for a given player within the
    current playoff context.
    """
    seq: list[int] = []
    for rnd in gs['playoff_history']:
        if player_id in rnd:
            seq.append(int(rnd[player_id]))
    return seq


def _finalize_player_from_current_tie(gs: dict, player_id: str):
    """
    Finalizes a player's standing in the current tie-breaker, removing them from the active playoff pool."""
    seq = _tb_sequence_for_player_in_current_tie(gs, player_id)
    if seq:
        gs['final_playoff_scores'][player_id] = seq[-1]
    else:
        gs['final_playoff_scores'][player_id] = 0
    if player_id in gs['playoff_pool']:
        gs['playoff_pool'].remove(player_id)


def resolve_playoff_round(gs: dict):
    """
    Resolves the current playoff round, updates player standings, and
    determines the next playoff group or transitions to final ranking.
    """
    logging.debug(f"[resolve_playoff_round] Starting resolution. Current playoff_round_scores: {gs.get('playoff_round_scores')}")
    scores = gs['playoff_round_scores']
    gs['playoff_history'].append(scores.copy())

    for pid, tb in scores.items():
        gs['all_playoff_history'].setdefault(pid, []).append(int(tb))

    gs['playoff_round_scores'] = {}
    logging.debug(f"[resolve_playoff_round] playoff_round_scores reset. New state: {gs.get('playoff_round_scores')}")
    gs['current_player_index'] = 0

    def worst_subgroup_in_pool() -> list[str]:
        pool = list(gs['playoff_pool'])
        if not pool:
            return []
        seqs = {pid: tuple(_tb_sequence_for_player_in_current_tie(gs, pid)) for pid in pool}
        worst_seq = max(seqs.values()) if seqs else tuple()
        worst_players = [pid for pid, s in seqs.items() if s == worst_seq]
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
            logging.info(f"Player ID '{loser}' has been finalized and removed from the playoff pool.")
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
    # The stats are now pre-calculated and stored in the session.
    game_stats = session.get('game_stats_cache', {})

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

    return render_template(
        'index.html',
        show_stats=True,
        game=gs,
        **game_stats
    )

@app.route('/api/game-stats-status')
def api_game_stats_status():
    """
    Checks if the game stats have been calculated and are in the session.
    """
    is_ready = 'game_stats_cache' in session
    return jsonify({'ready': is_ready})

@app.route('/api/calculate-game-stats', methods=['POST'])
def api_calculate_game_stats():
    """
    Calculates the detailed game stats and stores them in the session.
    This is called in the background from the final standings page.
    """
    gs = _get_state()
    

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
                        'player': p, # Keep the ID here
                        'improvement': int(max_improvement),
                        'from_score': int(seq[from_idx]),
                        'to_score': int(seq[to_idx]),
                        'from_round': int(from_idx) + 1,
                        'to_round': int(to_idx) + 1,
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
                        'player': p, # Keep the ID here
                        'worsening': int(max_worsening),
                        'from_score': int(seq[from_idx]),
                        'to_score': int(seq[to_idx]),
                        'from_round': int(from_idx) + 1,
                        'to_round': int(to_idx) + 1,
                    }
        return out

    # --- Start of heavy calculation logic ---
    def _extract_per_player_sequences(gs: dict) -> dict[str, list[int]]:
        rp = _compute_rounds_played(gs)
        rh = gs.get('round_history', [])[:rp]
        
        # Use the player IDs from the game state as the source of truth.
        player_ids_in_game = gs.get('players', [])
        player_map = gs.get('player_map', {})
        per: dict[str, list[int]] = {}
        for pid in player_ids_in_game:
            per[pid] = [ (rh[i].get(pid, 0) if i < len(rh) else 0) for i in range(rp) ]
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

    per = _extract_per_player_sequences(gs)
    birdie_streaks = {}
    bogey_streaks = {}
    birdie_counts = {}
    bogey_counts = {}
    cumulative_per = {p: np.cumsum(seq).tolist() for p, seq in per.items()}
    on_target_percentages = {}
    avgs = {}
    rounds_counts = {}

    for pid, seq in per.items():
        birdie_streaks[pid] = _longest_streak(seq, lambda x: x < 0)
        bogey_streaks[pid] = _longest_streak(seq, lambda x: x > 0)
        birdie_counts[pid] = sum(1 for v in seq if v < 0)
        bogey_counts[pid] = sum(1 for v in seq if v > 0)
        if seq:
            on_target_rounds = sum(1 for v in seq if v <= 0)
            on_target_percentages[pid] = (on_target_rounds / len(seq)) * 100 if len(seq) > 0 else 0
            avgs[pid] = statistics.mean(seq)
            rounds_counts[pid] = len(seq)
        else:
            on_target_percentages[pid] = 0
            avgs[pid] = 0.0
            rounds_counts[pid] = 0

    players_with_data = [pid for pid in gs.get('players', []) if rounds_counts.get(pid, 0) > 0]
    player_map = gs.get('player_map', {})
    def get_name(pid):
        return player_map.get(pid, {}).get('name', 'Unknown')

    birdie_streak_order = sorted(players_with_data, key=lambda p: (-birdie_streaks[p], -birdie_counts[p], avgs[p]))
    bogey_streak_order = sorted(players_with_data, key=lambda p: (-bogey_streaks[p], -bogey_counts[p], -avgs[p]))
    average_order = sorted(players_with_data, key=lambda p: (avgs[p], -rounds_counts[p]))
    most_birdies_order = sorted(players_with_data, key=lambda p: (-birdie_counts[p], avgs[p]))
    most_bogeys_order = sorted(players_with_data, key=lambda p: (-bogey_counts[p], -avgs[p]))
    on_target_order = sorted(players_with_data, key=lambda p: (-on_target_percentages.get(p, 0), avgs.get(p, 0)))

    birdie_streak_ranking = [{'name': get_name(p), 'streak': birdie_streaks[p], 'count': birdie_counts[p], 'average': avgs[p]} for p in birdie_streak_order]
    bogey_streak_ranking = [{'name': get_name(p), 'streak': bogey_streaks[p], 'count': bogey_counts[p], 'average': avgs[p]} for p in bogey_streak_order]
    average_ranking = [{'name': get_name(p), 'average': avgs[p], 'rounds': rounds_counts[p]} for p in average_order]
    most_birdies_ranking = [{'name': get_name(p), 'count': birdie_counts[p], 'average': avgs[p]} for p in most_birdies_order]
    most_bogeys_ranking = [{'name': get_name(p), 'count': bogey_counts[p], 'average': avgs[p]} for p in most_bogeys_order]
    on_target_ranking = [{'name': get_name(p), 'percentage': on_target_percentages.get(p, 0)} for p in on_target_order]

    best_comebacks = _best_comebacks_by_player(cumulative_per)
    # Now, map the player ID to a name for the final ranking
    comeback_ranking = sorted(
        [{**d, 'player': get_name(d['player'])} for d in best_comebacks.values()], 
        key=lambda d: d['improvement'], reverse=True
    )
    biggest_falls = _biggest_falls_by_player(cumulative_per)
    fall_ranking = sorted(
        [{**d, 'player': get_name(d['player'])} for d in biggest_falls.values()],
        key=lambda d: d['worsening'], reverse=True
    )

    max_birdie_streak = max(birdie_streaks.values(), default=0)
    max_bogey_streak = max(bogey_streaks.values(), default=0)
    max_on_target_percentage = max(on_target_percentages.values(), default=0)
    max_birdie_count = max(birdie_counts.values(), default=0)
    max_bogey_count = max(bogey_counts.values(), default=0)
    max_average_abs = max((abs(a) for a in avgs.values()), default=0)
    max_comeback_improvement = max((cb['improvement'] for cb in best_comebacks.values()), default=0)
    max_fall_worsening = max((f['worsening'] for f in biggest_falls.values()), default=0)

    # Store the calculated stats in the session
    session['game_stats_cache'] = {
        "birdie_streak_ranking": birdie_streak_ranking,
        "bogey_streak_ranking": bogey_streak_ranking,
        "average_ranking": average_ranking,
        "most_birdies_ranking": most_birdies_ranking,
        "most_bogeys_ranking": most_bogeys_ranking,
        "on_target_ranking": on_target_ranking,
        "comeback_ranking": comeback_ranking,
        "fall_ranking": fall_ranking,
        "max_birdie_streak": max_birdie_streak,
        "max_bogey_streak": max_bogey_streak,
        "max_birdie_count": max_birdie_count,
        "max_on_target_percentage": max_on_target_percentage,
        "max_bogey_count": max_bogey_count,
        "max_average_abs": max_average_abs,
        "max_comeback_improvement": max_comeback_improvement,
        "max_fall_worsening": max_fall_worsening,
    }
    session.modified = True

    return jsonify(ok=True)

@app.route('/api/player/<player_name>', methods=['DELETE'])
def delete_player_stats(player_name):
    """
    Deletes all statistics for a given player from Redis or the in-memory fallback.
    This is a destructive operation and cannot be undone.
    """
    if not player_name:
        return jsonify(ok=False, error="Player name is required."), 400

    logging.info(f"Attempting to delete stats for player: {player_name}")

    try:
        if _redis:
            player_id = request.args.get('id') # The frontend will now send the ID
            if not player_id: return jsonify(ok=False, error="Player ID is required."), 400
            player_key = f"player_stats:{player_id}"
            # Check if the player exists before attempting to delete
            if not _redis.exists(player_key):
                return jsonify(ok=False, error="Player not found in Redis."), 404
            
            # Delete the player's stats hash
            _redis.delete(player_key)
            logging.info(f"Deleted Redis key: {player_key}")
        else: # In-memory fallback
            if player_id not in _player_stats_fallback:
                return jsonify(ok=False, error="Player not found in in-memory stats."), 404
            
            del _player_stats_fallback[player_id]
            logging.info(f"Deleted '{player_id}' from in-memory stats.")

        return jsonify(ok=True, message=f"Stats for player '{player_name}' deleted successfully.")

    except Exception as e:
        logging.error(f"Error deleting player stats for {player_name}: {e}")
        return jsonify(ok=False, error="An internal server error occurred."), 500

@app.route('/api/player/<player_name>/rename', methods=['POST'])
def rename_player(player_name):
    """
    Renames a player in the persistent stats database (Redis or fallback).
    """
    data = request.get_json()
    new_name = data.get('new_name', '').strip()
    player_id = data.get('id', '').strip()

    if not player_id:
        return jsonify(ok=False, error="Player ID is required for renaming."), 400

    player_db = get_player_db()
    player_to_rename = player_db.get(player_id)

    if not new_name or len(new_name) > 14:
        return jsonify(ok=False, error="New name is invalid or too long."), 400

    if new_name.strip() == player_name.strip():
        return jsonify(ok=False, error="New name is the same as the old name."), 400 # Still disallow identical names

    # Check if the new name already exists for another player
    for pid, pdata in player_db.items():
        if pid != player_id and pdata['name'].lower() == new_name.lower():
            return jsonify(ok=False, error=f"A player named '{new_name}' already exists."), 409

    logging.info(f"Attempting to rename player '{player_name}' (ID: {player_id}) to '{new_name}'")

    try:
        # Update the name in the central player database (session)
        if player_to_rename:
            player_to_rename['name'] = new_name
            session.modified = True

        # Update the name in the recent players list for all sessions
        _update_recent_names_after_rename(player_name, new_name)

        # Update the name in the current game state if the player is in it
        gs = _get_state()
        if player_id in gs.get('player_map', {}):
            gs['player_map'][player_id]['name'] = new_name
            gs['player_names'] = [gs['player_map'][pid]['name'] for pid in gs['players']]

        # If the rename happens on the final stats page, clear the cache to force a recalc
        if gs.get('phase') == 'final_ranking' and 'game_stats_cache' in session:
            session.pop('game_stats_cache', None)

        _persist(gs)
        return jsonify(ok=True, message=f"Player renamed to '{new_name}'.")

    except Exception as e:
        logging.error(f"Error renaming player {player_name}: {e}")
        return jsonify(ok=False, error="An internal server error occurred."), 500

@app.route('/api/player/<player_name>')
def api_player_stats(player_name):
    """
    Returns historical stats for a single player as JSON.
    """
    player_id = request.args.get('id')
    if not player_id:
        return jsonify({'ok': False, 'error': 'Player ID is required'}), 400

    if _redis:
        player_key = f"player_stats:{player_id}"
        stats_raw = _redis.hgetall(player_key)
        stats = {k: int(v) for k, v in stats_raw.items()}
    else:
        stats = _player_stats_fallback.get(player_id, {})

    if not stats:
        # If no stats exist, return a zeroed-out structure
        stats = {}

    # Calculate derived stats
    games_played = stats.get('games_played', 0)
    wins = stats.get('wins', 0)
    total_rounds = stats.get('total_rounds_all_games', 0)
    total_score = stats.get('total_score_all_games', 0)
    total_on_target = stats.get('total_on_target_rounds_all_games', 0)

    # Calculate current derived stats
    stats['win_percentage'] = (wins / games_played) * 100 if games_played > 0 else 0
    stats['average_score'] = total_score / total_rounds if total_rounds > 0 else 0
    stats['on_target_percentage'] = (total_on_target / total_rounds) * 100 if total_rounds > 0 else 0

    # Fetch and consume deltas if they exist
    if _redis:
        delta_key = f"player_stats_delta:{player_id}"
        delta_raw = _redis.get(delta_key)
        if delta_raw:
            stats['last_game_deltas'] = json.loads(delta_raw)
            _redis.delete(delta_key)

    # If there are deltas, calculate the previous state to show the change
    deltas = stats.get('last_game_deltas')
    if deltas:
        prev_games = games_played - deltas.get('games_played', 0)
        prev_wins = wins - deltas.get('wins', 0)
        prev_total_rounds = total_rounds - deltas.get('total_rounds_all_games', 0)
        prev_total_score = total_score - deltas.get('total_score_all_games', 0)
        prev_total_on_target = total_on_target - deltas.get('total_on_target_rounds_all_games', 0)

        prev_win_perc = (prev_wins / prev_games) * 100 if prev_games > 0 else 0
        prev_avg_score = prev_total_score / prev_total_rounds if prev_total_rounds > 0 else 0
        prev_on_target_perc = (prev_total_on_target / prev_total_rounds) * 100 if prev_total_rounds > 0 else 0

        # Add the change to the deltas dict
        deltas['win_percentage_delta'] = stats['win_percentage'] - prev_win_perc
        deltas['average_score_delta'] = stats['average_score'] - prev_avg_score
        deltas['on_target_percentage_delta'] = stats['on_target_percentage'] - prev_on_target_perc
        
        # Calculate deltas for best/worst scores
        # A delta is the difference between the new record and the previous one.
        if 'best_game_score_delta' in deltas:
            # The delta is already negative, representing the improvement
            pass
        if 'worst_game_score_delta' in deltas:
            # The delta is positive, representing the increase
            pass

        # For display, we need the previous best/worst to show the change from.
        # These are not currently stored, so we'll just show the new record for now.
        # A future improvement could be to store the previous best/worst.


        # Put the updated deltas back into the stats object
        stats['last_game_deltas'] = deltas

    return jsonify({'ok': True, 'player_name': player_name, 'stats': stats})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # For production, debug mode should be off.
    # It can be enabled for local development by setting the DEBUG environment variable.
    is_debug = os.environ.get("DEBUG", "0").lower() in ("1", "true", "yes")
    app.logger.info(f"Starting Flask server on 0.0.0.0:{port} (Debug: {is_debug})")
    app.run(debug=is_debug, host='0.0.0.0', port=port)
