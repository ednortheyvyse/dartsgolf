import os
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from collections import defaultdict

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

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

    # Step back to the previous player
    prev_idx = (gs['current_player_index'] - 1 + len(gs['players'])) % len(gs['players'])
    gs['current_player_index'] = prev_idx

    # If we wrapped from player 0 back to last, we also step the round back
    if prev_idx == len(gs['players']) - 1:
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

    # If any tie remains (duplicate TB values), continue another TB round
    if len(set(scores.values())) < len(scores):
        gs['playoff_round'] += 1
        gs['current_player_index'] = 0
        gs['playoff_round_scores'] = {}
        return

    for p, tb in scores.items():
        gs['final_playoff_scores'][p] = tb

    start_next_playoff(gs)
# ----------------------------------------

# ---------------- PNG Export (server-side via Pillow) ----------------
def _ordinal(n: int) -> str:
    s = 'th'
    if n % 10 == 1 and n % 100 != 11: s = 'st'
    elif n % 10 == 2 and n % 100 != 12: s = 'nd'
    elif n % 10 == 3 and n % 100 != 13: s = 'rd'
    return f"{n}{s}"

def _pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try DejaVu (bundled with many Pillow installs), fallback to default bitmap
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def _draw_centered_text(draw: ImageDraw.ImageDraw, xy, text, font, fill, box_w, box_h):
    # Center text inside a box anchored at (xy)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = xy[0] + (box_w - tw) / 2
    y = xy[1] + (box_h - th) / 2
    draw.text((x, y), text, font=font, fill=fill)

def _ensure_final(gs: dict):
    """Make sure final_standings/max_playoff_rounds are populated (mirrors index())."""
    if gs['phase'] != 'final_ranking':
        return
    if gs['final_standings']:
        return
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

def render_final_png(gs: dict, width: int, height: int) -> BytesIO:
    """
    Draw a PNG of the final standings (headers, Final row, tie-breakers, and 20 base rounds).
    Colors mirror your UI scheme. It scales to the requested width/height.
    """
    # Colors (match your CSS theme)
    BG = (26, 26, 26)              # --bg-color
    TEXT = (240, 240, 240)         # --text-color
    BORDER = (68, 68, 68)          # --border-color
    RED = (217, 39, 39)            # --primary-color  (pos / +)
    GREEN = (0, 135, 70)           # --secondary-color (neg / -)
    ZERO = (97, 97, 97)            # zero cells
    ACCENT_BG = (229, 216, 177)    # default cell background
    ACCENT_TXT = (0, 0, 0)
    TOTAL_BG = (51, 51, 51)        # total row background

    # Dimensions & layout
    padding = 16
    cols = max(1, len(gs['final_standings'])) + 1  # +1 for label column
    # Rows: 2 header rows + 1 final + TB + 20 base rounds
    tb_rows = int(gs.get('max_playoff_rounds', 0))
    row_count = 2 + 1 + tb_rows + 20
    # Fit to requested width/height
    inner_w = max(200, width - 2 * padding)
    inner_h = max(200, height - 2 * padding)
    cell_w = inner_w // cols
    cell_h = inner_h // row_count
    # If the requested height is too small, keep a minimum row height
    cell_h = max(28, cell_h)
    # Recompute inner_h & total height from actual cell size
    inner_h = cell_h * row_count
    canvas_w = cell_w * cols + 2 * padding
    canvas_h = inner_h + 2 * padding

    # Create image
    im = Image.new("RGB", (canvas_w, canvas_h), BG)
    d = ImageDraw.Draw(im)

    # Fonts (scaled to row height)
    f_small = _pick_font(max(12, int(cell_h * 0.42)))
    f_medium = _pick_font(max(14, int(cell_h * 0.52)))
    f_big = _pick_font(max(16, int(cell_h * 0.60)))

    # Helpers for rectangles
    def rect(x, y, w, h, fill=None, outline=BORDER):
        d.rectangle([x, y, x + w, y + h], fill=fill, outline=outline)

    # Column headers (row 0 & row 1)
    y = padding
    # Row 0: ordinals
    rect(padding, y, inner_w, cell_h, fill=None, outline=BORDER)
    # Labels column blank
    _x = padding
    rect(_x, y, cell_w, cell_h, fill=None, outline=BORDER)
    _x += cell_w
    for st in gs['final_standings']:
        rect(_x, y, cell_w, cell_h, fill=None, outline=BORDER)
        _draw_centered_text(d, (_x, y), _ordinal(int(st['rank'])), f_big, TEXT, cell_w, cell_h)
        _x += cell_w
    y += cell_h

    # Row 1: Player names
    rect(padding, y, inner_w, cell_h, fill=None, outline=BORDER)
    _x = padding
    rect(_x, y, cell_w, cell_h, fill=None, outline=BORDER)
    _draw_centered_text(d, (_x, y), "Player", f_medium, TEXT, cell_w, cell_h)
    _x += cell_w
    for st in gs['final_standings']:
        rect(_x, y, cell_w, cell_h, fill=None, outline=BORDER)
        _draw_centered_text(d, (_x, y), st['name'], f_medium, TEXT, cell_w, cell_h)
        _x += cell_w
    y += cell_h

    # Row 2: Final totals
    _x = padding
    for c in range(cols):
        fill = TOTAL_BG if c >= 0 else None
        rect(_x, y, cell_w, cell_h, fill=fill, outline=BORDER)
        _x += cell_w

    _x = padding
    rect(_x, y, cell_w, cell_h, fill=TOTAL_BG, outline=BORDER)
    _draw_centered_text(d, (_x, y), "Final", f_medium, TEXT, cell_w, cell_h)
    _x += cell_w

    for st in gs['final_standings']:
        s = int(st['score'])
        if s > 0:
            fill, t = RED, TEXT
            txt = f"+{s}"
        elif s < 0:
            fill, t = GREEN, TEXT
            txt = str(s)
        else:
            fill, t = ZERO, TEXT
            txt = "0"
        rect(_x, y, cell_w, cell_h, fill=fill, outline=BORDER)
        _draw_centered_text(d, (_x, y), txt, f_medium, t, cell_w, cell_h)
        _x += cell_w
    y += cell_h

    # Tie-breaker rows (TB N..1)
    for i in range(tb_rows, 0, -1):
        _x = padding
        for c in range(cols):
            rect(_x, y, cell_w, cell_h, fill=None, outline=BORDER)
            _x += cell_w

        _x = padding
        rect(_x, y, cell_w, cell_h, fill=None, outline=BORDER)
        _draw_centered_text(d, (_x, y), f"TB {i:02d}", f_medium, TEXT, cell_w, cell_h)
        _x += cell_w

        for st in gs['final_standings']:
            hist = gs['all_playoff_history'].get(st['name'], [])
            if i <= len(hist):
                v = int(hist[i-1])
                if v > 0:
                    fill, t, txt = RED, TEXT, f"+{v}"
                elif v < 0:
                    fill, t, txt = GREEN, TEXT, str(v)
                else:
                    fill, t, txt = ZERO, TEXT, "0"
                rect(_x, y, cell_w, cell_h, fill=fill, outline=BORDER)
                _draw_centered_text(d, (_x, y), txt, f_small, t, cell_w, cell_h)
            else:
                rect(_x, y, cell_w, cell_h, fill=ACCENT_BG, outline=BORDER)
                _draw_centered_text(d, (_x, y), "-", f_small, ACCENT_TXT, cell_w, cell_h)
            _x += cell_w
        y += cell_h

    # Base rounds 20..1
    for round_index in range(19, -1, -1):
        _x = padding
        for c in range(cols):
            rect(_x, y, cell_w, cell_h, fill=None, outline=BORDER)
            _x += cell_w

        _x = padding
        rect(_x, y, cell_w, cell_h, fill=None, outline=BORDER)
        _draw_centered_text(d, (_x, y), f"{round_index+1:02d}", f_medium, TEXT, cell_w, cell_h)
        _x += cell_w

        for st in gs['final_standings']:
            name = st['name']
            v = gs['round_history'][round_index].get(name)
            if v is None:
                rect(_x, y, cell_w, cell_h, fill=ACCENT_BG, outline=BORDER)
            else:
                v = int(v)
                if v > 0:
                    fill, t, txt = RED, TEXT, f"+{v}"
                elif v < 0:
                    fill, t, txt = GREEN, TEXT, str(v)
                else:
                    fill, t, txt = ZERO, TEXT, "0"
                rect(_x, y, cell_w, cell_h, fill=fill, outline=BORDER)
                _draw_centered_text(d, (_x, y), txt, f_small, t, cell_w, cell_h)
            _x += cell_w
        y += cell_h

    # Output buffer
    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf

@app.get('/export.png')
def export_png():
    if not PIL_AVAILABLE:
        return "Pillow not installed. Run: pip install pillow", 500

    gs = _get_state()
    if gs.get('phase') != 'final_ranking':
        return "Final standings not available yet.", 400

    # Make sure final standings are computed
    _ensure_final(gs)

    # Dimensions from client (CSS pixels * devicePixelRatio recommended)
    try:
        width = int(request.args.get('width', '900'))
        height = int(request.args.get('height', '600'))
    except Exception:
        width, height = 900, 600

    # Safety clamps
    width = max(400, min(width, 6000))
    height = max(300, min(height, 6000))

    img_buf = render_final_png(gs, width, height)
    return send_file(img_buf, mimetype='image/png', as_attachment=True, download_name='final-standings.png')
# ---------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
