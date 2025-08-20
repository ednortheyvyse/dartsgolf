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
    Draw a PNG of the final standings with legible text and per-round running totals ("T +…")
    shown as a pill beneath the delta in each base-round cell.
    """
    from PIL import Image, ImageDraw, ImageFont

    # Colors (match your CSS theme)
    BG = (26, 26, 26)
    TEXT = (240, 240, 240)
    BORDER = (68, 68, 68)
    RED = (217, 39, 39)          # + (pos)
    GREEN = (0, 135, 70)         # - (neg)
    ZERO = (97, 97, 97)          # 0
    ACCENT_BG = (229, 216, 177)  # empty cell background
    ACCENT_TXT = (0, 0, 0)
    TOTAL_BG = (51, 51, 51)

    # Layout tuned for readability
    PADDING = 12
    MIN_CELL_H = 36  # keep text chunky/legible

    cols = max(1, len(gs['final_standings'])) + 1  # +1 for label column
    tb_rows = int(gs.get('max_playoff_rounds', 0))
    row_count = 2 + 1 + tb_rows + 20  # headers(2) + final(1) + TB + rounds(20)

    # Compute cell size from requested canvas, then enforce readable minimums
    inner_w = max(200, width - 2 * PADDING)
    inner_h = max(200, height - 2 * PADDING)
    cell_w = inner_w // cols
    cell_h = inner_h // row_count
    cell_h = max(MIN_CELL_H, cell_h)

    # Recompute exact canvas dimensions
    inner_h = cell_h * row_count
    inner_w = cell_w * cols
    canvas_w = inner_w + 2 * PADDING
    canvas_h = inner_h + 2 * PADDING

    im = Image.new("RGB", (canvas_w, canvas_h), BG)
    d = ImageDraw.Draw(im)

    def _pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    # Font sizes
    f_small  = _pick_font(max(14, int(cell_h * 0.62)))  # TB values + round deltas (if tight)
    f_medium = _pick_font(max(16, int(cell_h * 0.72)))  # headers, labels
    f_big    = _pick_font(max(18, int(cell_h * 0.82)))  # ordinal header
    f_delta  = _pick_font(max(16, int(cell_h * 0.70)))  # big delta in base rounds
    f_pill   = _pick_font(max(12, int(cell_h * 0.48)))  # pill text "T +…"

    def rect(x, y, w, h, fill=None, outline=BORDER):
        d.rectangle([x, y, x + w, y + h], fill=fill, outline=outline)

    def _ordinal(n: int) -> str:
        s = 'th'
        if n % 10 == 1 and n % 100 != 11: s = 'st'
        elif n % 10 == 2 and n % 100 != 12: s = 'nd'
        elif n % 10 == 3 and n % 100 != 13: s = 'rd'
        return f"{n}{s}"

    def _draw_centered_text(draw, xy, text, font, fill, box_w, box_h):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = xy[0] + (box_w - tw) / 2
        y = xy[1] + (box_h - th) / 2
        draw.text((x, y), text, font=font, fill=fill)

    def _pill_colors(total_val: int):
        if total_val > 0:  # positive running total -> red pill, white text
            return RED, (255, 255, 255)
        if total_val < 0:  # negative running total -> green pill, white text
            return GREEN, (255, 255, 255)
        return ZERO, (255, 255, 255)  # zero -> grey pill, white text

    def _cell_fill_for_delta(v: int):
        if v > 0:  return RED, TEXT, f"+{v}"
        if v < 0:  return GREEN, TEXT, str(v)
        return ZERO, TEXT, "0"

    # Precompute running totals per player for the 20 base rounds
    players = [st['name'] for st in gs['final_standings']]
    running_totals = {p: [] for p in players}
    totals_so_far = {p: 0 for p in players}
    for r in range(20):
        for p in players:
            v = int(gs['round_history'][r].get(p, 0))
            totals_so_far[p] += v
            running_totals[p].append(totals_so_far[p])

    # --- Row 0: Ordinals ---
    y = PADDING
    rect(PADDING, y, inner_w, cell_h)
    x = PADDING
    rect(x, y, cell_w, cell_h); x += cell_w  # label col
    for st in gs['final_standings']:
        rect(x, y, cell_w, cell_h)
        _draw_centered_text(d, (x, y), _ordinal(int(st['rank'])), f_big, TEXT, cell_w, cell_h)
        x += cell_w
    y += cell_h

    # --- Row 1: Player names ---
    rect(PADDING, y, inner_w, cell_h)
    x = PADDING
    rect(x, y, cell_w, cell_h)
    _draw_centered_text(d, (x, y), "Player", f_medium, TEXT, cell_w, cell_h)
    x += cell_w
    for st in gs['final_standings']:
        rect(x, y, cell_w, cell_h)
        _draw_centered_text(d, (x, y), st['name'], f_medium, TEXT, cell_w, cell_h)
        x += cell_w
    y += cell_h

    # --- Row 2: Final totals ---
    x = PADDING
    for _ in range(cols):
        rect(x, y, cell_w, cell_h, fill=TOTAL_BG)
        x += cell_w
    x = PADDING
    _draw_centered_text(d, (x, y), "Final", f_medium, TEXT, cell_w, cell_h)
    x += cell_w
    for st in gs['final_standings']:
        s = int(st['score'])
        fill, t, txt = _cell_fill_for_delta(s)
        rect(x, y, cell_w, cell_h, fill=fill)
        _draw_centered_text(d, (x, y), txt, f_medium, t, cell_w, cell_h)
        x += cell_w
    y += cell_h

    # --- Tie-breakers: TB N..1 (delta only; no running total) ---
    for i in range(tb_rows, 0, -1):
        x = PADDING
        for _ in range(cols):
            rect(x, y, cell_w, cell_h)
            x += cell_w
        x = PADDING
        _draw_centered_text(d, (x, y), f"TB {i:02d}", f_medium, TEXT, cell_w, cell_h)
        x += cell_w
        for st in gs['final_standings']:
            hist = gs['all_playoff_history'].get(st['name'], [])
            if i <= len(hist):
                v = int(hist[i-1])
                fill, t, txt = _cell_fill_for_delta(v)
                rect(x, y, cell_w, cell_h, fill=fill)
                _draw_centered_text(d, (x, y), txt, f_small, t, cell_w, cell_h)
            else:
                rect(x, y, cell_w, cell_h, fill=ACCENT_BG)
                _draw_centered_text(d, (x, y), "-", f_small, ACCENT_TXT, cell_w, cell_h)
            x += cell_w
        y += cell_h

    # --- Base rounds: 20..1 (delta + running total pill) ---
    for r_idx in range(19, -1, -1):
        x = PADDING
        for _ in range(cols):
            rect(x, y, cell_w, cell_h)
            x += cell_w

        # label column
        x = PADDING
        _draw_centered_text(d, (x, y), f"{r_idx+1:02d}", f_medium, TEXT, cell_w, cell_h)
        x += cell_w

        for st in gs['final_standings']:
            name = st['name']
            raw = gs['round_history'][r_idx].get(name)
            if raw is None:
                rect(x, y, cell_w, cell_h, fill=ACCENT_BG)
            else:
                v = int(raw)
                fill, t, txt = _cell_fill_for_delta(v)
                rect(x, y, cell_w, cell_h, fill=fill)

                # Two-line layout within the cell
                # 1) Delta (big, top)
                delta_bbox = d.textbbox((0, 0), txt, font=f_delta)
                delta_w = delta_bbox[2] - delta_bbox[0]
                delta_h = delta_bbox[3] - delta_bbox[1]

                # 2) Pill ("T +…", centered under delta)
                total_val = int(running_totals[name][r_idx])
                pill_text = f"T {'+' if total_val > 0 else ''}{total_val}"
                pill_bg, pill_fg = _pill_colors(total_val)

                pill_bbox = d.textbbox((0, 0), pill_text, font=f_pill)
                tw = pill_bbox[2] - pill_bbox[0]
                th = pill_bbox[3] - pill_bbox[1]
                pad_x = max(8, int(cell_w * 0.08))
                pad_y = max(3, int(cell_h * 0.10))
                pill_w = tw + 2 * pad_x
                pill_h = th + 2 * pad_y

                total_block_h = delta_h + 6 + pill_h  # spacing = 6px
                base_y = y + (cell_h - total_block_h) / 2

                # draw delta
                dx = x + (cell_w - delta_w) / 2
                dy = base_y
                d.text((dx, dy), txt, font=f_delta, fill=t)

                # draw pill background (rounded)
                px0 = x + (cell_w - pill_w) / 2
                py0 = dy + delta_h + 6
                px1 = px0 + pill_w
                py1 = py0 + pill_h
                radius = pill_h / 2
                try:
                    d.rounded_rectangle([px0, py0, px1, py1], radius=radius, fill=pill_bg)
                except Exception:
                    # Fallback for very old Pillow: draw a normal rectangle
                    d.rectangle([px0, py0, px1, py1], fill=pill_bg)

                # pill text
                tx = px0 + pad_x
                ty = py0 + pad_y
                d.text((tx, ty), pill_text, font=f_pill, fill=pill_fg)

            x += cell_w
        y += cell_h

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

    _ensure_final(gs)

    try:
        width = int(request.args.get('width', '900'))
        height = int(request.args.get('height', '600'))
    except Exception:
        width, height = 900, 600

    # Tighter clamps for a smaller, readable PNG
    width = max(500, min(width, 1100))
    height = max(400, min(height, 800))

    img_buf = render_final_png(gs, width, height)
    return send_file(img_buf, mimetype='image/png', as_attachment=True, download_name='final-standings.png')

# ---------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
