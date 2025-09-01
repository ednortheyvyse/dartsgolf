import os
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from collections import defaultdict
import statistics  # for consistency (std dev)

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
        'round_history': [],             # list of dicts: round -> {player: delta}
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
        'recent_names': [],

        # End game toggle
        'end_after_round': False,

        # How many rounds actually had any scores (for early end)
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
    # Sort by (base_total, tb1, tb2, ...) — lower is better
    def key(player: str):
        base = int(gs['scores'].get(player, 0))
        tb_seq = gs['all_playoff_history'].get(player, [])
        return (base, *tb_seq)
    return sorted(gs['players'], key=key)

def _compute_rounds_played(gs: dict) -> int:
    """Count non-empty rounds (any player recorded a value)."""
    return sum(1 for r in gs.get('round_history', []) if r)  # empty dicts are falsy

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
        'end_after_round': bool(gs.get('end_after_round', False)),
        'rounds_played': int(gs.get('rounds_played', 0)),
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
        gs['rounds_played'] = _compute_rounds_played(gs)  # ensure up to date
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

# Toggle end-after-round
@app.post('/api/end')
def api_end_after_round():
    gs = _get_state()
    if gs['phase'] == 'playing':
        gs['end_after_round'] = not bool(gs.get('end_after_round', False))
    return jsonify({'ok': True, 'game': _serialize_game(gs)})
# ----------------------------------------------------

# ---------------- Helpers ----------------
def _apply_score(gs: dict, score_change: int):
    if gs['phase'] == 'playing':
        player = gs['players'][gs['current_player_index']]
        gs['scores'][player] += score_change
        gs['round_history'][gs['current_round'] - 1][player] = score_change
        gs['undo_history'].append({'player_index': gs['current_player_index'], 'score_change': score_change})

        # Is this the last player for this round?
        last_index = len(gs['players']) - 1
        was_last_in_round = (gs['current_player_index'] == last_index)

        if was_last_in_round and gs.get('end_after_round'):
            # End immediately after finishing this round
            initiate_playoffs(gs)
            return

        # Otherwise continue normal progression
        gs['current_player_index'] += 1
        if gs['current_player_index'] >= len(gs['players']):
            gs['current_player_index'] = 0
            gs['current_round'] += 1

        # Normal auto-end at 20 rounds
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
    # Clear the flag once the game actually ends
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
        gs['rounds_played'] = _compute_rounds_played(gs)  # only show rounds actually played
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
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def _draw_centered_text(draw: ImageDraw.ImageDraw, xy, text, font, fill, box_w, box_h):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = xy[0] + (box_w - tw) / 2
    y = xy[1] + (box_h - th) / 2
    draw.text((x, y), text, font=font, fill=fill)

def _ensure_final(gs: dict):
    if gs['phase'] != 'final_ranking':
        return
    if gs['final_standings']:
        return
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

def render_final_png(gs: dict, width: int, height: int) -> BytesIO:
    """
    Final standings PNG showing only the rounds actually played.
    """
    from PIL import Image, ImageDraw, ImageFont

    # Colors
    BG = (26, 26, 26)
    TEXT = (240, 240, 240)
    BORDER = (68, 68, 68)
    RED   = (0xE3, 0x29, 0x2E)   # #E3292E
    GREEN = (0x30, 0x9F, 0x6A)    # #309F6A
    ZERO = (0, 0, 0)            # BLACK for zero
    ACCENT_BG = (249, 223, 188) # #F9DFBC for score cells
    TOTAL_BG = (51, 51, 51)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    PADDING = 12
    MIN_CELL_H = 36

    cols = max(1, len(gs['final_standings'])) + 1  # +1 label col
    tb_rows = int(gs.get('max_playoff_rounds', 0))
    rounds_played = int(gs.get('rounds_played', 0)) or _compute_rounds_played(gs)

    row_count = 2 + 1 + tb_rows + rounds_played  # headers(2) + final(1) + TB + played rounds

    inner_w = max(200, width - 2 * PADDING)
    inner_h = max(200, height - 2 * PADDING)
    cell_w = inner_w // cols
    cell_h = max(MIN_CELL_H, inner_h // row_count)

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

    # Fonts
    f_small  = _pick_font(max(14, int(cell_h * 0.62)))  # TB/labels
    f_medium = _pick_font(max(16, int(cell_h * 0.72)))  # headers, labels
    f_big    = _pick_font(max(18, int(cell_h * 0.82)))  # ordinals
    f_total  = _pick_font(max(18, int(cell_h * 0.78)))  # running total
    f_delta  = _pick_font(max(11, int(cell_h * 0.50)))  # wedge text

    def rect(x, y, w, h, fill=None, outline=BORDER):
        d.rectangle([x, y, x + w, y + h], fill=fill, outline=outline)

    # Precompute running totals
    players = [st['name'] for st in gs['final_standings']]
    running_totals = {p: [] for p in players}
    totals_so_far = {p: 0 for p in players}
    for r in range(rounds_played):
        for p in players:
            v = int(gs['round_history'][r].get(p, 0))
            totals_so_far[p] += v
            running_totals[p].append(totals_so_far[p])

    # Row 0: Ordinals
    y = PADDING
    rect(PADDING, y, inner_w, cell_h)
    x = PADDING
    rect(x, y, cell_w, cell_h); x += cell_w
    for st in gs['final_standings']:
        rect(x, y, cell_w, cell_h)
        _draw_centered_text(d, (x, y), _ordinal(int(st['rank'])), f_big, TEXT, cell_w, cell_h)
        x += cell_w
    y += cell_h

    # Row 1: Names
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

    # Row 2: Final totals (neutral dark row)
    x = PADDING
    for _ in range(cols):
        rect(x, y, cell_w, cell_h, fill=TOTAL_BG)
        x += cell_w
    x = PADDING
    _draw_centered_text(d, (x, y), "Final", f_medium, TEXT, cell_w, cell_h)
    x += cell_w
    for st in gs['final_standings']:
        s = int(st['score'])
        txt = f"+{s}" if s > 0 else f"{s}"
        rect(x, y, cell_w, cell_h, fill=TOTAL_BG)
        _draw_centered_text(d, (x, y), txt, f_medium, TEXT, cell_w, cell_h)
        x += cell_w
    y += cell_h

    # Tie-breakers (neutral cells)
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
                rect(x, y, cell_w, cell_h)
                _draw_centered_text(d, (x, y), f"+{v}" if v>0 else f"{v}", f_small, TEXT, cell_w, cell_h)
            else:
                rect(x, y, cell_w, cell_h)
                _draw_centered_text(d, (x, y), "-", f_small, TEXT, cell_w, cell_h)
            x += cell_w
        y += cell_h

    # Base rounds: only those played
    for r_idx in range(rounds_played - 1, -1, -1):
        x = PADDING
        for _ in range(cols):
            rect(x, y, cell_w, cell_h)
            x += cell_w

        # Label col
        x = PADDING
        _draw_centered_text(d, (x, y), f"{r_idx+1:02d}", f_medium, TEXT, cell_w, cell_h)
        x += cell_w

        for st in gs['final_standings']:
            name = st['name']
            raw = gs['round_history'][r_idx].get(name)
            if raw is None:
                rect(x, y, cell_w, cell_h)
            else:
                v = int(raw)
                rect(x, y, cell_w, cell_h, fill=ACCENT_BG)

                total_val = int(running_totals[name][r_idx])
                total_text = f"{'+' if total_val > 0 else ''}{total_val}"
                tbx = d.textbbox((0, 0), total_text, font=f_total)
                tw = tbx[2] - tbx[0]; th = tbx[3] - tbx[1]
                cx = x + (cell_w - tw) / 2
                cy = y + (cell_h - th) / 2
                d.text((cx, cy), total_text, font=f_total, fill=BLACK)

                wedge_r = int(min(cell_w, cell_h) * 0.52)
                bx1 = x + cell_w; by1 = y + cell_h
                bx0 = bx1 - 2*wedge_r; by0 = by1 - 2*wedge_r
                delta_text = f"+{v}" if v > 0 else f"{v}"
                fill_color = (0xE3, 0x29, 0x2E) if v > 0 else ((0x30, 0x9F, 0x6A) if v < 0 else (0,0,0))
                d.pieslice([bx0, by0, bx1, by1], start=270, end=360, fill=fill_color)

                tbd = d.textbbox((0,0), delta_text, font=f_delta)
                dw = tbd[2]-tbd[0]; dh = tbd[3]-tbd[1]
                tx = x + cell_w - max(8, int(wedge_r*0.45)) - dw/2
                ty = y + cell_h - max(8, int(wedge_r*0.45)) - dh/2
                d.text((tx, ty), delta_text, font=f_delta, fill=(255,255,255))

            x += cell_w
        y += cell_h

    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf
    
@app.route('/stats')
def stats():
    gs = _get_state()
    if gs['phase'] != 'final_ranking':
        flash("Stats are only available after the game ends.", "error")
        return redirect(url_for('index'))

    round_history = gs['round_history']
    players = gs['players']

    # Per-player aggregates
    stats_data = {}
    for p in players:
        streak = {'birdie': 0, 'bogey': 0, 'best_birdie': 0, 'best_bogey': 0}
        scores = []
        for r in round_history:
            v = r.get(p)
            if v is not None:
                scores.append(v)
                if v < 0:
                    streak['birdie'] += 1
                    streak['bogey'] = 0
                elif v > 0:
                    streak['bogey'] += 1
                    streak['birdie'] = 0
                else:
                    streak['bogey'] = 0
                    streak['birdie'] = 0
                streak['best_birdie'] = max(streak['best_birdie'], streak['birdie'])
                streak['best_bogey'] = max(streak['best_bogey'], streak['bogey'])
        total = sum(scores)
        rounds = len(scores)
        avg = total / rounds if rounds else 0
        stats_data[p] = {
            'average': avg,
            'birdie_streak': streak['best_birdie'],
            'bogey_streak': streak['best_bogey']
        }

    best_birdie = max(stats_data.items(), key=lambda x: x[1]['birdie_streak']) if stats_data else ("—", {'birdie_streak': 0})
    best_bogey  = max(stats_data.items(), key=lambda x: x[1]['bogey_streak'])  if stats_data else ("—", {'bogey_streak': 0})
    best_avg    = min(stats_data.items(), key=lambda x: x[1]['average'])       if stats_data else ("—", {'average': 0})
    worst_avg   = max(stats_data.items(), key=lambda x: x[1]['average'])       if stats_data else ("—", {'average': 0})

    # Best Single Round
    best_round = None
    for i, round_data in enumerate(round_history):
        for player, val in round_data.items():
            if best_round is None or val < best_round['score']:
                best_round = {'round': i + 1, 'player': player, 'score': val}

    # Most Birdies / Bogeys
    birdie_counts = {p: sum(1 for r in round_history if r.get(p, 0) < 0) for p in players}
    bogey_counts  = {p: sum(1 for r in round_history if r.get(p, 0) > 0) for p in players}
    most_birdies  = max(birdie_counts.items(), key=lambda x: x[1]) if birdie_counts else ("—", 0)
    most_bogeys   = max(bogey_counts.items(),  key=lambda x: x[1]) if bogey_counts else ("—", 0)

    # Most Consistent Player (population stdev; tie-breaks: better avg, more rounds, smaller swing, name)
    consistency_pool = []
    for p in players:
        scores = [r.get(p) for r in round_history if p in r]
        if len(scores) >= 6:
            sd = statistics.pstdev(scores)
            avg = sum(scores) / len(scores)
            swing = (max(scores) - min(scores)) if scores else 0
            consistency_pool.append((p, sd, avg, len(scores), swing))

    (most_consistent_name,
     most_consistent_stdev,
     most_consistent_avg,
     most_consistent_rounds,
     most_consistent_swing) = (None, None, None, None, None)

    if consistency_pool:
        consistency_pool.sort(key=lambda t: (t[1], t[2], -t[3], t[4], t[0].lower()))
        (most_consistent_name,
         most_consistent_stdev,
         most_consistent_avg,
         most_consistent_rounds,
         most_consistent_swing) = consistency_pool[0]

    # Comeback Player (worst -> best cumulative swing)
    comebacks = []
    for p in players:
        cum = []
        total = 0
        for r in round_history:
            total += r.get(p, 0)
            cum.append(total)
        if not cum:
            continue
        prev_worst = cum[0]
        prev_worst_idx = 0
        best_improve = 0
        best_from_val = prev_worst
        best_from_idx = 0
        best_to_val = cum[0]
        best_to_idx = 0
        for i in range(1, len(cum)):
            v = cum[i]
            improvement = prev_worst - v  # positive improvement = moved more negative (better in golf)
            if (improvement > best_improve or
                (improvement == best_improve and (v < best_to_val or (v == best_to_val and i < best_to_idx)))):
                best_improve = improvement
                best_from_val = prev_worst
                best_from_idx = prev_worst_idx
                best_to_val = v
                best_to_idx = i
            if v > prev_worst:
                prev_worst = v
                prev_worst_idx = i
        if best_improve > 0:
            comebacks.append({
                'player': p,
                'improvement': int(best_improve),
                'from_score': int(best_from_val),
                'from_round': best_from_idx + 1,
                'to_score': int(best_to_val),
                'to_round': best_to_idx + 1,
            })
    if comebacks:
        comebacks.sort(key=lambda x: (-x['improvement'], x['to_score'], x['to_round'], x['player'].lower()))
        comeback_player = comebacks[0]
    else:
        comeback_player = None

    return render_template(
        'stats.html',
        best_birdie=best_birdie,
        best_bogey=best_bogey,
        best_avg=best_avg,
        worst_avg=worst_avg,
        best_round=best_round,
        most_birdies=most_birdies,
        most_bogeys=most_bogeys,
        most_consistent_name=most_consistent_name,
        most_consistent_stdev=most_consistent_stdev,
        most_consistent_avg=most_consistent_avg,
        most_consistent_rounds=most_consistent_rounds,
        most_consistent_swing=most_consistent_swing,
        comeback_player=comeback_player
    )

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

    width = max(500, min(width, 1100))
    height = max(400, min(height, 800))

    img_buf = render_final_png(gs, width, height)
    return send_file(img_buf, mimetype='image/png', as_attachment=True, download_name='final-standings.png')

# ---------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
