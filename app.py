from flask import Flask, render_template, request, redirect, url_for, flash
from collections import defaultdict

app = Flask(__name__)
# Set a secret key for flashing messages (change to something secure for production)
app.secret_key = "change-this-to-a-secure-random-string"

# This dictionary will hold the entire state of our game
game_state = {
    'players': [], 'scores': {}, 'round_history': [], 'current_round': 1,
    'current_player_index': 0, 'phase': 'setup', 'winner': None, 'undo_history': [],
    # Updated state for the advanced playoff system
    'pending_playoffs': [], 'playoff_group': [], 'playoff_round': 1,
    'playoff_round_scores': {}, 'playoff_history': [], 'playoff_base_score': 0,
    'final_standings': [], 'final_playoff_scores': {}, 'all_playoff_history': {},
    'max_playoff_rounds': 0
}

def reset_game():
    """Resets the game state to its initial values."""
    global game_state
    game_state = {
        'players': [], 'scores': {}, 'round_history': [], 'current_round': 1,
        'current_player_index': 0, 'phase': 'setup', 'winner': None,
        'undo_history': [], 'pending_playoffs': [], 'playoff_group': [],
        'playoff_round': 1, 'playoff_round_scores': {}, 'playoff_history': [],
        'playoff_base_score': 0, 'final_standings': [], 'final_playoff_scores': {},
        'all_playoff_history': {}, 'max_playoff_rounds': 0
    }

@app.route('/')
def index():
    """Renders the main game page."""
    if game_state['phase'] == 'final_ranking' and not game_state['final_standings']:
        # This logic runs once to calculate the final ranks and display scores
        player_scores = sorted([(score, player) for player, score in game_state['scores'].items()])
        
        standings = []
        last_score = -999
        current_rank = 0
        for score, player in player_scores:
            if score != last_score:
                current_rank = len(standings) + 1
            
            base_score = int(score)
            tie_breaker_score = game_state['final_playoff_scores'].get(player, 0)
            display_score = base_score + tie_breaker_score
            
            standings.append({
                'rank': current_rank, 'name': player, 'score': display_score
            })
            last_score = score
        game_state['final_standings'] = standings

        # Calculate the max number of playoff rounds any player had
        max_rounds = 0
        if game_state['all_playoff_history']:
            max_rounds = max(len(h) for h in game_state['all_playoff_history'].values())
        game_state['max_playoff_rounds'] = max_rounds

    return render_template('index.html', game=game_state)

@app.route('/start', methods=['POST'])
def start_game():
    player_names = request.form.get('players', '')
    # Split, trim, and filter empties
    players = [name.strip() for name in player_names.split(',') if name and name.strip()]
    if not players:
        flash("Please enter at least one player name.", "error")
        return redirect(url_for('index'))

    # CHECK 1: Ensure no duplicates (case-insensitive)
    lowered = [p.lower() for p in players]
    if len(set(lowered)) != len(lowered):
        # Build a simple message listing the duplicates
        dups = sorted({name for name in players if lowered.count(name.lower()) > 1})
        flash(f"Duplicate player name(s) not allowed: {', '.join(dups)}. Please enter unique names.", "error")
        return redirect(url_for('index'))

    # If all good, reset and start
    reset_game()
    game_state['players'] = players
    game_state['scores'] = {player: 0 for player in players}
    game_state['round_history'] = [{} for _ in range(20)]
    game_state['phase'] = 'playing'
    return redirect(url_for('index'))

@app.route('/score', methods=['POST'])
def record_score():
    score_change = int(request.form.get('score'))
    if game_state['phase'] == 'playing':
        player = game_state['players'][game_state['current_player_index']]
        game_state['scores'][player] += score_change
        game_state['round_history'][game_state['current_round'] - 1][player] = score_change
        game_state['undo_history'].append({'player_index': game_state['current_player_index'], 'score_change': score_change})
        game_state['current_player_index'] += 1
        if game_state['current_player_index'] >= len(game_state['players']):
            game_state['current_player_index'] = 0
            game_state['current_round'] += 1
        if game_state['current_round'] > 20:
            initiate_playoffs()
    elif game_state['phase'] == 'playoff':
        player = game_state['playoff_group'][game_state['current_player_index']]
        game_state['playoff_round_scores'][player] = score_change
        game_state['current_player_index'] += 1
        if game_state['current_player_index'] >= len(game_state['playoff_group']):
            resolve_playoff_round()
    return redirect(url_for('index'))

def initiate_playoffs():
    scores_to_players = defaultdict(list)
    for player, score in game_state['scores'].items():
        scores_to_players[int(score)].append(player)
    game_state['pending_playoffs'] = []
    for score, players in scores_to_players.items():
        if len(players) > 1:
            game_state['pending_playoffs'].append({'score': score, 'players': players})
    game_state['pending_playoffs'].sort(key=lambda p: p['score'])
    start_next_playoff()

def start_next_playoff():
    if game_state['pending_playoffs']:
        next_playoff = game_state['pending_playoffs'].pop(0)
        game_state['phase'] = 'playoff'
        game_state['playoff_group'] = next_playoff['players']
        game_state['playoff_base_score'] = next_playoff['score']
        game_state['current_player_index'] = 0
        game_state['playoff_round'] = 1
        game_state['playoff_round_scores'] = {}
        game_state['playoff_history'] = []
    else:
        game_state['phase'] = 'final_ranking'
        if game_state['scores']:
            min_score = min(game_state['scores'].values())
            winners = [p for p, s in game_state['scores'].items() if s == min_score]
            game_state['winner'] = winners[0] if winners else None

def resolve_playoff_round():
    scores = game_state['playoff_round_scores']
    
    game_state['playoff_history'].append(scores.copy())
    for player, score in scores.items():
        if player not in game_state['all_playoff_history']:
            game_state['all_playoff_history'][player] = []
        game_state['all_playoff_history'][player].append(score)

    # If any tie remains (i.e., duplicate values), keep playoff going
    if len(set(scores.values())) < len(scores):
        game_state['playoff_round'] += 1
        game_state['current_player_index'] = 0
        game_state['playoff_round_scores'] = {}
        return
        
    # Lowest playoff score is better. Sort players by score ascending.
    sorted_players = sorted(scores.keys(), key=lambda p: scores[p])
    
    base_score = game_state['playoff_base_score']
    for i, player in enumerate(sorted_players):
        game_state['scores'][player] = base_score + (i + 1) * 0.01
        game_state['final_playoff_scores'][player] = scores[player]
    start_next_playoff()

@app.route('/undo', methods=['POST'])
def undo_last_move():
    if game_state['phase'] != 'playing' or not game_state['undo_history']:
        return redirect(url_for('index'))
    last_move = game_state['undo_history'].pop()
    prev_player_index = (game_state['current_player_index'] - 1 + len(game_state['players'])) % len(game_state['players'])
    game_state['current_player_index'] = prev_player_index
    if game_state['current_player_index'] == len(game_state['players']) - 1:
        game_state['current_round'] -= 1
    player_to_undo = game_state['players'][prev_player_index]
    game_state['scores'][player_to_undo] -= last_move['score_change']
    game_state['round_history'][game_state['current_round'] - 1].pop(player_to_undo, None)
    return redirect(url_for('index'))

@app.route('/restart', methods=['POST'])
def restart():
    reset_game()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
