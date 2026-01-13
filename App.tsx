import { useState } from 'react';
import { StartScreen } from './components/StartScreen';
import { GameScreen } from './components/GameScreen';
import { ResultsScreen } from './components/ResultsScreen';
import { GameState, Player, ScoreValue, PLAYER_COLORS } from './types';

const generateId = () => Math.random().toString(36).substr(2, 9);

const INITIAL_STATE: GameState = {
  status: 'SETUP',
  players: [],
  currentRound: 1,
  currentPlayerIndex: 0,
  isEndingPrematurely: false,
  maxRounds: 20,
};

export default function App() {
  const [gameState, setGameState] = useState<GameState>(INITIAL_STATE);
  const [direction, setDirection] = useState(1); // 1 for forward (score), -1 for backward (undo)
  
  // History for Undo functionality
  const [history, setHistory] = useState<GameState[]>([]);

  const handleStartGame = (playerNames: string[]) => {
    const newPlayers: Player[] = playerNames.map((name, index) => ({
      id: generateId(),
      name,
      color: PLAYER_COLORS[index % PLAYER_COLORS.length],
      scores: [],
      totalScore: 0,
      tiebreakerScores: []
    }));

    const newState: GameState = {
      ...INITIAL_STATE,
      status: 'PLAYING',
      players: newPlayers
    };

    setDirection(1);
    setHistory([newState]);
    setGameState(newState);
  };

  const getNextPlayerIndex = (players: Player[], currentIndex: number, activeIds?: string[]) => {
    let nextIndex = currentIndex + 1;
    if (activeIds) {
      let count = 0;
      while (count < players.length * 2) {
        if (nextIndex >= players.length) nextIndex = 0;
        if (activeIds.includes(players[nextIndex].id)) return nextIndex;
        nextIndex++;
        count++;
      }
      return 0;
    }
    if (nextIndex >= players.length) nextIndex = 0;
    return nextIndex;
  };

  const resolveGameStatus = (state: GameState): GameState => {
    const { players } = state;
    
    // Sort logic: Total Score + Tiebreaker Penalty
    const getSortScore = (p: Player) => p.totalScore + (p.tiebreakerPenalty || 0);

    const unrankedPlayers = players.filter(p => p.finalRank === undefined);

    if (unrankedPlayers.length === 0) {
      return { ...state, status: 'FINISHED' };
    }

    const sortedUnranked = [...unrankedPlayers].sort((a, b) => getSortScore(a) - getSortScore(b));

    const bestScore = getSortScore(sortedUnranked[0]);
    const tiedGroup = sortedUnranked.filter(p => getSortScore(p) === bestScore);

    const rankedCount = players.filter(p => p.finalRank !== undefined).length;
    const nextRank = rankedCount + 1;

    if (tiedGroup.length === 1) {
      const winner = tiedGroup[0];
      const updatedPlayers = players.map(p => 
        p.id === winner.id ? { ...p, finalRank: nextRank } : p
      );
      
      return resolveGameStatus({
        ...state,
        players: updatedPlayers
      });
    }

    const activeIds = tiedGroup.map(p => p.id);
    const firstPlayerIndex = players.findIndex(p => p.id === activeIds[0]);

    return {
      ...state,
      status: 'TIEBREAKER',
      currentPlayerIndex: firstPlayerIndex,
      tiebreaker: {
        activePlayerIds: activeIds,
        fightingForRank: nextRank,
        round: 1,
        roundScores: {}
      }
    };
  };

  const handleScore = (score: ScoreValue) => {
    setDirection(1);
    const currentState = gameState;
    setHistory(prev => [...prev.slice(-49), currentState]);

    if (currentState.status === 'TIEBREAKER' && currentState.tiebreaker) {
        handleTiebreakerScore(currentState, score);
        return;
    }

    const { players, currentPlayerIndex, currentRound, maxRounds, isEndingPrematurely } = currentState;
    const nextPlayers = players.map(p => ({ ...p, scores: [...p.scores] }));
    const currentPlayer = nextPlayers[currentPlayerIndex];

    currentPlayer.scores.push(score);
    currentPlayer.totalScore += score;

    let nextPlayerIndex = currentPlayerIndex + 1;
    let nextRound = currentRound;
    
    if (nextPlayerIndex >= players.length) {
      nextPlayerIndex = 0;
      nextRound += 1;
      
      if (nextRound > maxRounds || isEndingPrematurely) {
        const stateToResolve = {
            ...currentState,
            players: nextPlayers,
            status: 'PLAYING' as const
        };
        const resolvedState = resolveGameStatus(stateToResolve);
        setGameState(resolvedState);
        return;
      }
    }

    setGameState({
      ...currentState,
      players: nextPlayers,
      currentPlayerIndex: nextPlayerIndex,
      currentRound: nextRound,
    });
  };

  const handleTiebreakerScore = (currentState: GameState, score: ScoreValue) => {
    const { players, currentPlayerIndex, tiebreaker } = currentState;
    if (!tiebreaker) return;

    const nextPlayers = players.map(p => ({ 
        ...p, 
        tiebreakerScores: [...p.tiebreakerScores] 
    }));
    
    const currentPlayer = nextPlayers[currentPlayerIndex];
    currentPlayer.tiebreakerScores.push(score);

    const nextRoundScores = { ...tiebreaker.roundScores, [currentPlayer.id]: score };
    const activeIds = tiebreaker.activePlayerIds;
    
    const allPlayed = activeIds.every(id => 
        (id === currentPlayer.id) || (nextRoundScores[id] !== undefined)
    );

    if (!allPlayed) {
        const nextIndex = getNextPlayerIndex(nextPlayers, currentPlayerIndex, activeIds);
        setGameState({
            ...currentState,
            players: nextPlayers,
            currentPlayerIndex: nextIndex,
            tiebreaker: {
                ...tiebreaker,
                roundScores: nextRoundScores
            }
        });
        return;
    }

    // Round Complete
    const groups: Record<number, Player[]> = {};
    activeIds.forEach(id => {
        const s = nextRoundScores[id];
        if (!groups[s]) groups[s] = [];
        groups[s].push(nextPlayers.find(p => p.id === id)!);
    });
    
    const sortedScores = Object.keys(groups).map(Number).sort((a, b) => a - b);

    if (sortedScores.length === 1) {
         // Tie continues
         const nextStartIdx = getNextPlayerIndex(nextPlayers, currentPlayerIndex, activeIds);
         setGameState({
             ...currentState,
             players: nextPlayers,
             currentPlayerIndex: nextStartIdx,
             tiebreaker: {
                 ...tiebreaker,
                 round: tiebreaker.round + 1,
                 roundScores: {}
             }
         });
         return;
    } 

    // Split occurred - apply penalties to losers
    sortedScores.forEach((score, idx) => {
        if (idx > 0) {
            groups[score].forEach(p => {
                 const target = nextPlayers.find(np => np.id === p.id);
                 if (target) {
                     // Add penalty based on round score rank
                     target.tiebreakerPenalty = (target.tiebreakerPenalty || 0) + (idx * 0.1);
                 }
            });
        }
    });

    const stateForResolution = {
        ...currentState,
        players: nextPlayers,
        status: 'PLAYING' as const, 
        tiebreaker: undefined
    };
    
    const resolvedState = resolveGameStatus(stateForResolution);
    setGameState(resolvedState);
  };

  const handleUndo = () => {
    if (history.length > 0) {
      setDirection(-1);
      const previousState = history[history.length - 1];
      setGameState(previousState);
      setHistory(prev => prev.slice(0, -1));
    }
  };

  const handleRequestEndGame = () => {
    if (gameState.status === 'TIEBREAKER') {
        setGameState(prev => ({ ...prev, status: 'FINISHED' }));
    } else {
        setGameState(prev => ({
            ...prev,
            isEndingPrematurely: !prev.isEndingPrematurely
        }));
    }
  };

  const handleRestart = () => {
    setDirection(1);
    setGameState(INITIAL_STATE);
    setHistory([]);
  };

  return (
    <div className="h-full w-full bg-black text-slate-50 font-sans selection:bg-amber-500/30 overflow-hidden">
      {gameState.status === 'SETUP' && (
        <StartScreen onStartGame={handleStartGame} />
      )}
      {(gameState.status === 'PLAYING' || gameState.status === 'TIEBREAKER') && (
        <GameScreen
          players={gameState.players}
          currentRound={gameState.currentRound}
          maxRounds={gameState.maxRounds}
          currentPlayerIndex={gameState.currentPlayerIndex}
          isEndingPrematurely={gameState.isEndingPrematurely}
          onScore={handleScore}
          onUndo={handleUndo}
          onRequestEndGame={handleRequestEndGame}
          tiebreaker={gameState.tiebreaker}
          direction={direction}
        />
      )}
      {gameState.status === 'FINISHED' && (
        <ResultsScreen 
          players={gameState.players} 
          onRestart={handleRestart} 
        />
      )}
    </div>
  );
}