import React, { useState, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Undo2, Flag, Activity, X, Swords, Trophy } from 'lucide-react';
import { Button } from './ui/Button';
import { Player, ScoreValue, TiebreakerState } from '../types';

interface GameScreenProps {
  players: Player[];
  currentRound: number;
  maxRounds: number;
  currentPlayerIndex: number;
  isEndingPrematurely: boolean;
  tiebreaker?: TiebreakerState;
  onScore: (score: ScoreValue) => void;
  onUndo: () => void;
  onRequestEndGame: () => void;
  direction: number;
}

const scores: ScoreValue[] = [-1, 0, -2, 1, -3, 2];

const variants = {
  enter: (direction: number) => ({
    x: direction > 0 ? '100%' : '-100%',
    opacity: 0
  }),
  center: {
    zIndex: 1,
    x: 0,
    opacity: 1
  },
  exit: (direction: number) => ({
    zIndex: 0,
    x: direction < 0 ? '100%' : '-100%',
    opacity: 0
  })
};

const LeaderDisplay = ({ leaders }: { leaders: Player[] }) => {
    if (leaders.length === 0) return null;

    return (
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-10">
            <div className="text-amber-400 font-bold uppercase text-xs tracking-widest mb-1 flex items-center gap-1.5">
                <Trophy className="w-3.5 h-3.5" />
                Leader
            </div>
            <div className="flex items-center gap-2">
                {leaders.map(leader => (
                    <div key={leader.id} className="flex items-center gap-1.5 bg-green-900/50 border border-green-700/60 px-2 py-0.5 rounded-full text-sm font-medium text-white">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: leader.color }} />
                        <span>{leader.name}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export const GameScreen: React.FC<GameScreenProps> = ({
  players,
  currentRound,
  maxRounds,
  currentPlayerIndex,
  isEndingPrematurely,
  tiebreaker,
  onScore,
  onUndo,
  onRequestEndGame,
  direction,
}) => {
  const [showScoreboard, setShowScoreboard] = useState(false);
  const currentPlayer = players[currentPlayerIndex];

  // Scroll current player into view in the mini-list
  const playerListRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (playerListRef.current) {
        const activeEl = playerListRef.current.querySelector('[data-active="true"]');
        if (activeEl) {
            activeEl.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
        }
    }
  }, [currentPlayerIndex]);

  const isTiebreaker = !!tiebreaker;

  const leaders = useMemo(() => {
    if (isTiebreaker || players.length < 2) return [];

    const allScores = players.map(p => p.totalScore);
    const uniqueScores = [...new Set(allScores)];

    if (uniqueScores.length <= 1) {
        return [];
    }

    const minScore = Math.min(...allScores);
    return players.filter(p => p.totalScore === minScore);
  }, [players, isTiebreaker]);

  const leaderIds = useMemo(() => new Set(leaders.map(l => l.id)), [leaders]);


  return (
    <div className="flex flex-col h-full bg-black relative overflow-hidden">
      {/* Header Info */}
      <div className={`relative px-4 py-2 border-b flex items-center justify-between z-10 shadow-md transition-colors h-28 shrink-0 ${isTiebreaker ? 'bg-amber-950/40 border-amber-800' : 'bg-neutral-900 border-neutral-800'}`}>
        {/* Left: Round Count */}
        <div className="flex flex-col z-20 justify-center h-full shrink-0">
            <span className={`text-xs font-bold uppercase tracking-wider mb-1 ${isTiebreaker ? 'text-amber-500' : 'text-neutral-400'}`}>
                {isTiebreaker ? 'Tiebreaker' : 'Round'}
            </span>
            <div className={`text-6xl font-black leading-none tracking-tighter flex items-baseline whitespace-nowrap ${isTiebreaker ? 'text-amber-400' : 'text-amber-400'}`}>
                {isTiebreaker ? tiebreaker.round : currentRound}
                {!isTiebreaker && <span className="text-2xl text-neutral-600 font-bold ml-1">/{maxRounds}</span>}
            </div>
        </div>

        {/* Center: Tiebreaker Status or Leader */}
        {isTiebreaker ? (
             <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-10">
                <div className="text-amber-300 font-bold uppercase text-xs tracking-widest mb-0.5">
                    Fighting for Rank
                </div>
                <div className="text-3xl font-black text-white leading-none">
                    {tiebreaker.fightingForRank}
                </div>
                <div className="text-[10px] text-amber-500 font-bold flex items-center gap-1 mt-1">
                    <Swords className="w-3 h-3" /> Sudden Death
                </div>
             </div>
        ) : (
            <LeaderDisplay leaders={leaders} />
        )}
        
        {/* Warning Badge (Absolute near center-ish or right) */}
        {isEndingPrematurely && (
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-red-900/90 border border-red-500 text-red-200 px-4 py-2 rounded-full text-xs font-bold animate-pulse z-30 shadow-xl whitespace-nowrap pointer-events-none">
                LAST ROUND
            </div>
        )}

        {/* Right: Actions */}
        <div className="flex gap-2 z-20 h-full items-center shrink-0">
            <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => setShowScoreboard(true)}
                className="text-neutral-400 hover:bg-neutral-800 h-12 w-12 rounded-full border border-neutral-800 shrink-0"
            >
                <Activity className="w-6 h-6" />
            </Button>
        </div>
      </div>

      {/* Players Strip */}
      <div 
        ref={playerListRef}
        className="flex gap-3 overflow-x-auto p-4 bg-neutral-900/50 backdrop-blur-sm border-b border-neutral-800 snap-x shrink-0"
      >
        {players.map((p, idx) => {
            const isActive = idx === currentPlayerIndex;
            const isInTiebreaker = tiebreaker?.activePlayerIds.includes(p.id);
            const isRelevant = !isTiebreaker || isInTiebreaker;
            const isLeader = leaderIds.has(p.id);
            
            return (
                <div
                    key={p.id}
                    data-active={isActive}
                    className={`
                        snap-center shrink-0 min-w-[120px] p-3 rounded-xl border-2 transition-all duration-300 flex flex-col items-center gap-2
                        ${isActive 
                            ? 'bg-neutral-800 border-amber-400 scale-105 shadow-lg shadow-amber-900/10' 
                            : 'bg-neutral-900 border-neutral-800'
                        }
                        ${!isRelevant ? 'opacity-30 grayscale' : 'opacity-100'}
                    `}
                >
                    <div 
                        className="w-8 h-8 shrink-0 rounded-full flex items-center justify-center text-xs font-black text-black shadow-inner"
                        style={{ backgroundColor: p.color }}
                    >
                        {p.name.substring(0, 1).toUpperCase()}
                    </div>
                    <span className={`text-sm font-bold truncate max-w-full ${isActive ? 'text-white' : 'text-neutral-400'}`}>
                        {isLeader && <Trophy className="w-3.5 h-3.5 inline-block mr-1.5 text-amber-400" />}
                        {p.name}
                    </span>
                    <span className={`text-2xl font-black ${p.totalScore < 0 ? 'text-green-400' : p.totalScore > 0 ? 'text-red-400' : 'text-neutral-200'}`}>
                        {p.totalScore > 0 ? '+' : ''}{p.totalScore}
                    </span>
                </div>
            );
        })}
      </div>

      {/* Main Action Area */}
      <div className="flex-1 flex flex-col items-center justify-center p-4 relative z-0 overflow-y-auto overflow-x-hidden">
        <div className="text-center mb-8 shrink-0 w-full relative h-20">
            <div className="absolute top-0 left-0 right-0 text-neutral-500 text-sm mb-1 uppercase tracking-widest font-bold">
                {isTiebreaker ? 'Tiebreaker Throw' : 'Current Throw'}
            </div>
            <AnimatePresence mode="popLayout" custom={direction}>
                <motion.div 
                    key={currentPlayer.id}
                    custom={direction}
                    variants={variants}
                    initial="enter"
                    animate="center"
                    exit="exit"
                    transition={{ duration: 0.25, ease: "easeInOut" }}
                    className="flex items-center justify-center gap-3 absolute top-6 left-0 right-0"
                >
                    <div 
                        className="w-10 h-10 shrink-0 rounded-full flex items-center justify-center text-lg font-black text-black shadow-lg"
                        style={{ backgroundColor: currentPlayer.color }}
                    >
                        {currentPlayer.name.substring(0, 1).toUpperCase()}
                    </div>
                    <div className="text-5xl font-black text-white truncate max-w-[250px]">
                        {currentPlayer.name}
                    </div>
                </motion.div>
            </AnimatePresence>
        </div>

        {/* Scoring Grid */}
        <div className="grid grid-cols-2 gap-4 w-full max-w-sm shrink-0">
            {scores.map((score) => (
                <Button
                    key={score}
                    onClick={() => onScore(score)}
                    variant={score < 0 ? 'success' : score > 0 ? 'danger' : 'secondary'}
                    size="xl"
                    className={`h-20 text-3xl shadow-xl border-neutral-900 ${score === 0 ? 'text-black' : ''}`}
                    style={score === 0 ? { backgroundColor: '#F9DFBC' } : undefined}
                >
                    {score > 0 ? `+${score}` : score}
                </Button>
            ))}
        </div>
      </div>

      {/* Control Footer */}
      <div className="p-4 pb-8 bg-neutral-900 border-t border-neutral-800 flex gap-4 shrink-0">
        <Button 
            variant="secondary" 
            onClick={onUndo} 
            className="flex-1 bg-neutral-800 border-neutral-950 text-neutral-300"
        >
            <Undo2 className="w-5 h-5" /> Undo
        </Button>
        <Button 
            variant={isEndingPrematurely ? 'danger' : 'outline'}
            onClick={onRequestEndGame} 
            className={`flex-1 ${isEndingPrematurely ? '' : 'border-neutral-700 text-neutral-400 hover:bg-neutral-800'}`}
        >
            <Flag className="w-5 h-5" /> 
            {isEndingPrematurely ? 'Cancelling...' : isTiebreaker ? 'Skip Tiebreaker' : 'End Game'}
        </Button>
      </div>

      {/* Full Scoreboard Overlay */}
      <AnimatePresence>
        {showScoreboard && (
            <motion.div
                initial={{ opacity: 0, y: '100%' }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: '100%' }}
                className="absolute inset-0 z-50 bg-black flex flex-col"
            >
                <div className="p-4 border-b border-neutral-800 flex justify-between items-center bg-neutral-900 shrink-0">
                    <h2 className="text-xl font-bold text-white">Cumulative Scores</h2>
                    <Button variant="ghost" size="sm" onClick={() => setShowScoreboard(false)}>
                        <X className="w-6 h-6 text-neutral-400" />
                    </Button>
                </div>
                <div className="flex-1 overflow-auto p-4">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="text-neutral-500 border-b border-neutral-800">
                                <th className="py-2 pl-2">Rd</th>
                                {players.map(p => (
                                    <th key={p.id} className="py-2 px-2 text-center text-xs font-bold uppercase truncate max-w-[80px]">
                                        <div className="flex flex-col items-center gap-1">
                                            <div 
                                                className="w-4 h-4 rounded-full"
                                                style={{ backgroundColor: p.color }}
                                            />
                                            {p.name}
                                        </div>
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {Array.from({ length: currentRound }).map((_, i) => {
                                const roundNum = i + 1;
                                const isCurrent = roundNum === currentRound;
                                return (
                                    <tr key={roundNum} className={`border-b border-neutral-800/50 ${isCurrent ? 'bg-neutral-900' : ''}`}>
                                        <td className="py-3 pl-2 text-neutral-500 font-mono text-sm">{roundNum}</td>
                                        {players.map(p => {
                                            const roundScore = p.scores[i];
                                            const hasPlayedRound = roundScore !== undefined;

                                            const cumulativeScore = hasPlayedRound 
                                                ? p.scores.slice(0, i + 1).reduce<number>((sum, s) => sum + s, 0)
                                                : null;

                                            let color = 'text-neutral-400';
                                            if (cumulativeScore !== null) {
                                                if (cumulativeScore < 0) color = 'text-green-400 font-bold';
                                                if (cumulativeScore > 0) color = 'text-red-400';
                                            }
                                            
                                            return (
                                                <td key={p.id} className={`py-3 px-2 text-center ${color}`}>
                                                    {cumulativeScore !== null ? (cumulativeScore > 0 ? `+${cumulativeScore}` : cumulativeScore) : '-'}
                                                </td>
                                            )
                                        })}
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                    
                    {/* Tiebreaker Section in Scoreboard */}
                    {players.some(p => p.tiebreakerScores.length > 0) && (
                        <div className="mt-8">
                             <h3 className="text-amber-500 font-bold text-sm uppercase mb-2">Tiebreaker Shots</h3>
                             <table className="w-full text-left border-collapse">
                                <tbody>
                                    {/* Find max tiebreaker rounds played */}
                                    {Array.from({ length: Math.max(...players.map(p => p.tiebreakerScores.length)) }).map((_, i) => (
                                        <tr key={i} className="border-b border-neutral-800/50">
                                            <td className="py-3 pl-2 text-amber-700 font-mono text-sm">T{i + 1}</td>
                                            {players.map(p => {
                                                const score = p.tiebreakerScores[i];
                                                if (score === undefined) return <td key={p.id}></td>;
                                                return (
                                                    <td key={p.id} className="py-3 px-2 text-center text-neutral-300">
                                                        {score > 0 ? `+${score}` : score}
                                                    </td>
                                                );
                                            })}
                                        </tr>
                                    ))}
                                </tbody>
                             </table>
                        </div>
                    )}
                </div>
            </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};