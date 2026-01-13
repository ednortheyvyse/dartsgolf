import React from 'react';
import { motion } from 'framer-motion';
import { Trophy, RefreshCw, Target, TrendingUp, TrendingDown, Table, Activity, Minus } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Button } from './ui/Button';
import { Player } from '../types';

interface ResultsScreenProps {
  players: Player[];
  onRestart: () => void;
}

export const ResultsScreen: React.FC<ResultsScreenProps> = ({ players, onRestart }) => {
  // Sort players by finalRank if available, otherwise total score
  const sortedPlayers = [...players].sort((a, b) => {
      if (a.finalRank !== undefined && b.finalRank !== undefined) {
          return a.finalRank - b.finalRank;
      }
      return a.totalScore - b.totalScore;
  });
  
  const winner = sortedPlayers[0];
  const totalRounds = players[0]?.scores.length || 0;
  const maxTiebreakerRounds = Math.max(0, ...players.map(p => p.tiebreakerScores.length));

  // Prepare data for chart
  const chartData = Array.from({ length: totalRounds }).map((_, i) => {
    const point: any = { round: i + 1 };
    players.forEach(p => {
      const cumScore = p.scores.slice(0, i + 1).reduce<number>((sum, s) => sum + s, 0);
      point[p.name] = cumScore;
    });
    return point;
  });

  const getPlayerStats = (p: Player) => {
    // Accuracy: Par or better (<= 0)
    const hits = p.scores.filter(s => s <= 0).length;
    const total = p.scores.length;
    const accuracy = total > 0 ? Math.round((hits / total) * 100) : 0;
    
    // Birdies: Strictly under par (< 0) -> -1, -2, -3
    const birdies = p.scores.filter(s => s < 0).length;

    // Pars: Exactly 0
    const pars = p.scores.filter(s => s === 0).length;
    
    // Bogeys: Strictly over par (> 0) -> +1, +2
    const bogeys = p.scores.filter(s => s > 0).length;

    return { accuracy, birdies, pars, bogeys };
  };

  return (
    <div className="flex flex-col h-full bg-black overflow-y-auto">
      {/* Top Header Section: Title & Winner */}
      <div className="p-8 text-center bg-neutral-900 border-b border-neutral-800 shadow-xl z-10 shrink-0">
        <h1 className="text-neutral-500 font-bold uppercase tracking-[0.3em] text-sm mb-6">Final Standings</h1>
        
        <motion.div 
            initial={{ scale: 0 }} 
            animate={{ scale: 1 }} 
            transition={{ type: "spring", stiffness: 200, damping: 10 }}
            className="inline-flex items-center justify-center p-6 bg-amber-400 rounded-full shadow-lg shadow-amber-900/40 mb-4"
        >
            <Trophy className="w-12 h-12 text-black" />
        </motion.div>
        
        <div className="text-xl font-bold text-amber-500 uppercase tracking-widest mb-1">Winner</div>
        <div className="text-5xl font-black text-white mb-2">{winner.name}</div>
        <div className="text-2xl text-neutral-400 font-bold">
            Score: <span className={winner.totalScore < 0 ? 'text-green-400' : 'text-white'}>{winner.totalScore > 0 ? '+' : ''}{winner.totalScore}</span>
        </div>
      </div>

      <div className="p-4 space-y-6 pb-20 max-w-4xl mx-auto w-full">
        
        {/* Detailed Standings & Stats */}
        <div className="space-y-3">
            {sortedPlayers.map((p, idx) => {
                const stats = getPlayerStats(p);
                return (
                    <motion.div 
                        key={p.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className="bg-neutral-900/80 rounded-2xl border border-neutral-800 overflow-hidden"
                    >
                        <div className="flex items-center justify-between p-4 bg-neutral-900 border-b border-neutral-800/50">
                            <div className="flex items-center gap-4">
                                <span className={`w-8 h-8 shrink-0 flex items-center justify-center rounded-full font-bold text-sm ${idx === 0 ? 'bg-amber-400 text-black' : idx === 1 ? 'bg-neutral-400 text-black' : idx === 2 ? 'bg-amber-700 text-amber-100' : 'bg-neutral-800 text-neutral-500 border border-neutral-700'}`}>
                                    {p.finalRank ?? idx + 1}
                                </span>
                                <div className="flex items-center gap-2">
                                    <div 
                                        className="w-6 h-6 shrink-0 rounded-full flex items-center justify-center text-[10px] font-black text-black"
                                        style={{ backgroundColor: p.color }}
                                    >
                                        {p.name.substring(0, 1).toUpperCase()}
                                    </div>
                                    <span className="font-bold text-lg text-white">{p.name}</span>
                                </div>
                            </div>
                            <span className={`text-2xl font-black ${p.totalScore < 0 ? 'text-green-400' : p.totalScore > 0 ? 'text-red-400' : 'text-neutral-400'}`}>
                                {p.totalScore > 0 ? '+' : ''}{p.totalScore}
                            </span>
                        </div>
                        
                        {/* Stats Grid */}
                        <div className="grid grid-cols-4 divide-x divide-neutral-800 bg-neutral-900/50">
                            <div className="p-3 flex flex-col items-center">
                                <div className="flex items-center gap-1 text-[10px] font-bold text-neutral-500 uppercase tracking-wide mb-1">
                                    <Target className="w-3 h-3" /> Acc
                                </div>
                                <span className={`text-lg font-bold ${stats.accuracy >= 50 ? 'text-green-400' : 'text-neutral-300'}`}>
                                    {stats.accuracy}%
                                </span>
                            </div>
                            <div className="p-3 flex flex-col items-center">
                                <div className="flex items-center gap-1 text-[10px] font-bold text-neutral-500 uppercase tracking-wide mb-1">
                                    <Minus className="w-3 h-3 text-neutral-400" /> Pars
                                </div>
                                <span className="text-lg font-bold text-neutral-300">
                                    {stats.pars}
                                </span>
                            </div>
                            <div className="p-3 flex flex-col items-center">
                                <div className="flex items-center gap-1 text-[10px] font-bold text-neutral-500 uppercase tracking-wide mb-1">
                                    <TrendingDown className="w-3 h-3 text-green-500" /> Birdies
                                </div>
                                <span className="text-lg font-bold text-green-400">
                                    {stats.birdies}
                                </span>
                            </div>
                            <div className="p-3 flex flex-col items-center">
                                <div className="flex items-center gap-1 text-[10px] font-bold text-neutral-500 uppercase tracking-wide mb-1">
                                    <TrendingUp className="w-3 h-3 text-red-500" /> Bogeys
                                </div>
                                <span className="text-lg font-bold text-red-400">
                                    {stats.bogeys}
                                </span>
                            </div>
                        </div>
                    </motion.div>
                );
            })}
        </div>

        {/* Chart */}
        <div className="bg-neutral-900 rounded-2xl border border-neutral-800 h-[450px] flex flex-col overflow-hidden">
            <div className="p-4 border-b border-neutral-800 flex items-center gap-2 shrink-0 bg-neutral-900">
                <Activity className="w-5 h-5 text-amber-500" />
                <h3 className="text-lg font-bold text-neutral-300">Score Timeline</h3>
            </div>
            <div className="flex-1 w-full p-4 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                        <XAxis dataKey="round" stroke="#525252" tick={{fill: '#a3a3a3'}} />
                        <YAxis stroke="#525252" tick={{fill: '#a3a3a3'}} />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#171717', border: '1px solid #404040', borderRadius: '8px' }} 
                            itemStyle={{ color: '#fff' }}
                        />
                        <Legend />
                        {players.map((p) => (
                            <Line 
                                key={p.id}
                                type="monotone" 
                                dataKey={p.name} 
                                stroke={p.color} 
                                strokeWidth={3}
                                dot={false}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>

        {/* Scorecard Table */}
        <div className="bg-neutral-900 rounded-2xl border border-neutral-800 overflow-hidden">
             <div className="p-4 border-b border-neutral-800 flex items-center gap-2">
                <Table className="w-5 h-5 text-amber-500" />
                <h3 className="text-lg font-bold text-neutral-300">Scorecard</h3>
             </div>
             <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse min-w-[300px]">
                    <thead>
                        <tr className="bg-neutral-950 text-neutral-500 border-b border-neutral-800">
                            <th className="py-3 pl-4 font-mono text-xs w-12 sticky left-0 bg-neutral-950 z-10 border-r border-neutral-800">#</th>
                            {players.map(p => (
                                <th key={p.id} className="py-3 px-2 text-center text-xs font-bold uppercase min-w-[60px]">
                                    <div className="flex flex-col items-center gap-1">
                                        <div className="w-2 h-2 shrink-0 rounded-full" style={{ backgroundColor: p.color }} />
                                        {p.name.substring(0, 3)}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {Array.from({ length: totalRounds }).map((_, i) => (
                            <tr key={i} className="border-b border-neutral-800/50 hover:bg-neutral-800/30 transition-colors">
                                <td className="py-2 pl-4 text-neutral-500 font-mono text-xs sticky left-0 bg-black/90 border-r border-neutral-800 z-10">{i + 1}</td>
                                {players.map(p => {
                                    // Calculate cumulative score up to round i
                                    const cumScore = p.scores.slice(0, i + 1).reduce<number>((sum, s) => sum + s, 0);
                                    let color = 'text-neutral-500';
                                    if (cumScore < 0) color = 'text-green-500 font-bold';
                                    if (cumScore > 0) color = 'text-red-400';
                                    
                                    return (
                                        <td key={p.id} className={`py-2 px-2 text-center text-sm ${color}`}>
                                            {cumScore > 0 ? `+${cumScore}` : cumScore}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                        {Array.from({ length: maxTiebreakerRounds }).map((_, i) => (
                            <tr key={`t-${i}`} className="border-b border-neutral-800/50 hover:bg-neutral-800/30 transition-colors bg-amber-900/10">
                                <td className="py-2 pl-4 text-amber-600 font-mono text-xs sticky left-0 bg-black/90 border-r border-neutral-800 z-10">T{i + 1}</td>
                                {players.map(p => {
                                    const score = p.tiebreakerScores[i];
                                    if (score === undefined) {
                                        return <td key={p.id} className="py-2 px-2 text-center text-sm text-neutral-800">-</td>;
                                    }
                                    let color = 'text-neutral-400';
                                    if (score < 0) color = 'text-green-400 font-bold';
                                    if (score > 0) color = 'text-red-400';
                                    
                                    return (
                                        <td key={p.id} className={`py-2 px-2 text-center text-sm ${color}`}>
                                            {score > 0 ? `+${score}` : score}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
             </div>
        </div>
        
        <Button 
            onClick={onRestart} 
            size="xl" 
            className="w-full border-neutral-800 shrink-0"
            variant="primary"
        >
            <RefreshCw className="w-6 h-6" /> Start New Game
        </Button>
      </div>
    </div>
  );
};