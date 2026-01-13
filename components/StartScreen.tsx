import React, { useState } from 'react';
import { Reorder, AnimatePresence, motion } from 'framer-motion';
import { Plus, GripVertical, X, Play, Trophy } from 'lucide-react';
import { Button } from './ui/Button';
import { PLAYER_COLORS } from '../types';

interface StartScreenProps {
  onStartGame: (players: string[]) => void;
}

export const StartScreen: React.FC<StartScreenProps> = ({ onStartGame }) => {
  const [names, setNames] = useState<string[]>([]);
  const [inputValue, setInputValue] = useState('');

  const addPlayer = () => {
    if (inputValue.trim()) {
      setNames([...names, inputValue.trim()]);
      setInputValue('');
    }
  };

  const removePlayer = (index: number) => {
    setNames(names.filter((_, i) => i !== index));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      addPlayer();
    }
  };

  return (
    <div className="flex flex-col h-full max-w-md mx-auto p-6 w-full bg-black">
      <div className="text-center mb-8 shrink-0">
        <div className="inline-block p-4 rounded-full bg-neutral-900 mb-4 border-4 border-red-700 shadow-lg shadow-red-900/20">
            <Trophy className="w-12 h-12 text-amber-400" />
        </div>
        <h1 className="text-4xl font-black text-white tracking-tight">
          <span className="text-red-600">DARTS</span> <span className="text-green-600">GOLF</span>
        </h1>
        <p className="text-neutral-400 mt-2">Setup Players</p>
      </div>

      <div className="flex gap-2 mb-6 shrink-0">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter player name"
          className="flex-1 bg-neutral-900 border-2 border-neutral-800 rounded-xl px-4 py-3 text-white placeholder-neutral-500 focus:border-amber-400 focus:outline-none transition-colors"
        />
        <Button onClick={addPlayer} disabled={!inputValue.trim()} size="md" className="shrink-0">
          <Plus className="w-6 h-6" />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0 mb-6 pr-2">
        <Reorder.Group axis="y" values={names} onReorder={setNames} className="space-y-3">
          <AnimatePresence>
            {names.map((name, index) => (
              <Reorder.Item
                key={name}
                value={name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="bg-neutral-900 rounded-xl p-3 pl-4 flex items-center justify-between group shadow-md border border-neutral-800 select-none touch-none"
              >
                <div className="flex items-center gap-3">
                  <GripVertical className="text-neutral-600 cursor-grab active:cursor-grabbing w-6 h-6 shrink-0" />
                  <div 
                      className="w-8 h-8 shrink-0 rounded-full flex items-center justify-center text-xs font-black text-black shadow-inner"
                      style={{ backgroundColor: PLAYER_COLORS[index % PLAYER_COLORS.length] }}
                  >
                      {name.substring(0, 1).toUpperCase()}
                  </div>
                  <span className="font-bold text-lg text-white truncate">{name}</span>
                </div>
                <button 
                    onClick={() => removePlayer(names.indexOf(name))}
                    className="p-2 text-neutral-500 hover:text-red-400 transition-colors shrink-0"
                >
                  <X className="w-5 h-5" />
                </button>
              </Reorder.Item>
            ))}
          </AnimatePresence>
        </Reorder.Group>
        
        {names.length === 0 && (
            <div className="text-center py-10 text-neutral-600 italic">
                Add players to begin...
            </div>
        )}
      </div>

      <Button
        onClick={() => onStartGame(names)}
        disabled={names.length === 0}
        size="xl"
        variant="primary"
        className="w-full border-neutral-800 shrink-0"
      >
        <Play className="w-6 h-6 fill-current" />
        START GAME
      </Button>
    </div>
  );
};