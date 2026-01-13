import React, { useState, useEffect } from 'react';
import { Reorder, AnimatePresence, motion } from 'framer-motion';
import { Plus, GripVertical, X, Play, Trophy, Download, Share, PlusSquare } from 'lucide-react';
import { Button } from './ui/Button';
import { PLAYER_COLORS } from '../types';

interface StartScreenProps {
  onStartGame: (players: string[]) => void;
}

export const StartScreen: React.FC<StartScreenProps> = ({ onStartGame }) => {
  const [names, setNames] = useState<string[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null);
  const [isIOS, setIsIOS] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [showIOSInstructions, setShowIOSInstructions] = useState(false);
  const [isStandalone, setIsStandalone] = useState(false);

  useEffect(() => {
    const ua = navigator.userAgent;
    const mobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua);
    const ios = /iPad|iPhone|iPod/.test(ua) && !(window as any).MSStream;
    
    setIsMobile(mobile);
    setIsIOS(ios);
    
    const checkStandalone = window.matchMedia('(display-mode: standalone)').matches || (window.navigator as any).standalone === true;
    setIsStandalone(checkStandalone);

    const handler = (e: any) => {
      e.preventDefault();
      setDeferredPrompt(e);
    };

    window.addEventListener('beforeinstallprompt', handler);

    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

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

  const handleInstallClick = () => {
    if (isIOS) {
      setShowIOSInstructions(true);
    } else if (deferredPrompt) {
      deferredPrompt.prompt();
      deferredPrompt.userChoice.then((choiceResult: any) => {
        if (choiceResult.outcome === 'accepted') {
          setDeferredPrompt(null);
        }
      });
    }
  };

  const showInstallButton = isMobile && !isStandalone && (isIOS || deferredPrompt);

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

      {showInstallButton && (
        <Button
          onClick={handleInstallClick}
          variant="outline"
          size="lg"
          className="w-full border-neutral-800 text-neutral-400 mt-4 shrink-0"
        >
          <Download className="w-5 h-5 mr-2" />
          Install App
        </Button>
      )}

      <AnimatePresence>
        {showIOSInstructions && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-end sm:items-center justify-center p-4"
            onClick={() => setShowIOSInstructions(false)}
          >
            <motion.div
              initial={{ y: '100%' }}
              animate={{ y: 0 }}
              exit={{ y: '100%' }}
              className="bg-neutral-900 w-full max-w-sm rounded-2xl border border-neutral-800 p-6 shadow-2xl"
              onClick={e => e.stopPropagation()}
            >
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-xl font-bold text-white">Install App</h3>
                <button onClick={() => setShowIOSInstructions(false)} className="text-neutral-400 hover:text-white">
                  <X className="w-6 h-6" />
                </button>
              </div>
              
              <div className="space-y-4 text-neutral-300">
                <p>Install this app on your iPhone for the best experience:</p>
                <div className="flex items-center gap-3">
                  <Share className="w-6 h-6 text-blue-500" />
                  <span>1. Tap the <strong>Share</strong> button in the toolbar.</span>
                </div>
                <div className="flex items-center gap-3">
                  <PlusSquare className="w-6 h-6 text-neutral-400" />
                  <span>2. Scroll down and tap <strong>Add to Home Screen</strong>.</span>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};