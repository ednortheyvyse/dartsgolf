import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Check } from 'lucide-react';
import { PLAYER_COLORS, PLAYER_ICONS } from '../../types';
import { PlayerIcon } from './PlayerIcon';
import { Button } from './Button';

interface PlayerEditModalProps {
  isOpen: boolean;
  initialName: string;
  initialColor: string;
  initialIcon: string;
  onSave: (name: string, color: string, icon: string) => void;
  onClose: () => void;
}

export const PlayerEditModal: React.FC<PlayerEditModalProps> = ({
  isOpen,
  initialName,
  initialColor,
  initialIcon,
  onSave,
  onClose
}) => {
  const [name, setName] = useState(initialName);
  const [color, setColor] = useState(initialColor);
  const [icon, setIcon] = useState(initialIcon);

  // Reset state when opened
  React.useEffect(() => {
    if (isOpen) {
      setName(initialName);
      setColor(initialColor);
      setIcon(initialIcon);
    }
  }, [isOpen, initialName, initialColor, initialIcon]);

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="bg-neutral-900 border border-neutral-800 rounded-2xl w-full max-w-md max-h-[90vh] flex flex-col overflow-hidden shadow-2xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-3 border-b border-neutral-800 shrink-0">
              <h2 className="text-lg font-bold text-white">Edit Player</h2>
              <button 
                onClick={onClose}
                className="p-1.5 rounded-full text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="p-4 space-y-5 flex-1">
              {/* Name */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-neutral-400 uppercase tracking-wider">Player Name</label>
                <input 
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-neutral-800 border-2 border-neutral-700 rounded-xl px-3 py-2 text-white focus:border-amber-400 focus:outline-none transition-colors"
                  placeholder="Enter name..."
                />
              </div>

              {/* Color */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-neutral-400 uppercase tracking-wider">Color</label>
                <div className="grid grid-cols-5 gap-2">
                  {PLAYER_COLORS.map(c => (
                    <button
                      key={c}
                      onClick={() => setColor(c)}
                      className={`w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center transition-transform hover:scale-110 ${color === c ? 'ring-2 ring-white ring-offset-2 ring-offset-neutral-900' : ''}`}
                      style={{ backgroundColor: c }}
                    >
                      {color === c && <Check className="w-4 h-4 sm:w-5 sm:h-5 text-white" />}
                    </button>
                  ))}
                </div>
              </div>

              {/* Icon */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-neutral-400 uppercase tracking-wider">Avatar</label>
                <div className="grid grid-cols-5 gap-2">
                  {PLAYER_ICONS.map(i => (
                    <button
                      key={i}
                      onClick={() => setIcon(i)}
                      className={`w-10 h-10 sm:w-12 sm:h-12 rounded-xl flex items-center justify-center transition-all shadow-inner ${icon === i ? 'ring-2 ring-white scale-110' : 'opacity-70 hover:opacity-100 hover:scale-105'}`}
                      style={{ backgroundColor: color }}
                    >
                      <PlayerIcon name={i} color="#000" size={1} />
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="p-3 border-t border-neutral-800 bg-neutral-950 shrink-0 flex gap-3">
              <Button variant="secondary" onClick={onClose} className="flex-1">
                Cancel
              </Button>
              <Button 
                variant="primary" 
                onClick={() => {
                  if (name.trim()) {
                    onSave(name.trim(), color, icon);
                  }
                }}
                disabled={!name.trim()}
                className="flex-1"
              >
                Save Changes
              </Button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};