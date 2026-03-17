import React from 'react';
import { motion } from 'framer-motion';
import { Check } from 'lucide-react';
import { PLAYER_COLORS } from '../../types';

interface ColorSwatchProps {
  selectedColor: string;
  onSelect: (color: string) => void;
}

export const ColorSwatch: React.FC<ColorSwatchProps> = ({ selectedColor, onSelect }) => {
  return (
    <div className="grid grid-cols-5 gap-3 p-4 bg-neutral-800 rounded-lg">
      {PLAYER_COLORS.map((color) => (
        <motion.div
          key={color}
          onClick={() => onSelect(color)}
          className="w-10 h-10 rounded-full cursor-pointer flex items-center justify-center border-2 border-transparent"
          style={{ backgroundColor: color }}
          whileHover={{ scale: 1.1 }}
          animate={{ borderColor: selectedColor === color ? '#ffffff' : 'transparent' }}
        >
          {selectedColor === color && <Check className="w-6 h-6 text-white" />}
        </motion.div>
      ))}
    </div>
  );
};