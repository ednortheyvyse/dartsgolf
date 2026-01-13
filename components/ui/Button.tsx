import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface ButtonProps extends HTMLMotionProps<"button"> {
  variant?: 'primary' | 'secondary' | 'danger' | 'success' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

export const Button: React.FC<ButtonProps> = ({ 
  className, 
  variant = 'primary', 
  size = 'md', 
  children, 
  ...props 
}) => {
  const variants = {
    primary: 'bg-green-700 text-white shadow-lg shadow-green-900/50 border-b-4 border-green-900 active:border-b-0 active:translate-y-1',
    secondary: 'bg-slate-700 text-white shadow-lg shadow-slate-900/50 border-b-4 border-slate-900 active:border-b-0 active:translate-y-1',
    danger: 'bg-red-700 text-white shadow-lg shadow-red-900/50 border-b-4 border-red-900 active:border-b-0 active:translate-y-1',
    success: 'bg-emerald-600 text-white shadow-lg shadow-emerald-900/50 border-b-4 border-emerald-900 active:border-b-0 active:translate-y-1',
    outline: 'bg-transparent border-2 border-slate-500 text-slate-300 active:bg-slate-800',
    ghost: 'bg-transparent text-slate-400 hover:text-white active:scale-95'
  };

  const sizes = {
    sm: 'px-3 py-1 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg font-bold',
    xl: 'px-8 py-4 text-xl font-bold'
  };

  return (
    <motion.button
      whileTap={{ scale: 0.95 }}
      className={cn(
        'rounded-xl font-medium transition-colors flex items-center justify-center gap-2 select-none touch-manipulation',
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </motion.button>
  );
};