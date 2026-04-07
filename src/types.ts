export type ScoreValue = -3 | -2 | -1 | 0 | 1 | 2;

export interface Player {
  id: string;
  name: string;
  color: string; // Hex color code
  scores: ScoreValue[]; // Index corresponds to round - 1
  totalScore: number;
  // Tiebreaker specific
  tiebreakerScores: ScoreValue[];
  tiebreakerPenalty?: number; // Used for internal sorting during tiebreaks
  finalRank?: number; // 1-based rank
}

export type GameStatus = 'SETUP' | 'PLAYING' | 'TIEBREAKER' | 'FINISHED';

export interface TiebreakerState {
  activePlayerIds: string[]; // IDs of players in the current tiebreaker
  fightingForRank: number; // The best rank they can achieve
  round: number; // Current tiebreaker round number (1-based)
  roundScores: Record<string, ScoreValue>; // Scores for the current tiebreaker round
}

export interface GameState {
  status: GameStatus;
  players: Player[];
  currentRound: number; // 1-20
  currentPlayerIndex: number;
  isEndingPrematurely: boolean; // If true, game ends after current round finishes
  maxRounds: number;
  tiebreaker?: TiebreakerState;
}

export const PLAYER_COLORS = [
  '#ef4444', // Red
  '#f97316', // Orange
  '#d97706', // Gold
  '#4d7c0f', // Olive
  '#06b6d4', // Cyan
  '#f59e0b', // Amber
  '#eab308', // Yellow
  '#22c55e', // Green
  '#0ea5e9', // Sky
  '#64748b', // Slate
  '#84cc16', // Lime
  '#10b981', // Emerald
  '#3b82f6', // Blue
  '#8b5cf6', // Violet
  '#d946ef', // Fuchsia
  '#14b8a6', // Teal
  '#6366f1', // Indigo
  '#a855f7', // Purple
  '#ec4899', // Pink
  '#f43f5e', // Rose
];