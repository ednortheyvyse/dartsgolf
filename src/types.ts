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
  '#22c55e', // Green
  '#3b82f6', // Blue
  '#eab308', // Yellow
  '#a855f7', // Purple
  '#ec4899', // Pink
  '#f97316', // Orange
  '#14b8a6', // Teal
  '#6366f1', // Indigo
  '#84cc16', // Lime
];