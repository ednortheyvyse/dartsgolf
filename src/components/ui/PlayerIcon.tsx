import React from 'react';
import Icon from '@mdi/react';
import * as mdiIcons from '@mdi/js';

interface PlayerIconProps {
  name: string;
  className?: string;
  color?: string;
  size?: number | string;
}

export const PlayerIcon: React.FC<PlayerIconProps> = ({ name, className, color, size }) => {
  const iconPath = (mdiIcons as Record<string, string>)[name];
  
  if (!iconPath) {
    const fallbackPath = mdiIcons.mdiAccount;
    return <Icon path={fallbackPath} className={className} color={color} size={size ?? 1} />;
  }

  return <Icon path={iconPath} className={className} color={color} size={size ?? 1} />;
};
