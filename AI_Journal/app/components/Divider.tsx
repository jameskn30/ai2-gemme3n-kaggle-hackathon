import React from 'react';
import { View } from 'react-native';

type DividerProps = {
  mode?: 'horizontal' | 'vertical';
};

const Divider = ({ mode = 'horizontal' }: DividerProps) => (
  <View
    style={{
      height: mode === 'horizontal' ? 1 : 'auto',
      width: mode === 'vertical' ? 1 : 'auto',
      backgroundColor: '#e5e7eb',
    }}
  />
);

export default Divider;
