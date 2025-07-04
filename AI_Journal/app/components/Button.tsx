import React from 'react';
import { Pressable, Text, StyleSheet, ViewStyle, View } from 'react-native';

type ButtonProps = {
  title: string;
  onPress: () => void;
  style?: ViewStyle;
};

const Button: React.FC<ButtonProps> = ({ title, onPress, style }) => {
  return (
    <View>
      <Pressable
        onPress={onPress}
        style={({ pressed }) => [
          styles.button,
          style,
          pressed && styles.pressed,
        ]}
      >
        <Text style={styles.text}>{title}</Text>
      </Pressable>
    </View>
  );
};

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pressed: {
    opacity: 0.7,
  },
  text: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default Button;
