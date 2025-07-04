import React, { useRef, useEffect } from 'react';
import { Animated, ViewProps, Text, StyleSheet, View } from 'react-native';
import Divider from './Divider';

type CardViewProps = ViewProps & {
  title: string;
  content: string;
};

const ThreeDots = () => {
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', marginLeft: 8 }}>
      <View
        style={{
          width: 4,
          height: 4,
          borderRadius: 2,
          backgroundColor: '#888',
          marginHorizontal: 1,
        }}
      />
      <View
        style={{
          width: 4,
          height: 4,
          borderRadius: 2,
          backgroundColor: '#888',
          marginHorizontal: 1,
        }}
      />
      <View
        style={{
          width: 4,
          height: 4,
          borderRadius: 2,
          backgroundColor: '#888',
          marginHorizontal: 1,
        }}
      />
    </View>
  );
};

const CardView: React.FC<CardViewProps> = ({
  title,
  content,
  style,
  ...rest
}) => {
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const opacityAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.spring(scaleAnim, {
        toValue: 1,
        useNativeDriver: true,
        friction: 7,
        tension: 60,
      }),
      Animated.timing(opacityAnim, {
        toValue: 1,
        duration: 250,
        useNativeDriver: true,
      }),
    ]).start();
  }, [scaleAnim, opacityAnim]);

  return (
    <Animated.View
      style={[
        styles.card,
        style,
        {
          transform: [{ scale: scaleAnim }],
          opacity: opacityAnim,
        },
      ]}
      {...rest}
    >
      <View>
        <Text style={styles.title}>{title}</Text>
        <Text
          style={[styles.content, { maxHeight: 150 }]}
          numberOfLines={8}
          ellipsizeMode="tail"
        >
          {content}
        </Text>
      </View>
      <Divider />
      <View
        style={{
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginTop: 12,
          paddingHorizontal: 8,
        }}
      >
        <Text style={{ fontSize: 12, color: '#888' }}>
          {new Date().toLocaleDateString()}
        </Text>
        <ThreeDots />
      </View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  title: {
    fontWeight: 'bold',
    fontSize: 18,
    marginBottom: 4,
    paddingHorizontal: 8,
  },
  content: {
    fontSize: 16,
    color: '#222',
    paddingHorizontal: 8,
    paddingVertical: 6,
    marginBottom: 6,
  },
  card: {
    marginHorizontal: 12,
    marginVertical: 8,
    backgroundColor: 'white',
    borderRadius: 12,
    // iOS shadow
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.12,
    shadowRadius: 6,
    // Android shadow
    elevation: 4,
    padding: 12,
  },
});

export default CardView;
