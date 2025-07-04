import React, { useEffect, useState } from 'react';
import {
  ViewProps,
  Text,
  StyleSheet,
  View,
  TouchableOpacity,
} from 'react-native';
// Animation imports from react-native-reanimated
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  interpolate,
} from 'react-native-reanimated';
import Divider from './Divider';
import { Ionicons } from '@expo/vector-icons';

type CardViewProps = ViewProps & {
  title: string;
  content: string;
};

const CardView: React.FC<CardViewProps> = ({
  title,
  content,
  style,
  ...rest
}: CardViewProps) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Animation shared values for controlling animations
  const scaleAnim = useSharedValue(0.8); // Controls card scale animation
  const expandAnim = useSharedValue(0); // Controls content expansion animation

  useEffect(() => {
    // Initial animation when component mounts
    scaleAnim.value = withSpring(1, {
      damping: 7,
      stiffness: 60,
    });
  }, []);

  const handlePress = () => {
    const newExpandedState = !isExpanded;
    setIsExpanded(newExpandedState);

    if (newExpandedState) {
      // Use spring animation for expand (with bounce)
      expandAnim.value = withSpring(1, {
        damping: 7,
        stiffness: 60,
      });
    } else {
      // Use timing animation for collapse (no bounce) with faster duration
      expandAnim.value = withTiming(0, {
        duration: 150,
      });
    }
  };

  // Animated style for card scale
  const animatedCardStyle = useAnimatedStyle(() => {
    return {
      transform: [{ scale: scaleAnim.value }],
    };
  });

  // Animated style for content expansion
  const animatedContentStyle = useAnimatedStyle(() => {
    return {
      maxHeight: interpolate(expandAnim.value, [0, 1], [150, 1000]),
      overflow: 'hidden',
    };
  });

  return (
    <TouchableOpacity onPress={handlePress} activeOpacity={0.9}>
      <Animated.View
        style={[
          styles.card,
          style,
          animatedCardStyle, // Apply animated card style
        ]}
        {...rest}
      >
        <View>
          <Text style={styles.titleText}>{title}</Text>
          <Animated.View style={animatedContentStyle}>
            {/* Apply animated content style */}
            <Text
              style={styles.content}
              numberOfLines={isExpanded ? undefined : 8}
              ellipsizeMode="tail"
            >
              {content}
            </Text>
          </Animated.View>
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
          <Ionicons name="ellipsis-horizontal" size={16} color="#888" />
        </View>
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  titleText: {
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
