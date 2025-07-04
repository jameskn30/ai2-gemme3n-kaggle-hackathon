import React from 'react';
import { TouchableOpacity, View, StyleSheet } from 'react-native';

const FAB = ({ onPress }: { onPress?: () => void }) => (
  <TouchableOpacity style={styles.fab} onPress={onPress} activeOpacity={0.7}>
    <View style={styles.plusContainer}>
      <View style={styles.horizontal} />
      <View style={styles.vertical} />
    </View>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  fab: {
    position: 'absolute',
    bottom: 30,
    zIndex: 10,
    width: 70,
    height: 70,
    borderRadius: 70,
    backgroundColor: '#f1f5f9', // slate-100
    justifyContent: 'center',
    alignItems: 'center',
    // Shadow for iOS
    shadowColor: 'gray',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.99,
    shadowRadius: 20,
    // Shadow for Android
    elevation: 6,
    marginBottom: 10,
  },
  plusContainer: {
    position: 'relative',
    width: 24,
    height: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  horizontal: {
    position: 'absolute',
    width: 18,
    height: 4,
    backgroundColor: '#1d4ed8', // blue-700
    borderRadius: 2,
    left: 3,
    top: 10,
  },
  vertical: {
    position: 'absolute',
    width: 4,
    height: 18,
    backgroundColor: '#1d4ed8', // blue-700
    borderRadius: 2,
    left: 10,
    top: 3,
  },
});

export default FAB;
