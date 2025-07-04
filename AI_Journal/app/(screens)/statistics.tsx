import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { router } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Animated, { useSharedValue, useAnimatedStyle, withSpring } from 'react-native-reanimated';

// Statistics Dashboard Component
const StatisticsDashboard: React.FC = () => {
  // Animation values
  const fadeAnim = useSharedValue(0);
  const scaleAnim = useSharedValue(0.8);

  React.useEffect(() => {
    fadeAnim.value = withSpring(1, { damping: 10, stiffness: 100 });
    scaleAnim.value = withSpring(1, { damping: 10, stiffness: 100 });
  }, []);

  const fadeStyle = useAnimatedStyle(() => ({
    opacity: fadeAnim.value,
  }));

  const scaleStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scaleAnim.value }],
  }));

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <Animated.View style={[styles.header, fadeStyle]}>
        <View style={{flexDirection: 'column', flex: 1}}>
            <Text style={styles.headerTitle}>Statistics</Text>
            <Text style={styles.headerSubtitle}>Your journal insights</Text>
        </View>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="close" size={24} color="#4a5568" />
        </TouchableOpacity>
      </Animated.View>

      {/* Stats Grid - 2x2 */}
      <Animated.View style={[styles.statsGrid, scaleStyle]}>
        <View style={styles.statCard}>
          <View style={styles.statIconContainer}>
            <Ionicons name="albums" size={24} color="#3182ce" />
          </View>
          <Text style={styles.statNumber}>24</Text>
          <Text style={styles.statLabel}>Total Entries</Text>
        </View>

        <View style={styles.statCard}>
          <View style={styles.statIconContainer}>
            <Ionicons name="heart" size={24} color="#e53e3e" />
          </View>
          <Text style={styles.statNumber}>18</Text>
          <Text style={styles.statLabel}>Emotions Tracked</Text>
        </View>

        <View style={styles.statCard}>
          <View style={styles.statIconContainer}>
            <Ionicons name="calendar" size={24} color="#805ad5" />
          </View>
          <Text style={styles.statNumber}>7</Text>
          <Text style={styles.statLabel}>Day Streak</Text>
        </View>

        <View style={styles.statCard}>
          <View style={styles.statIconContainer}>
            <Ionicons name="time" size={24} color="#38a169" />
          </View>
          <Text style={styles.statNumber}>12</Text>
          <Text style={styles.statLabel}>Days Journaled</Text>
        </View>
      </Animated.View>

      {/* Activity Calendar */}
      <Animated.View style={[styles.section, fadeStyle]}>
        <Text style={styles.sectionTitle}>Activity Calendar</Text>
        <View style={styles.calendarContainer}>
          {Array.from({ length: 52 }, (_, weekIndex) => (
            <View key={weekIndex} style={styles.weekRow}>
              {Array.from({ length: 7 }, (_, dayIndex) => {
                const intensity = Math.floor(Math.random() * 4); // 0-3 for different intensities
                return (
                  <View
                    key={dayIndex}
                    style={[
                      styles.calendarDay,
                      { backgroundColor: getActivityColor(intensity) }
                    ]}
                  />
                );
              })}
            </View>
          ))}
        </View>
        <View style={styles.calendarLegend}>
          <Text style={styles.legendText}>Less</Text>
          {[0, 1, 2, 3].map(intensity => (
            <View
              key={intensity}
              style={[
                styles.legendBox,
                { backgroundColor: getActivityColor(intensity) }
              ]}
            />
          ))}
          <Text style={styles.legendText}>More</Text>
        </View>
      </Animated.View>

      {/* Pie Chart Section */}
      <Animated.View style={[styles.section, fadeStyle]}>
        <Text style={styles.sectionTitle}>Emotion Distribution</Text>
        <View style={styles.pieChartContainer}>
          <View style={styles.pieChart}>
            {/* Pie chart segments */}
            <View style={[styles.pieSegment, { backgroundColor: '#e53e3e', transform: [{ rotate: '0deg' }] }]} />
            <View style={[styles.pieSegment, { backgroundColor: '#3182ce', transform: [{ rotate: '90deg' }] }]} />
            <View style={[styles.pieSegment, { backgroundColor: '#38a169', transform: [{ rotate: '180deg' }] }]} />
            <View style={[styles.pieSegment, { backgroundColor: '#d69e2e', transform: [{ rotate: '270deg' }] }]} />
          </View>
          <View style={styles.pieChartCenter}>
            <Text style={styles.pieChartCenterText}>100%</Text>
          </View>
        </View>
        
        {/* Legend */}
        <View style={styles.pieLegend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#e53e3e' }]} />
            <Text style={styles.legendLabel}>Happy (40%)</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#3182ce' }]} />
            <Text style={styles.legendLabel}>Calm (25%)</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#38a169' }]} />
            <Text style={styles.legendLabel}>Excited (20%)</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#d69e2e' }]} />
            <Text style={styles.legendLabel}>Anxious (15%)</Text>
          </View>
        </View>
      </Animated.View>
    </ScrollView>
  );
};

// Helper function for activity colors
const getActivityColor = (intensity: number): string => {
  switch (intensity) {
    case 0: return '#ebedf0';
    case 1: return '#9be9a8';
    case 2: return '#40c463';
    case 3: return '#30a14e';
    default: return '#ebedf0';
  }
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f7f0f0',
  },
  header: {
    padding: 20,
    paddingTop: 40,
    flexDirection: 'row',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1a202c',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#718096',
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 16,
    marginBottom: 24,
  },
  statCard: {
    width: '48%',
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    margin: '1%',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#f7fafc',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1a202c',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#718096',
    textAlign: 'center',
  },
  section: {
    backgroundColor: 'white',
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1a202c',
    marginBottom: 16,
  },
  calendarContainer: {
    marginBottom: 12,
    flexDirection: 'row',
  },
  weekRow: {
    flexDirection: 'row',
    marginBottom: 2,
  },
  calendarDay: {
    width: 8,
    height: 8,
    margin: 1,
    borderRadius: 1,
  },
  calendarLegend: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  legendText: {
    fontSize: 12,
    color: '#718096',
    marginHorizontal: 8,
  },
  legendBox: {
    width: 8,
    height: 8,
    margin: 1,
    borderRadius: 1,
  },
  pieChartContainer: {
    alignItems: 'center',
    marginBottom: 16,
  },
  pieChart: {
    width: 120,
    height: 120,
    borderRadius: 60,
    position: 'relative',
    overflow: 'hidden',
  },
  pieSegment: {
    position: 'absolute',
    width: '50%',
    height: '50%',
    top: 0,
    left: '50%',
    transformOrigin: '0% 100%',
  },
  pieChartCenter: {
    position: 'absolute',
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'white',
    alignItems: 'center',
    justifyContent: 'center',
    top: 30,
    left: 30,
  },
  pieChartCenterText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#1a202c',
  },
  pieLegend: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-around',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 4,
    marginHorizontal: 8,
  },
  legendColor: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  legendLabel: {
    fontSize: 14,
    color: '#4a5568',
  },
});

export default StatisticsDashboard;
