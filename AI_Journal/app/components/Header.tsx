import React from 'react';
import { View, StyleSheet, Text, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';

type HeaderButtonItemProps = {
  title: string;
  subTitle: string;
  icon: React.ReactNode;
};
const HeaderButtonItem = ({ title, subTitle, icon }: HeaderButtonItemProps) => {
  return (
    <View style={styles.buttonItem}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 5 }}>
        {icon}
        <Text style={styles.buttonHeaderText}>{title}</Text>
      </View>
      <Text style={styles.buttonSubText}>{subTitle}</Text>
    </View>
  );
};

const Header: React.FC = () => {
  return (
    <View style={styles.container}>
      <View
        style={{
          flexDirection: 'row',
          alignItems: 'center',
          gap: 5,
          paddingHorizontal: 12,
        }}
      >
        <Text style={styles.title}>Journal</Text>
        {/* button group */}
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 5 }}>
          <TouchableOpacity
            style={styles.buttonItemContainer}
            onPress={() => {
              // handle first button press
              console.log('search pressed');
            }}
          >
            <Ionicons name="search" size={20} color="#4a5568" />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.buttonItemContainer}
            onPress={() => {
              router.push('/(screens)/settings');
            }}
          >
            <Ionicons name="settings" size={20} color="#4a5568" />
          </TouchableOpacity>
        </View>
      </View>
      {/* button groups here */}
      <TouchableOpacity
        onPress={() => {
          router.push('/(screens)/statistics');
        }}
        style={{
          flexDirection: 'row',
          marginTop: 12,
        }}
      >
        <HeaderButtonItem
          title="6"
          subTitle="Entries"
          icon={<Ionicons name="albums" size={20} color="#3182ce" />}
        />
        <HeaderButtonItem
          title="18"
          subTitle="Emotions"
          icon={<Ionicons name="heart" size={20} color="#e53e3e" />}
        />
        <HeaderButtonItem
          title="3"
          subTitle="Days Streak"
          icon={<Ionicons name="calendar" size={20} color="#805ad5" />}
        />
        <HeaderButtonItem
          title="12"
          subTitle="Days Journaled"
          icon={<Ionicons name="calendar" size={20} color="#38a169" />}
        />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'transparent',
    height: '16%',
    width: '100%',
    alignItems: 'flex-start',
    justifyContent: 'flex-end',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#000',
    flex: 1,
  },
  buttonItem: {
    flex: 1,
    flexDirection: 'column',
    alignItems: 'flex-start',
    justifyContent: 'center',
    gap: 3,
  },
  buttonHeaderText: {
    fontSize: 16,
    color: '#000',
  },
  buttonSubText: {
    fontSize: 12,
    color: 'slategray',
  },
  buttonItemContainer: {
    width: 32,
    height: 32,
    backgroundColor: '#e0e0e0',
    borderRadius: 8,
    marginHorizontal: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default Header;
