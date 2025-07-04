import React from 'react';
import { View, StyleSheet, Text } from 'react-native';

type HeaderButtonItemProps = {
  title: string;
  subTitle: string;
};
const HeaderButtonItem = ({ title, subTitle }: HeaderButtonItemProps) => {
  return (
    <View style={styles.buttonItem}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 5 }}>
        {/* icon */}
        <View
          style={{
            width: 20,
            height: 20,
            backgroundColor: 'red',
            borderRadius: 4,
            marginBottom: 4,
          }}
        />
        <Text style={styles.buttonHeaderText}>{title}</Text>
      </View>
      <Text style={styles.buttonSubText}>{subTitle}</Text>
    </View>
  );
};

const Header: React.FC = () => {
  return (
    <View style={styles.container}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 5, paddingHorizontal: 12}}>
        <Text style={styles.title}>Journal</Text>
        {/* button group */}
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 5 }}>
        <View
          style={{
            width: 32,
            height: 32,
            backgroundColor: '#e0e0e0',
            borderRadius: 8,
            marginHorizontal: 2,
            alignItems: 'center',
            justifyContent: 'center',
          }}
        />
        <View
          style={{
            width: 32,
            height: 32,
            backgroundColor: '#e0e0e0',
            borderRadius: 8,
            marginHorizontal: 2,
            alignItems: 'center',
            justifyContent: 'center',
          }}
        />
        </View>
      </View>
      {/* button groups here */}
      <View style={{ flexDirection: 'row', marginTop: 12 }}>
        <HeaderButtonItem title="All" subTitle="Entries this year" />
        <HeaderButtonItem title="All" subTitle="Entries this year" />
        <HeaderButtonItem title="All" subTitle="Entries this year" />
      </View>
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
    flex: 1
  },
  buttonItem: {
    flex: 1,
    alignItems: 'flex-start',
    justifyContent: 'center',
  },
  buttonHeaderText: {
    fontSize: 16,
    color: '#000',
  },
  buttonSubText: {
    fontSize: 12,
    color: 'slategray',
  },
});

export default Header;
