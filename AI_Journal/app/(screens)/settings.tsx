import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Divider from '../components/Divider';
import { router } from 'expo-router';

type SettingsItemProps = {
  title: string;
  subtitle?: string;
  icon: React.ReactNode;
  onPress?: () => void;
  showArrow?: boolean;
};

const SettingsItem = ({
  title,
  subtitle,
  icon,
  onPress,
  showArrow = true,
}: SettingsItemProps) => (
  <TouchableOpacity style={styles.settingsItem} onPress={onPress}>
    <View style={styles.settingsItemLeft}>
      <View style={styles.iconContainer}>{icon}</View>
      <View style={styles.textContainer}>
        <Text style={styles.settingsTitle}>{title}</Text>
        {subtitle && <Text style={styles.settingsSubtitle}>{subtitle}</Text>}
      </View>
    </View>
    {showArrow && <Ionicons name="chevron-forward" size={20} color="#9ca3af" />}
  </TouchableOpacity>
);

const SettingsSection = ({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) => (
  <View style={styles.section}>
    <Text style={styles.sectionTitle}>{title}</Text>
    <View style={styles.sectionContent}>{children}</View>
  </View>
);

export default function Settings() {
  const handleSignOut = () => {
    Alert.alert('Sign Out', 'Are you sure you want to sign out?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Sign Out',
        style: 'destructive',
        onPress: () => console.log('Sign out pressed'),
      },
    ]);
  };

  const handleSignIn = () => {
    console.log('Sign in pressed');
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Settings</Text>
        <TouchableOpacity
          onPress={() => router.back()}
        >
          <Ionicons name="close" size={24} color="#4a5568" />
        </TouchableOpacity>
      </View>

      <SettingsSection title="Account">
        <SettingsItem
          title="Profile"
          subtitle="Edit your profile information"
          icon={<Ionicons name="person" size={20} color="#3182ce" />}
          onPress={() => console.log('Profile pressed')}
        />
        <Divider />
        <SettingsItem
          title="Privacy"
          subtitle="Manage your privacy settings"
          icon={<Ionicons name="shield-checkmark" size={20} color="#38a169" />}
          onPress={() => console.log('Privacy pressed')}
        />
        <Divider />
      </SettingsSection>

      <SettingsSection title="Preferences">
        <SettingsItem
          title="Notifications"
          subtitle="Manage notification preferences"
          icon={<Ionicons name="notifications" size={20} color="#d69e2e" />}
          onPress={() => console.log('Notifications pressed')}
        />
        <Divider />
        <SettingsItem
          title="Theme"
          subtitle="Light, dark, or system"
          icon={<Ionicons name="color-palette" size={20} color="#805ad5" />}
          onPress={() => console.log('Theme pressed')}
        />
        <Divider />
      </SettingsSection>

      <SettingsSection title="Support">
        <SettingsItem
          title="Help & Support"
          subtitle="Get help and contact support"
          icon={<Ionicons name="help-circle" size={20} color="#38a169" />}
          onPress={() => console.log('Help pressed')}
        />
        <Divider />
        <SettingsItem
          title="About"
          subtitle="App version and information"
          icon={
            <Ionicons name="information-circle" size={20} color="#9ca3af" />
          }
          onPress={() => console.log('About pressed')}
        />
      </SettingsSection>

      <View style={styles.authSection}>
        <TouchableOpacity style={styles.signInButton} onPress={handleSignIn}>
          <Ionicons name="log-in" size={20} color="#3182ce" />
          <Text style={styles.signInText}>Sign In</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.signOutButton} onPress={handleSignOut}>
          <Ionicons name="log-out" size={20} color="#e53e3e" />
          <Text style={styles.signOutText}>Sign Out</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f7f0f0',
  },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 20,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1a202c',
    flex: 1
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#4a5568',
    marginBottom: 8,
    paddingHorizontal: 16,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  sectionContent: {
    backgroundColor: 'white',
    borderRadius: 12,
    marginHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 8,
  },
  settingsItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 16,
    paddingHorizontal: 16,
  },
  settingsItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  iconContainer: {
    width: 36,
    height: 36,
    borderRadius: 8,
    backgroundColor: '#f7fafc',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  textContainer: {
    flex: 1,
  },
  settingsTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: '#1a202c',
  },
  settingsSubtitle: {
    fontSize: 14,
    color: '#718096',
    marginTop: 2,
  },
  authSection: {
    paddingHorizontal: 16,
    paddingVertical: 20,
    gap: 12,
  },
  signInButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ebf8ff',
    paddingVertical: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#3182ce',
  },
  signInText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#3182ce',
    marginLeft: 8,
  },
  signOutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fed7d7',
    paddingVertical: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e53e3e',
  },
  signOutText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#e53e3e',
    marginLeft: 8,
  },
});
