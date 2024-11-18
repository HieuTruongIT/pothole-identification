import { DefaultTheme } from '@react-navigation/native';
import React from 'react';
import styles from './morepagestyle';

import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Image } from 'react-native';

function MorePage() {
  return (
    <ScrollView style={styles.container}>
      <View style={styles.headerContainer}>
        <Text style={styles.headerText}>More</Text>
      </View>

      <View style={styles.infoContainer}>
        <Text style={styles.title}>Pothole Detection Project</Text>
        <Text style={styles.description}>
          This project focuses on using machine learning algorithms to detect potholes on the road using image processing and sensor data.
          The system helps to ensure safer travel conditions by alerting drivers about potential potholes ahead.
        </Text>
        <Text style={styles.date}>Start Date: January 1, 2024</Text>
        <Text style={styles.date}>End Date: December 31, 2024</Text>
      </View>

      <View style={styles.teamContainer}>
        <Text style={styles.subHeader}>Project Team Members</Text>
        <Text style={styles.teamMember}>- Hiếu</Text>
        <Text style={styles.teamMember}>- Hiển</Text>
        <Text style={styles.teamMember}>- Sĩ</Text>
        <Text style={styles.teamMember}>- Lộc</Text>
      </View>

      <View style={styles.iconsContainer}>
        <TouchableOpacity style={styles.iconButton}>
          <Image source={require('../../assets/pothole_detection_image/github.png')} style={styles.iconImage} />
          <Text style={styles.iconLabel}>Github</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.iconButton}>
          <Image source={require('../../assets/pothole_detection_image/aetting.png')} style={styles.iconImage} />
          <Text style={styles.iconLabel}>Settings</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={styles.mapButton}>
        <Text style={styles.mapButtonText}>Show Potholes Map</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}
export default MorePage;