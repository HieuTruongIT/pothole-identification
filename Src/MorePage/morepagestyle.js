import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#f5f5f5',
  },
  headerContainer: {
    marginBottom: 16,
    alignItems: 'center',
  },
  headerText: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#005eb8',
  },
  infoContainer: {
    marginBottom: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#333',
  },
  description: {
    fontSize: 16,
    color: '#666',
    marginBottom: 8,
  },
  date: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  teamContainer: {
    marginBottom: 16,
  },
  subHeader: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  teamMember: {
    fontSize: 16,
    color: '#666',
    marginBottom: 4,
  },
  iconsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 24,
  },
  iconButton: {
    alignItems: 'center',
  },
  iconImage: {
    width: 40,
    height: 40,
    marginBottom: 8,
  },
  iconLabel: {
    fontSize: 14,
    color: '#666',
  },
  mapButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#005eb8',
    paddingVertical: 12,
    borderRadius: 8,
  },
  mapIcon: {
    width: 24,
    height: 24,
    marginRight: 8,
  },
  mapButtonText: {
    fontSize: 16,
    color: '#fff',
  },
});

export default styles;
