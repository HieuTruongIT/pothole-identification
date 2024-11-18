import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#e6f7ff', // Light blue background
    justifyContent: 'flex-start', // Keep content at the top
  },
  backgroundImage: {
    width: '100%',
    height: 600, // Adjust height as needed
    resizeMode: 'cover', // Ensure image covers the container
  },
  headerContainer: {
    marginTop: 20,
    alignItems: 'center', // Center header text
    marginBottom: 20, // Space below the header
  },
  headerText: {
    fontSize: 40,
    fontWeight: 'bold',
    color: 'white', // White text color for the header
  },
  descriptionContainer: {
    padding: 20,
    backgroundColor: '#f8f8f8', // Light grey background for description
    borderRadius: 10, // Rounded corners for the description
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5, // Shadow for Android
    marginBottom: 50, // Space between the description and footer
  },
  description: {
    fontSize: 16,
    textAlign: 'center',
    color: '#333', // Darker text for better readability
  },
  footer: {
    padding: 20,
    alignItems: 'center', // Center all elements in the footer
  },
  footerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  developers: {
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 20,
    color: '#666',
  },
  socialLinks: {
    flexDirection: 'row', // Arrange social icons in a row
    justifyContent: 'center',
  },
  icon: {
    width: 40,
    height: 40,
    marginHorizontal: 10, // Spacing between the social icons
  },
});

export default styles;
