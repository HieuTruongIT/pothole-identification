import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import IntroducePage from './Src/IntroPage/IntroducePage';
import HomePage from './Src/HomePage/homepage';
import AboutPage from './Src/AboutPage/about';
import ContactPage from './Src/ContactPage/contact';
import MorePage from './Src/MorePage/morepage';

const Stack = createStackNavigator();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Introduce">
        <Stack.Screen name="Introduce" component={IntroducePage} />
        <Stack.Screen name="Home" component={HomePage} />
        <Stack.Screen name="About" component={AboutPage} />
        <Stack.Screen name="Contact" component={ContactPage} />
        <Stack.Screen name="More" component={MorePage} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
