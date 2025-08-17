# Auto_Mind - AI Car Price Prediction

A Flask-based web application that uses machine learning to predict car prices based on various features like company, model, year, fuel type, and kilometers driven.

## Project Structure

```
Auto_Mind/
├── app.py                          # Main Flask application
├── static/
│   └── css/
│       └── styles.css             # External CSS stylesheet
├── templates/
│   ├── index.html                 # Home page
│   ├── predict.html               # Prediction form page
│   ├── about.html                 # About page
│   ├── services.html              # Services page
│   └── contact.html               # Contact page
├── LinearRegressionModel.pkl      # Trained ML model
├── Cleaned_Car.csv                # Cleaned car dataset
└── car_pred.csv                   # Original car dataset
```

## Features

- **AI-Powered Predictions**: Uses a trained Linear Regression model for accurate car price predictions
- **Responsive Design**: Modern, mobile-friendly interface with smooth animations
- **Dynamic Form**: Company selection automatically updates available car models
- **Real-time Results**: Instant price predictions with loading animations
- **Professional UI**: Dark theme with gradient accents and smooth transitions

## Recent Updates

### CSS Separation (Latest)
- **Extracted all internal CSS** from HTML files into a single external stylesheet
- **Created `/static/css/styles.css`** containing all styles organized by component
- **Updated all HTML templates** to link to the external CSS file
- **Improved maintainability** by centralizing all styling in one location
- **Added missing pages**: Services and Contact pages for complete navigation

### Benefits of CSS Separation
- **Easier Maintenance**: All styles in one place
- **Better Performance**: CSS can be cached by browsers
- **Cleaner HTML**: Reduced file sizes and improved readability
- **Consistent Styling**: Centralized design system across all pages
- **Easier Updates**: Modify styles without touching HTML files

## Installation & Usage

1. **Install Dependencies**:
   ```bash
   pip install flask pandas numpy scikit-learn
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Application**:
   - Open your browser and go to `http://localhost:5000`
   - Navigate between pages using the navigation menu
   - Use the prediction form to get car price estimates

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Model**: Linear Regression (scikit-learn)
- **Data**: Car dataset with features like company, model, year, fuel type, kilometers driven
- **Styling**: Custom CSS with CSS variables, animations, and responsive design

## CSS Architecture

The external CSS file (`styles.css`) is organized into logical sections:

- **CSS Variables**: Centralized color scheme and design tokens
- **Reset & Base Styles**: Global styling and typography
- **Navigation**: Header and navigation components
- **Hero Section**: Landing page main content
- **Features**: Service highlights and cards
- **Forms**: Prediction form styling
- **Responsive Design**: Mobile-first responsive breakpoints
- **Animations**: Keyframes and transitions

## Contributing

Feel free to contribute by:
- Improving the UI/UX design
- Adding new features
- Optimizing the ML model
- Enhancing the responsive design
- Adding more car brands and models

## License

This project is open source and available under the MIT License.
