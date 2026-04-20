"""
main.py - Car Evaluation Prediction System
Terminal application that loads the trained model and provides predictions based on user input.
"""

import joblib
import numpy as np
import os
import sys
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

class CarEvaluationSystem:
    """Car Evaluation Prediction System"""
    
    def __init__(self, model_path='car_model.joblib'):
        """
        Initialize the car evaluation system.
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.feature_encoder = None
        self.target_encoder = None
        self.categories = None
        self.column_names = None
        
    def load_model(self):
        """Load the trained model and encoders"""
        try:
            if not os.path.exists(self.model_path):
                print(Fore.RED + f"Error: Model file '{self.model_path}' not found!")
                print(Fore.YELLOW + "Please run 'model.py' first to train and save the model.")
                sys.exit(1)
            
            self.model_data = joblib.load(self.model_path)
            self.model = self.model_data['model']
            self.feature_encoder = self.model_data['feature_encoder']
            self.target_encoder = self.model_data['target_encoder']
            self.categories = self.model_data['categories']
            self.column_names = self.model_data['column_names']
            
            print(Fore.GREEN + "✓ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(Fore.RED + f"Error loading model: {str(e)}")
            sys.exit(1)
    
    def display_header(self):
        """Display application header"""
        print("\n" + "="*60)
        print(Fore.CYAN + Style.BRIGHT + "🚗 CAR EVALUATION PREDICTION SYSTEM 🚗")
        print("="*60)
        print(Fore.WHITE + "This system predicts the overall condition of a car")
        print("based on six key attributes.")
        print("-"*60)
    
    def display_categories(self):
        """Display available categories for each attribute"""
        print("\n" + Fore.YELLOW + Style.BRIGHT + "📋 ATTRIBUTE OPTIONS:")
        print("-"*60)
        
        descriptions = {
            'buying': "Buying price",
            'maint': "Maintenance cost",
            'doors': "Number of doors",
            'persons': "Passenger capacity",
            'lug_boot': "Luggage boot size",
            'safety': "Safety rating"
        }
        
        for attr, desc in descriptions.items():
            options = self.categories[attr]
            print(Fore.CYAN + f"{desc:20} " + Fore.WHITE + "→ " + 
                  Fore.GREEN + ", ".join(options))
    
    def display_example(self):
        """Display an example input"""
        print("\n" + Fore.YELLOW + Style.BRIGHT + "📝 EXAMPLE INPUT:")
        print(Fore.WHITE + "vhigh,med,3,2,small,low")
        print(Fore.CYAN + "  → buying: vhigh, maint: med, doors: 3, persons: 2, lug_boot: small, safety: low")
        print(Fore.CYAN + "  → Predicted condition: " + Fore.RED + "unacceptable")
    
    def validate_input(self, input_values):
        """
        Validate user input against allowed categories.
        
        Args:
            input_values (list): List of input strings
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Check length
        if len(input_values) != 6:
            return False, f"Expected 6 values, but got {len(input_values)}"
        
        # Validate each value against allowed categories
        for i, (value, attr) in enumerate(zip(input_values, self.column_names)):
            value = value.strip()
            if value not in self.categories[attr]:
                allowed = ", ".join(self.categories[attr])
                return False, f"Invalid value '{value}' for {attr}. Allowed: {allowed}"
        
        return True, None
    
    def predict(self, input_values):
        """
        Make a prediction based on user input.
        
        Args:
            input_values (list): List of validated input strings
            
        Returns:
            str: Predicted car condition
        """
        # Clean input values
        input_values = [v.strip() for v in input_values]
        
        # Encode the input values
        encoded_input = self.feature_encoder.transform([input_values])
        
        # Make prediction
        prediction_encoded = self.model.predict(encoded_input)[0]
        prediction = self.target_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(encoded_input)[0]
        
        return prediction, probabilities
    
    def get_prediction_color(self, prediction):
        """Return appropriate color for prediction"""
        colors = {
            'unacc': Fore.RED,
            'acc': Fore.YELLOW,
            'good': Fore.GREEN,
            'vgood': Fore.CYAN
        }
        return colors.get(prediction, Fore.WHITE)
    
    def format_prediction(self, prediction):
        """Format prediction with full text"""
        mapping = {
            'unacc': 'Unacceptable',
            'acc': 'Acceptable',
            'good': 'Good',
            'vgood': 'Very Good'
        }
        return mapping.get(prediction, prediction)
    
    def display_prediction_details(self, input_values, prediction, probabilities):
        """Display detailed prediction results"""
        print("\n" + "="*60)
        print(Fore.CYAN + Style.BRIGHT + "🔮 PREDICTION RESULTS")
        print("="*60)
        
        # Display input summary
        print(Fore.WHITE + "\n📊 Input Summary:")
        for attr, value in zip(self.column_names, input_values):
            print(f"  {attr:12} → {value}")
        
        # Display prediction
        print(Fore.WHITE + "\n🎯 Predicted Condition:")
        color = self.get_prediction_color(prediction)
        formatted_pred = self.format_prediction(prediction)
        print(color + Style.BRIGHT + f"  {formatted_pred} ({prediction})")
        
        # Display probabilities
        print(Fore.WHITE + "\n📈 Prediction Confidence:")
        for class_name, prob in zip(self.target_encoder.classes_, probabilities):
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            color = self.get_prediction_color(class_name)
            print(f"  {self.format_prediction(class_name):15} {color}{bar} {prob*100:.1f}%")
        
        print("\n" + "="*60)
    
    def run(self):
        """Main application loop"""
        self.load_model()
        self.display_header()
        self.display_categories()
        self.display_example()
        
        print("\n" + Fore.YELLOW + "💡 TIPS:")
        print("• Enter comma-separated values for all six attributes")
        print("• Type 'help' to see category options")
        print("• Type 'example' to see an example")
        print("• Type 'quit' or 'exit' to close the application")
        print("-"*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n" + Fore.CYAN + "Enter car attributes: " + Fore.WHITE).strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(Fore.YELLOW + "\nThank you for using the Car Evaluation System. Goodbye! 👋")
                    break
                elif user_input.lower() == 'help':
                    self.display_categories()
                    continue
                elif user_input.lower() == 'example':
                    self.display_example()
                    continue
                elif not user_input:
                    continue
                
                # Parse input
                input_values = [v.strip() for v in user_input.split(',')]
                
                # Validate input
                is_valid, error_msg = self.validate_input(input_values)
                
                if not is_valid:
                    print(Fore.RED + f"\n❌ Invalid input: {error_msg}")
                    print(Fore.YELLOW + "Please try again or type 'help' for guidance.")
                    continue
                
                # Make prediction
                prediction, probabilities = self.predict(input_values)
                
                # Display results
                self.display_prediction_details(input_values, prediction, probabilities)
                
            except KeyboardInterrupt:
                print(Fore.YELLOW + "\n\nInterrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(Fore.RED + f"\n❌ An error occurred: {str(e)}")
                print(Fore.YELLOW + "Please check your input and try again.")

def main():
    """Main function"""
    # Check for required packages
    try:
        import sklearn
        import pandas
        import colorama
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("\nPlease install required packages:")
        print("pip install scikit-learn pandas colorama joblib")
        sys.exit(1)
    
    # Create and run the system
    system = CarEvaluationSystem()
    system.run()

if __name__ == "__main__":
    main()