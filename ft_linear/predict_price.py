# Example:
# If your training data had:

# 10000 km, 20000 km, 30000 km

# Then:

# kilometers_average = (100000 + 150000 + 250000) / 3 = 166666.67
# kilometers_std = square root(((100000−166666.67)² + (150000−166666.67)² + (250000−166666.67)²) / 3) ≈ 62225.4



# What Is a Linear Function?
# linear function = Straght line equation

# What Does It Mean to Train a Linear Function?
# Letting the computer figure out the best values for theta0 and theta1

# What Is the Gradient Descent Algorithm?
# This is the learning tool. It helps the computer figure out the best line.




# what can be significantly impact the price of a car?
# 1. Make and Model
# 2. Year of Manufacture
# 3. Mileage (Kilometers Driven) (we are doing now)
# 4. Condition of the Car
# 5. etc...


# reading and writing the parameters (theta0 and theta1) stored in a JSON file
import json


# normalized_kilometers: The normalized kilometers driven by the car (input from the user).

# theta0: The starting price.
# theta1: how much the price drops as kilometers increase.

# It calculates the estimated price using the formula: 
# price = 𝜃0 + (𝜃1 × kilometers)
def estimate_price(normalized_kilometers, theta0, theta1):
    return theta0 + (theta1 * normalized_kilometers)

# Load the trained parameters
try:
    with open("parameters.json", "r") as file:
        params = json.load(file)
        theta0 = params["theta0"]
        theta1 = params["theta1"]
        kilometers_average = params["kilometers_average"]
        kilometers_std = params["kilometers_std"]
except FileNotFoundError:
    print("Error: Trained parameters not found. Please run the training program first.")
    exit()

# Calculate when the price will be 0
if theta1 != 0:  # Avoid division by zero
    zero_price_normalized_kilometers = -theta0 / theta1
    zero_price_kilometers = zero_price_normalized_kilometers * kilometers_std + kilometers_average  # Convert back to original scale
else:
    zero_price_kilometers = None

# Ask the user how far the car has been driven.
kilometers = float(input("Enter the kilometers driven by the car: "))

# Normalize the input kilometers
normalized_kilometers = (kilometers - kilometers_average) / kilometers_std

# Predict the price
# also make sure the price is not negative using max(0, ...)
price = max(0, estimate_price(normalized_kilometers, theta0, theta1))

print(f"The estimated price for a car with {kilometers:.2f} km is: €{price:.2f}")

# Display when the price will drop to 0
if zero_price_kilometers is not None:
    print(f"The price is expected to drop to €0 when the kilometers reach approximately {zero_price_kilometers:.1f} km.")
else:
    print("The model predicts the price will not drop to €0.")


