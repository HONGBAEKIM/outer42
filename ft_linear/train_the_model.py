# Example 
# Training Data (3 cars):

# Car	Normalized Kilometers (X)	Actual Price (Y)
# A	      -1	                     5000
# B	       0	                     3000
# C	       1	                     1000

# Starting Guesses:
# theta0 = 0
# theta1 = 0
# estimate_price(X) = theta0 + theta1 * X
#                   =    0   +    0   * X = 0

# A: estimate_price(-1) = 0 → error = 0 - 5000 = -5000
# B: estimate_price(0) = 0 → error = 0 - 3000 = -3000
# C: estimate_price(1) = 0 → error = 0 - 1000 = -1000

# sum_error_theta0 = -5000 + (-3000) + (-1000) = -9000


# sum_error_theta1 = sum((predicted_price - real_price) * X)
# A: error = -5000, X = -1 → -5000 * -1 = 5000
# B: error = -3000, X = 0 → -3000 * 0 = 0
# C: error = -1000, X = 1 → -1000 * 1 = -1000

# sum_error_theta1 = 5000 + 0 + (-1000) = 4000


# tmp_theta0 = 0 - (0.01 * (1 / 3) * -9000)
#            = 0 - (0.01 * -3000)
#            = 0 + 30
#            = 30

# tmp_theta1 = 0 - (0.01 * (1 / 3) * 4000)
#            = 0 - (0.01 * 1333.33)
#            ≈ 0 - 13.33
#            ≈ -13.33


import json
import csv


# Hypothesis function
def estimate_price(normalized_kilometers, theta0, theta1):
    return theta0 + (theta1 * normalized_kilometers)

# Update theta values

def gradient_descent(normalized_kilometers, price, theta0, theta1, learning_rate, iterations):
    m = len(normalized_kilometers)  # Stores the number of data points in the dataset.
    for _ in range(iterations):

        # Calculate how far off your guesses are

        # sum_error_theta0: are your guesses too high or too low overall?”
        # If they're too low → increase theta0 (move line up)
        sum_error_theta0 = sum(estimate_price(normalized_kilometers[i], theta0, theta1) - price[i] for i in range(m))
        
        # sum_error_theta1: Is the angle of your line wrong?
        # If left-side errors are big and right-side are small → increase slope (theta1)
        sum_error_theta1 = sum((estimate_price(normalized_kilometers[i], theta0, theta1) - price[i]) * normalized_kilometers[i] for i in range(m))
        
        # tmpθ₀ is starting point  
        tmp_theta0 = theta0 - (learning_rate * (1 / m) * sum_error_theta0)

        # tmpθ₁ is degree
        tmp_theta1 = theta1 - (learning_rate * (1 / m) * sum_error_theta1)
        
        theta0, theta1 = tmp_theta0, tmp_theta1  # Simultaneous update
    return theta0, theta1


# Load dataset and normalize kilometers
def load_dataset(file_path):
    kilometers, price = [], []
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            kilometers.append(float(row["km"]))
            price.append(float(row["price"]))
    
    # The average of all the kilometers in the training dataset.
    kilometers_average = sum(kilometers) / len(kilometers)

    # kilometers_std is the average distance
    # ** = to the power of
    kilometers_std = (sum((x - kilometers_average) ** 2 for x in kilometers) / len(kilometers)) ** 0.5
    
    # normalized_kilometers turns the original values into smaller numbers like -1, 0, or 1, so the model can learn more easily
    normalized_kilometers = [(x - kilometers_average) / kilometers_std for x in kilometers]
    return normalized_kilometers, price, kilometers_average, kilometers_std

# Main training script
file_path = "data.csv"  # Assumes your provided file is named "data.csv"
learning_rate = 0.01  # how big the steps are
iterations = 10000 # how many times to train

normalized_kilometers, price, kilometers_average, kilometers_std = load_dataset(file_path)

# Initialize theta values
theta0 = 0.0
theta1 = 0.0

# Train the model
theta0, theta1 = gradient_descent(normalized_kilometers, price, theta0, theta1, learning_rate, iterations)

# Save the trained parameters and normalization values
with open("parameters.json", "w") as file:
    json.dump({
        "theta0": theta0,
        "theta1": theta1,
        "kilometers_average": kilometers_average,
        "kilometers_std": kilometers_std
    }, file)

print(f"Training completed. Parameters saved.")
print(f"Theta0: {theta0}, Theta1: {theta1}")
