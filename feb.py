# ==========================================
# Product Recommendation AI Agent
# ==========================================

# Product database
products = {
    "electronics": ["Laptop", "Smartphone", "Wireless Earbuds", "Smart Watch"],
    "clothing": ["T-shirt", "Jeans", "Jacket", "Sneakers"],
    "books": ["Python Programming", "AI Basics", "Data Science Guide"],
    "fitness": ["Yoga Mat", "Dumbbells", "Protein Powder"],
    "gaming": ["Gaming Mouse", "Mechanical Keyboard", "Gaming Headset"]
}

# Recommendation function
def recommend_products(user_interest):

    user_interest = user_interest.lower()

    if user_interest in products:
        print("\nRecommended products for you:")
        for item in products[user_interest]:
            print("-", item)

    else:
        print("\nSorry, no recommendations found.")


# Chat loop
print("===== AI Product Recommendation Agent =====")

while True:

    interest = input("\nEnter your interest (electronics, clothing, books, fitness, gaming) or type exit: ")

    if interest.lower() == "exit":
        print("Thank you for using the recommendation system!")
        break

    recommend_products(interest)