# Vending Machine Simulator

stock = {
    1: {"name": "Pepsi", "price": 20, "qty": 5},
    2: {"name": "Chips", "price": 10, "qty": 5},
    3: {"name": "Chocolate", "price": 15, "qty": 5}
}

while True:
    print("\n--- VENDING MACHINE ---")
    
    for key, item in stock.items():
        print(f"{key}. {item['name']} (Rs {item['price']}) [Stock: {item['qty']}]")
    
    print("4. Exit")

    try:
        choice = int(input("Enter your choice: "))
    except:
        print("Invalid input!")
        continue

    if choice == 4:
        print("Thank you!")
        break

    if choice not in stock:
        print("Invalid choice!")
        continue

    if stock[choice]["qty"] > 0:
        stock[choice]["qty"] -= 1
        print(f"{stock[choice]['name']} dispensed successfully!")
    else:
        print("Out of stock!")
