import sqlite3
import os

def create_and_seed_db(db_path: str = "ecommerce.db"):
    """Creates a realistic e-commerce SQLite database for Text-to-SQL analytics."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Categories
    cursor.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT
    );
    """)

    # 2. Products
    cursor.execute("""
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        category_id INTEGER,
        name TEXT NOT NULL,
        price REAL NOT NULL,
        stock INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories (category_id)
    );
    """)

    # 3. Customers
    cursor.execute("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE,
        city TEXT,
        country TEXT,
        joined_at TEXT
    );
    """)

    # 4. Orders
    cursor.execute("""
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        order_date TEXT NOT NULL,
        status TEXT NOT NULL,
        total_amount REAL NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
    );
    """)

    # 5. Order Items
    cursor.execute("""
    CREATE TABLE order_items (
        item_id INTEGER PRIMARY KEY,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders (order_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    );
    """)

    # Seed Categories
    categories = [
        (1, "Electronics", "Gadgets, computing, audio, and devices"),
        (2, "Apparel", "Men and women casual and formal clothing"),
        (3, "Home & Kitchen", "Appliances, furniture, and kitchenware"),
        (4, "Books", "Technical, fiction, and business publications")
    ]
    cursor.executemany("INSERT INTO categories VALUES (?, ?, ?)", categories)

    # Seed Products
    products = [
        (101, 1, "Pro Noise-Cancelling Headphones", 249.99, 45),
        (102, 1, "Ultra HD 4K Monitor 27-inch", 399.00, 20),
        (103, 1, "Mechanical Gaming Keyboard", 119.50, 80),
        (104, 2, "Merino Wool Crewneck Sweater", 85.00, 110),
        (105, 2, "Waterproof Trail Running Jacket", 145.00, 60),
        (106, 3, "Espresso Machine & Milk Frother", 299.99, 30),
        (107, 3, "Cast Iron Skillet 12-inch", 49.95, 150),
        (108, 4, "Designing Data-Intensive Applications", 42.00, 200),
        (109, 4, "Site Reliability Engineering", 38.50, 120)
    ]
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products)

    # Seed Customers
    customers = [
        (1, "Alice Montgomery", "alice@example.com", "Seattle", "USA", "2023-01-15"),
        (2, "Bob Chen", "bob@example.com", "San Francisco", "USA", "2023-03-22"),
        (3, "Carlos Morales", "carlos@example.com", "Austin", "USA", "2023-05-10"),
        (4, "Diana Prince", "diana@example.com", "New York", "USA", "2023-07-01"),
        (5, "Evan Wright", "evan@example.com", "London", "UK", "2023-08-19"),
        (6, "Fiona Gallagher", "fiona@example.com", "Chicago", "USA", "2023-11-04")
    ]
    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)", customers)

    # Seed Orders
    orders = [
        (1001, 1, "2024-01-10", "completed", 648.99),
        (1002, 2, "2024-01-14", "completed", 399.00),
        (1003, 1, "2024-02-05", "completed", 85.00),
        (1004, 3, "2024-02-18", "completed", 349.94),
        (1005, 4, "2024-03-02", "completed", 80.50),
        (1006, 5, "2024-03-12", "pending", 249.99),
        (1007, 2, "2024-03-25", "completed", 42.00)
    ]
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)

    # Seed Order Items
    order_items = [
        (1, 1001, 101, 1, 249.99),
        (2, 1001, 102, 1, 399.00),
        (3, 1002, 102, 1, 399.00),
        (4, 1003, 104, 1, 85.00),
        (5, 1004, 106, 1, 299.99),
        (6, 1004, 107, 1, 49.95),
        (7, 1005, 108, 1, 42.00),
        (8, 1005, 109, 1, 38.50),
        (9, 1006, 101, 1, 249.99),
        (10, 1007, 108, 1, 42.00)
    ]
    cursor.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", order_items)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_and_seed_db()
    print("Database seeded successfully: ecommerce.db")

