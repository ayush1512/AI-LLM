import sqlite3
from typing import Dict, Any, List

SCHEMA_METADATA = {
    "categories": {
        "description": "Product category groupings and taxonomy",
        "columns": {
            "category_id": "Primary key (INTEGER)",
            "name": "Category name e.g. Electronics, Books (TEXT)",
            "description": "High level description of goods (TEXT)"
        }
    },
    "products": {
        "description": "Catalog of items available for sale with price and inventory",
        "columns": {
            "product_id": "Primary key (INTEGER)",
            "category_id": "Foreign key to categories.category_id (INTEGER)",
            "name": "Product title e.g. Headphones (TEXT)",
            "price": "Unit retail price in USD (REAL)",
            "stock": "Current warehouse inventory count (INTEGER)"
        }
    },
    "customers": {
        "description": "Registered shoppers and user accounts",
        "columns": {
            "customer_id": "Primary key (INTEGER)",
            "full_name": "Customer first and last name (TEXT)",
            "email": "Email address (TEXT)",
            "city": "Resident city (TEXT)",
            "country": "Resident country code/name (TEXT)",
            "joined_at": "Registration date YYYY-MM-DD (TEXT)"
        }
    },
    "orders": {
        "description": "Customer purchase transaction records and totals",
        "columns": {
            "order_id": "Primary key (INTEGER)",
            "customer_id": "Foreign key to customers.customer_id (INTEGER)",
            "order_date": "Date order placed YYYY-MM-DD (TEXT)",
            "status": "Order fulfillment status e.g. 'completed', 'pending' (TEXT)",
            "total_amount": "Total order charge in USD (REAL)"
        }
    },
    "order_items": {
        "description": "Line items per order linking products to orders",
        "columns": {
            "item_id": "Primary key (INTEGER)",
            "order_id": "Foreign key to orders.order_id (INTEGER)",
            "product_id": "Foreign key to products.product_id (INTEGER)",
            "quantity": "Number of units bought (INTEGER)",
            "unit_price": "Price per unit at time of purchase (REAL)"
        }
    }
}

def get_scoped_schema(user_query: str) -> str:
    """
    Dynamically scopes schema information based on query intent
    to avoid overloading the LLM context window with irrelevant tables.
    """
    lowered = user_query.lower()
    relevant_tables = set()

    keywords = {
        "categories": ["category", "categories", "department", "type"],
        "products": ["product", "item", "stock", "price", "inventory", "gadget", "headphones", "skillet"],
        "customers": ["customer", "client", "buyer", "user", "city", "country", "who", "alice", "bob"],
        "orders": ["order", "revenue", "sales", "spend", "transaction", "amount", "date", "status"],
        "order_items": ["quantity", "unit", "sold", "volume", "line item"]
    }

    for table, kws in keywords.items():
        if any(kw in lowered for kw in kws):
            relevant_tables.add(table)

    # If too few or generic, default to all tables for safe joins
    if len(relevant_tables) <= 1:
        relevant_tables = set(SCHEMA_METADATA.keys())

    # Build structured schema documentation string
    schema_lines = []
    for t_name in sorted(relevant_tables):
        meta = SCHEMA_METADATA[t_name]
        cols = ", ".join([f"{col}: {desc}" for col, desc in meta["columns"].items()])
        schema_lines.append(f"Table '{t_name}': {meta['description']}\n  Columns: {cols}")

    return "\n\n".join(schema_lines)

