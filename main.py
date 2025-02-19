import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from fpdf import FPDF

# ------------------------- FILE PATHS ------------------------- #
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

data_file = os.path.join(data_dir, "synthetic_purchase_data.csv")
report_file = os.path.join(data_dir, "customer_analysis_report.pdf")

# Number of customers, products and purchase records
num_customers = 500
num_products = 50
num_purchases = 5000

# Generate random IDs for customers and products
customers = [str(uuid.uuid4()) for _ in range(num_customers)]
products = [str(uuid.uuid4()) for _ in range(num_products)]
categories = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Books"]


def random_date():
    """
    Generates a random date.
    """
    return datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))


# --------------------- DATA GENERATION --------------------- #
def generate_data():
    """
    Generates a synthetic dataset of customer purchases,
    creates a DataFrame and saves it as 'synthetic_purchase_data.csv'.
    """
    data = []
    for _ in range(num_purchases):
        customer_id = random.choice(customers)
        product_id = random.choice(products)
        category = random.choice(categories)
        purchase_amount = round(random.uniform(5, 500), 2)
        purchase_date = random_date().strftime("%Y-%m-%d")
        data.append([customer_id, product_id, category, purchase_amount, purchase_date])

    df_generated = pd.DataFrame(
        data,
        columns=["Customer ID", "Product ID", "Product Category", "Purchase Amount", "Purchase Date"]
    )
    df_generated.to_csv(data_file, index=False)
    print(f"Synthetic dataset generated and saved as '{data_file}'")


# --------------------- DATA ANALYSIS --------------------- #
def analyze_data(df):
    """
    Returns:
      - top_categories: Series of top 5 product categories by count
      - top_products: Series of top 5 products by count
      - avg_spending: DataFrame with mean, sum, and median spending per customer
    """
    top_categories = df["Product Category"].value_counts().head()
    top_products = df["Product ID"].value_counts().head()
    avg_spending = df.groupby("Customer ID")["Purchase Amount"].agg(["mean", "sum", "median"])
    return top_categories, top_products, avg_spending


# --------------------- CUSTOMER CLASSIFICATION --------------------- #
def classify_customers(df):
    """
    Classifies customers based on:
      - frequency (number of purchases)
      - total_spending (sum of purchase amounts)
      - preferred_category (most frequent category)

    Uses K-Means with 3 clusters (Low/Medium/High Spenders).
    Returns a DataFrame with columns:
      [Customer ID, frequency, total_spending, preferred_category, Cluster, Segment]
    """
    customer_stats = df.groupby("Customer ID").agg(
        frequency=("Product ID", "count"),
        total_spending=("Purchase Amount", "sum"),
        preferred_category=("Product Category", lambda x: x.mode()[0])
    ).reset_index()

    # Normalize only numeric columns
    customer_stats_scaled = customer_stats.copy()
    numeric_cols = ["frequency", "total_spending"]
    # Avoid division by zero if std is zero
    for col in numeric_cols:
        if customer_stats_scaled[col].std() == 0:
            # If std == 0, all values are the same, skip normalization
            customer_stats_scaled[col] = 0
        else:
            customer_stats_scaled[col] = (
                                                 customer_stats_scaled[col] - customer_stats_scaled[col].mean()
                                         ) / customer_stats_scaled[col].std()

    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_stats["Cluster"] = kmeans.fit_predict(customer_stats_scaled[numeric_cols])

    # Map cluster IDs to segment names
    cluster_labels = {0: "Low Spenders", 1: "Medium Spenders", 2: "High Spenders"}
    customer_stats["Segment"] = customer_stats["Cluster"].map(cluster_labels)

    return customer_stats


# --------------------- TIME-WEIGHTED MATRIX --------------------- #
def build_time_weighted_matrix(df, alpha=0.01):
    """
    Builds a Customer-Product matrix, giving higher importance to recent purchases.
    Steps:
      1) Convert 'Purchase Date' to datetime if not already.
      2) latest_date = the max 'Purchase Date' in the dataset.
      3) days_diff = (latest_date - purchase_date).days
      4) weight = exp(-alpha * days_diff)
      5) weighted_amount = Purchase Amount * weight
      6) Pivot the table into a matrix [Customer ID x Product ID]
    """
    # Ensure 'Purchase Date' is datetime
    if not np.issubdtype(df["Purchase Date"].dtype, np.datetime64):
        df["Purchase Date"] = pd.to_datetime(df["Purchase Date"])

    latest_date = df["Purchase Date"].max()
    df_weighted = df.copy()
    df_weighted["days_diff"] = (latest_date - df_weighted["Purchase Date"]).dt.days

    # Exponential decay factor
    df_weighted["weight"] = np.exp(-alpha * df_weighted["days_diff"])

    # Weighted amount
    df_weighted["weighted_amount"] = df_weighted["Purchase Amount"] * df_weighted["weight"]

    # Build the pivot table
    customer_product_matrix = df_weighted.pivot_table(
        index="Customer ID",
        columns="Product ID",
        values="weighted_amount",
        aggfunc="sum",
        fill_value=0
    )
    return customer_product_matrix


# --------------------- HYBRID RECOMMENDER SYSTEM --------------------- #
def recommend_products(
        df,
        customer_id,
        weight_collab=0.7,
        weight_content=0.3,
        alpha=0.01
):
    """
    Hybrid recommender that combines:
      1) Time-weighted Collaborative Filtering
      2) Content-Based Filtering
      3) Weighted combination of final lists (70% CF, 30% CB).

    Parameters:
      - df: DataFrame with columns [Customer ID, Product ID, Product Category, Purchase Amount, Purchase Date]
      - customer_id: ID of the customer for whom we want recommendations
      - weight_collab: fraction of the final list to take from collaborative filtering
      - weight_content: fraction of the final list to take from content-based
      - alpha: exponential decay factor for time weighting (larger alpha -> faster decay of older purchases)

    Returns: A list of recommended product IDs.
    """
    # 1) Build time-weighted matrix for collaborative filtering
    customer_product_matrix = build_time_weighted_matrix(df, alpha=alpha)

    # 2) Product-Category matrix (standard Purchase Amount)
    product_category_matrix = df.pivot_table(
        index="Product ID",
        columns="Product Category",
        values="Purchase Amount",
        fill_value=0
    )

    # If the customer is not present, return an info
    if customer_id not in customer_product_matrix.index:
        return ["No recommendation available"]

    # ---------- COLLABORATIVE (time-weighted) ---------- #
    neigh = NearestNeighbors(metric="cosine", algorithm="brute")
    neigh.fit(customer_product_matrix.to_numpy())

    # Vector for the target customer
    customer_vector = customer_product_matrix.loc[customer_id].to_numpy().reshape(1, -1)
    distances, indices = neigh.kneighbors(customer_vector, n_neighbors=5)

    # Summation of weighted_amount for the k-nearest customers
    collab_scores = (
        customer_product_matrix.iloc[indices[0]]
        .sum()
        .sort_values(ascending=False)
    )
    collaborative_all = collab_scores.index.tolist()  # ranked list

    # ---------- CONTENT-BASED ---------- #
    purchased_products = df[df["Customer ID"] == customer_id]["Product ID"].unique()
    content_based_all = []

    for prod in purchased_products:
        if prod in product_category_matrix.index:
            sim_matrix = cosine_similarity(
                product_category_matrix.loc[prod].values.reshape(1, -1),
                product_category_matrix
            )
            # Sort similarity from highest to lowest
            sim_indices = np.argsort(sim_matrix[0])[::-1]
            # sim_indices[0] is the product itself, so we skip that
            top_similar = sim_indices[1:6]  # take next 5
            content_based_all.extend(product_category_matrix.index[top_similar].tolist())

    # ---------- WEIGHTED MERGE (70% Collab, 30% Content) ---------- #
    collab_count = int(len(collaborative_all) * weight_collab)
    content_count = int(len(content_based_all) * weight_content)

    # Take the top 'collab_count' from collaborative
    collab_slice = collaborative_all[:collab_count]
    # Take the top 'content_count' from content-based
    content_slice = content_based_all[:content_count]

    # Merge both lists, remove duplicates while preserving order
    combined_list = collab_slice + content_slice
    final_list = list(dict.fromkeys(combined_list))

    # Let's pick top 5 final recommendations
    final_recommendations = final_list[:5]

    if not final_recommendations:
        return ["No recommendation available"]
    return final_recommendations


# --------------------- PDF REPORT GENERATION --------------------- #
def generate_pdf_report(
        top_categories,
        top_products,
        avg_spending,
        customer_segments,
        example_customer,
        example_recommendation
):
    """
    Generates a PDF report with:
      - Top product categories
      - Top products
      - Sample spending statistics (first 5 customers)
      - Sample customer segmentation (first 5 customers)
      - Recommendation example for a single customer
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Customer Purchase Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Top product categories
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Top Selling Product Categories:", ln=True)
    for category, count in top_categories.items():
        pdf.cell(200, 10, f"{category}: {count} purchases", ln=True)
    pdf.ln(10)

    # Top products
    pdf.cell(200, 10, "Top Selling Products:", ln=True)
    for product, count in top_products.items():
        pdf.cell(200, 10, f"Product {product}: {count} purchases", ln=True)
    pdf.ln(10)

    # Sample spending stats (just 5 customers to keep PDF short)
    pdf.cell(200, 10, "Customer Spending Statistics (Sample of 5):", ln=True)
    sample_stats = avg_spending.head(5)
    for customer, row in sample_stats.iterrows():
        pdf.cell(
            200, 10,
            f"Customer {customer}: "
            f"Mean ${row['mean']:.2f}, "
            f"Total ${row['sum']:.2f}, "
            f"Median ${row['median']:.2f}",
            ln=True
        )
    pdf.ln(10)

    # Sample segmentation (first 5 customers)
    pdf.cell(200, 10, "Customer Segmentation (Sample of 5):", ln=True)
    for _, seg_row in customer_segments.head(5).iterrows():
        pdf.cell(
            200, 10,
            f"Customer {seg_row['Customer ID']}: "
            f"{seg_row['Segment']} - Prefers {seg_row['preferred_category']}",
            ln=True
        )
    pdf.ln(10)

    # Example recommendation
    pdf.cell(200, 10, f"Example Recommendation for Customer {example_customer}:", ln=True)
    if isinstance(example_recommendation, list):
        pdf.cell(200, 10, f"Recommended Products: {', '.join(example_recommendation)}", ln=True)
    else:
        pdf.cell(200, 10, f"Recommended Products: {example_recommendation}", ln=True)
    pdf.ln(10)

    pdf.output(report_file)
    print(f"Report generated: {report_file}")


# --------------------- MAIN FUNCTION --------------------- #
def main():
    # 1) Generate the synthetic data and save to CSV
    generate_data()

    # 2) Load the CSV
    df = pd.read_csv(data_file)

    # 3) Analyze data (top categories/products, average spending)
    top_categories, top_products, avg_spending = analyze_data(df)

    # 4) Classify customers (frequency, total_spending, preferred_category)
    customer_segments = classify_customers(df)

    # 5) Pick one example customer to show recommendation
    example_customer = df.iloc[0]["Customer ID"]
    example_recommendation = recommend_products(
        df,
        customer_id=example_customer,
        weight_collab=0.7,
        weight_content=0.3,
        alpha=0.01  # Increase alpha for faster decay of old purchases
    )

    # 6) Generate PDF report
    generate_pdf_report(
        top_categories,
        top_products,
        avg_spending,
        customer_segments,
        example_customer,
        example_recommendation
    )


if __name__ == "__main__":
    main()
