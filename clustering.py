import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
import pandas as pd
import numpy as np

# Step 1: Read data and preprocess
def read_and_preprocess_data(file_path):
    # Read data
    df = pd.read_csv(file_path)

    # Keep only 'Movie Name' and 'IMDB Rating' columns
    df = df[['Movie Name', 'IMDB Rating']]

    # Normalize numerical columns
    df['IMDB Rating'] /= 10  # Scale IMDB Rating to be between 0 and 1

    # Drop rows with missing values
    df = df.dropna()

    return df


# Step 2: Get k from user
def get_k_from_user():
    k = simpledialog.askinteger("Input", "Enter the number of clusters (k):")
    return k


# Step 3: Initialize centroids randomly
def initialize_centroids(df, k):
    centroids_indices = np.random.choice(df.index, size=k, replace=False)
    centroids = df.loc[centroids_indices].copy()  # Make a copy to avoid modifying the original DataFrame
    print("Initial centroids:")
    print(centroids)
    return centroids


# Step 4: Calculate Euclidean distance
def euclidean_distance(x1, x2):
    return np.abs(x1 - x2)


# Step 5: Assign data points to clusters based on centroids
def assign_to_clusters(df, centroids):
    clusters = {}
    for i in range(len(centroids)):
        clusters[i] = []

    for index, row in df.iterrows():
        distances = [euclidean_distance(row['IMDB Rating'], centroid['IMDB Rating']) for _, centroid in centroids.iterrows()]
        closest_centroid_index = np.argmin(distances)
        clusters[closest_centroid_index].append((row['Movie Name'], row['IMDB Rating']))

    return clusters


# Step 6: Update centroids based on cluster mean
def update_centroids(df, clusters):
    new_centroids = pd.DataFrame(columns=['Movie Name', 'IMDB Rating'])
    for cluster_index, data_points in clusters.items():
        if len(data_points) == 0:
            continue
        ratings = [point[1] for point in data_points]
        mean_rating = np.mean(ratings)
        movie_names = [point[0] for point in data_points]
        new_centroids.loc[cluster_index] = {'Movie Name': movie_names, 'IMDB Rating': mean_rating}
    return new_centroids


# Main function to perform k-means clustering
def k_means_clustering(df, k, max_iterations=1000):
    # Initialize centroids
    centroids = initialize_centroids(df, k)
    prev_centroids = None  # Store centroids from the previous iteration

    for _ in range(max_iterations):
        # Assign data points to clusters
        clusters = assign_to_clusters(df, centroids)

        # Update centroids
        centroids = update_centroids(df, clusters)  # Update centroids as DataFrame

        # Check for convergence by comparing centroids
        if centroids.equals(prev_centroids):
            break  # Convergence reached, exit loop

        prev_centroids = centroids.copy()  # Make a copy to store previous centroids

    return clusters, centroids


# Outlier detection using Interquartile Range (IQR)
def detect_outliers(df):
    q1 = df['IMDB Rating'].quantile(0.25)
    q3 = df['IMDB Rating'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df['IMDB Rating'] < lower_bound) | (df['IMDB Rating'] > upper_bound)]
    return list(zip(outliers['Movie Name'], outliers['IMDB Rating']))


def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = browse_file()
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return

    df = read_and_preprocess_data(file_path)

    k = get_k_from_user()

    clusters, centroids = k_means_clustering(df, k)

    display_clusters(clusters, df)


def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select a file", filetypes=[("CSV files", "*.csv")]
    )
    return file_path


def display_clusters(clusters, df):
    root = tk.Tk()
    root.title("K-Means Clustering Results")

    tab_parent = ttk.Notebook(root)
    for cluster_index, data_points in clusters.items():
        tab = tk.Frame(tab_parent)
        tab_parent.add(tab, text=f"Cluster {cluster_index + 1}")
        display_table(tab, data_points)

    outliers_tab = tk.Frame(tab_parent)
    tab_parent.add(outliers_tab, text="Outliers")
    display_table(outliers_tab, detect_outliers(df), is_outlier=True)

    tab_parent.pack(expand=1, fill="both")

    root.mainloop()


def display_table(parent, data, is_outlier=False):
    if is_outlier:
        label_text = "Outlier Data Points"
    else:
        label_text = "Cluster Data Points"

    label = tk.Label(parent, text=label_text, font=("Helvetica", 12, "bold"))
    label.pack(pady=10)

    tree = ttk.Treeview(parent)
    tree["columns"] = ("1", "2")
    tree.column("#0", width=150, minwidth=150, stretch=tk.NO)
    tree.column("1", anchor=tk.W, width=200, minwidth=200, stretch=tk.NO)
    tree.column("2", anchor=tk.W, width=150, minwidth=150, stretch=tk.NO)

    tree.heading("#0", text="Movie Name", anchor=tk.W)
    tree.heading("1", text="Movie Name", anchor=tk.W)
    tree.heading("2", text="IMDB Rating", anchor=tk.W)

    for index, (movie, rating) in enumerate(data, start=1):
        tree.insert("", index, text=str(index), values=(movie, rating))

    tree.pack(expand=1, fill="both")


if __name__ == "__main__":
    main()
