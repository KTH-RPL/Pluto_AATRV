import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt

def generate_b_spline_path(path, num_points=100):
    """
    Generates a smooth path using B-spline interpolation through the given waypoints using splprep.
    
    Args:
        path (list): List of waypoints [(x1, y1), (x2, y2), ...].
        num_points (int): Total number of points to generate along the spline.
    
    Returns:
        list: Smoothed path as a list of (x, y) tuples.
    """
    # Extract x and y coordinates from path
    x = [node[0] for node in path]
    y = [node[1] for node in path]
    

    # # Use splprep to create a B-spline representation of the path
    # tck = si.splprep([x, y], s=0, k=3)  # Cubic B-spline with no smoothing (s=0)
    
    # # Generate points along the spline (smooth points)
    # x_smooth = np.linspace(0, 1, num_points)  # More points for smoothness
    # y_smooth = si.BSpline(*tck)(x_smooth)  # Evaluate the B-spline at the smooth points


    tck = si.splrep(x, y, s=1, k=3)
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = si.BSpline(*tck)(x_smooth)

    # Create smoothed path with interpolated points
    smoothed_path = [(x_smooth[i], y_smooth[i]) for i in range(num_points)]
    return smoothed_path, x_smooth, y_smooth

def get_equidistant_points_on_curve(path, num_points):
    """
    Get equidistant points along the B-spline curve.
    
    Args:
        path (list): Smoothed B-spline path.
        num_points (int): Number of equidistant points to sample from the B-spline.
    
    Returns:
        list: Equidistant points along the smooth B-spline path.
    """
    # Calculate the cumulative distances along the spline
    distances = [0.0]
    for i in range(1, len(path)):
        dist = np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
        distances.append(distances[-1] + dist)

    # Total path length
    total_length = distances[-1]

    # Equidistant points
    equidistant_points = []
    for i in range(num_points):
        target_dist = total_length * (i / (num_points - 1))
        # Find the closest point in the smoothed path
        for j in range(1, len(distances)):
            if distances[j] >= target_dist:
                t = (target_dist - distances[j-1]) / (distances[j] - distances[j-1])
                x = path[j-1][0] + t * (path[j][0] - path[j-1][0])
                y = path[j-1][1] + t * (path[j][1] - path[j-1][1])
                equidistant_points.append((x, y))
                break

    return equidistant_points


# Example usage
if __name__ == "__main__":
    # Example path with 8 waypoints
    path = [(0, 0), (2, 3), (4, 2), (6, 5), (8, 3), (10, 4), (12, 6), (14, 0)]
    path = get_equidistant_points_on_curve(path, num_points=100)
    
    # Generate smoothed path using B-spline with splprep
    smoothed_path, x_smooth, y_smooth = generate_b_spline_path(path, num_points=100)

    # Get equidistant points along the smoothed B-spline path
    equidistant_points = get_equidistant_points_on_curve(smoothed_path, num_points=100)

    # Plot the original, smoothed, and equidistant points
    original_x, original_y = zip(*path)
    smoothed_x, smoothed_y = zip(*[(p[0], p[1]) for p in smoothed_path])
    eq_x, eq_y = zip(*equidistant_points)

    plt.figure(figsize=(8, 6))
    
    # Plot the original points
    plt.plot(original_x, original_y, 'ro-', label="Original Path")
    
    # Plot the smoothed B-spline path
    plt.plot(smoothed_x, smoothed_y, 'b--', label="Smoothed Path (B-Spline)")
    
    # Plot the equidistant points
    plt.scatter(eq_x, eq_y, color="green", label="Equidistant Points", zorder=5)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("B-Spline Path Smoothing with Equidistant Points")
    plt.grid(True)
    plt.show()
