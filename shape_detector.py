import cv2
import numpy as np
import math

class ShapeDetector:
    def __init__(self, min_area=1000, epsilon_factor=0.02):
        """
        Initialize shape detector with parameters
        
        Args:
            min_area: Minimum area for shape detection
            epsilon_factor: Factor for contour approximation
        """
        self.min_area = min_area
        self.epsilon_factor = epsilon_factor
    
    def detect_shapes(self, image):
        """
        Detect shapes in the given image
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detected shapes with their properties
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Approximate contour to polygon
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Classify shape
            shape_info = self._classify_shape(approx, area, contour)
            shape_info.update({
                'contour': contour,
                'approx': approx,
                'area': area,
                'center': (cx, cy),
                'bounding_rect': (x, y, w, h),
                'perimeter': cv2.arcLength(contour, True)
            })
            
            shapes.append(shape_info)
        
        return shapes
    
    def _classify_shape(self, approx, area, contour):
        """
        Classify shape based on contour approximation
        
        Args:
            approx: Approximated contour
            area: Contour area
            contour: Original contour
            
        Returns:
            Dictionary with shape information
        """
        vertices = len(approx)
        
        if vertices == 3:
            return self._analyze_triangle(approx)
        
        elif vertices == 4:
            return self._analyze_quadrilateral(approx)
        
        elif vertices > 10:
            # Check if it's a circle
            return self._analyze_circle(contour, area)
        
        else:
            # Polygon
            return {
                'type': 'polygon',
                'vertices': vertices,
                'points': approx.reshape(-1, 2).tolist()
            }
    
    def _analyze_triangle(self, approx):
        """Analyze triangle properties"""
        points = approx.reshape(-1, 2)
        
        # Calculate side lengths
        sides = []
        for i in range(3):
            p1 = points[i]
            p2 = points[(i + 1) % 3]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(length)
        
        # Classify triangle type
        sides.sort()
        if abs(sides[0] - sides[1]) < 10 and abs(sides[1] - sides[2]) < 10:
            triangle_type = "equilateral"
        elif abs(sides[0] - sides[1]) < 10 or abs(sides[1] - sides[2]) < 10:
            triangle_type = "isosceles"
        else:
            triangle_type = "scalene"
        
        return {
            'type': 'triangle',
            'triangle_type': triangle_type,
            'points': points.tolist(),
            'sides': sides
        }
    
    def _analyze_quadrilateral(self, approx):
        """Analyze quadrilateral properties"""
        points = approx.reshape(-1, 2)
        
        # Calculate side lengths
        sides = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(length)
        
        # Calculate angles
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        
        # Classify quadrilateral
        if 0.95 <= aspect_ratio <= 1.05:
            if self._is_rectangle(points):
                shape_type = "square"
            else:
                shape_type = "rhombus"
        else:
            if self._is_rectangle(points):
                shape_type = "rectangle"
            else:
                shape_type = "quadrilateral"
        
        return {
            'type': shape_type,
            'points': points.tolist(),
            'sides': sides,
            'aspect_ratio': aspect_ratio,
            'width': w,
            'height': h
        }
    
    def _analyze_circle(self, contour, area):
        """Analyze circle properties"""
        # Fit circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        
        # If circularity is close to 1, it's likely a circle
        if circularity > 0.7:
            return {
                'type': 'circle',
                'center': (int(x), int(y)),
                'radius': int(radius),
                'circularity': circularity
            }
        else:
            return {
                'type': 'ellipse',
                'center': (int(x), int(y)),
                'radius': int(radius),
                'circularity': circularity
            }
    
    def _is_rectangle(self, points):
        """Check if four points form a rectangle"""
        # Calculate all angles
        angles = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            
            # Vectors
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(np.degrees(angle))
        
        # Check if all angles are close to 90 degrees
        return all(abs(angle - 90) < 15 for angle in angles)
    
    def draw_contours(self, image, shapes):
        """
        Draw detected shapes on image
        
        Args:
            image: Input image
            shapes: List of detected shapes
            
        Returns:
            Image with drawn contours
        """
        result = image.copy()
        
        for i, shape in enumerate(shapes):
            # Different colors for different shapes
            color = self._get_shape_color(shape['type'])
            
            # Draw contour
            cv2.drawContours(result, [shape['contour']], -1, color, 3)
            
            # Draw center point
            cv2.circle(result, shape['center'], 5, (0, 0, 255), -1)
            
            # Add label
            cv2.putText(result, f"{shape['type']} {i+1}", 
                       (shape['center'][0] - 30, shape['center'][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
    
    def _get_shape_color(self, shape_type):
        """Get color for shape type"""
        colors = {
            'circle': (0, 255, 0),      # Green
            'ellipse': (0, 255, 255),   # Cyan
            'rectangle': (255, 0, 0),   # Blue
            'square': (255, 0, 255),    # Magenta
            'triangle': (0, 165, 255),  # Orange
            'polygon': (255, 255, 0),   # Yellow
            'quadrilateral': (128, 0, 128), # Purple
            'rhombus': (255, 20, 147)   # Deep Pink
        }
        return colors.get(shape_type, (128, 128, 128))
