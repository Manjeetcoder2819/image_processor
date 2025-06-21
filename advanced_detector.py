import cv2
import numpy as np
import math
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

class AdvancedShapeDetector:
    def __init__(self, min_area=100, epsilon_factor=0.02):
        """
        Advanced shape detector for precise shape extraction from any image/video
        
        Args:
            min_area: Minimum area for shape detection
            epsilon_factor: Factor for contour approximation
        """
        self.min_area = min_area
        self.epsilon_factor = epsilon_factor
        
    def detect_shapes_advanced(self, image):
        """
        Advanced shape detection with multiple algorithms for better accuracy
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detected shapes with precise properties
        """
        try:
            # Start with basic preprocessing
            processed_image = self._preprocess_image(image)
            
            # Multiple detection methods with error handling
            all_shapes = []
            
            # Method 1: Enhanced edge detection
            try:
                shapes_edges = self._detect_with_enhanced_edges(processed_image)
                all_shapes.extend(shapes_edges)
            except Exception as e:
                print(f"Edge detection failed: {e}")
            
            # Method 2: Adaptive threshold
            try:
                shapes_adaptive = self._detect_with_adaptive_threshold(processed_image)
                all_shapes.extend(shapes_adaptive)
            except Exception as e:
                print(f"Adaptive threshold failed: {e}")
            
            # Method 3: Color-based detection
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                shapes_color = self._detect_with_improved_color_segmentation(rgb_image)
                all_shapes.extend(shapes_color)
            except Exception as e:
                print(f"Color segmentation failed: {e}")
            
            # Method 4: Contour-based detection
            try:
                shapes_contour = self._detect_with_contour_analysis(processed_image)
                all_shapes.extend(shapes_contour)
            except Exception as e:
                print(f"Contour analysis failed: {e}")
            
            # Filter and merge results
            if all_shapes:
                filtered_shapes = self._filter_and_merge_shapes(all_shapes)
                
                # Extract visual properties with error handling
                for shape in filtered_shapes:
                    try:
                        shape.update(self._extract_visual_properties(image, shape))
                    except Exception as e:
                        print(f"Visual property extraction failed: {e}")
                        # Add default properties
                        shape.update({
                            'dominant_color': [128, 128, 128],
                            'hex_color': '#808080',
                            'brightness': 128,
                            'contrast': 0,
                            'pixel_count': int(shape.get('area', 100))
                        })
                
                return filtered_shapes
            else:
                # Fallback to basic detection if advanced methods fail
                return self._basic_fallback_detection(image)
                
        except Exception as e:
            print(f"Advanced detection failed, using fallback: {e}")
            return self._basic_fallback_detection(image)
    
    def _preprocess_image(self, image):
        """Preprocess image for better detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def _detect_with_enhanced_edges(self, image):
        """Enhanced edge-based detection"""
        # Multiple edge detection approaches
        edges_list = []
        
        # Canny with multiple parameters
        for lower, upper in [(50, 150), (30, 100), (100, 200)]:
            edges = cv2.Canny(image, lower, upper)
            edges_list.append(edges)
        
        # Combine edges
        combined_edges = np.zeros_like(edges_list[0])
        for edges in edges_list:
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        # Clean up edges
        kernel = np.ones((3,3), np.uint8)
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return self._process_contours(contours, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), "enhanced_edges")
    
    def _detect_with_adaptive_threshold(self, image):
        """Improved adaptive threshold-based detection"""
        # Handle both grayscale and color images
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image
        else:
            gray = image
        
        shapes = []
        
        # Multiple adaptive threshold approaches
        for block_size in [11, 15, 21]:
            for c_value in [2, 5, 10]:
                try:
                    adaptive_thresh = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value
                    )
                    
                    # Clean up with morphological operations
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
                    
                    # Find contours
                    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Process contours
                    processed_shapes = self._process_contours(contours, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), f"adaptive_{block_size}_{c_value}")
                    shapes.extend(processed_shapes)
                    
                except Exception as e:
                    continue
        
        return shapes
    
    def _detect_with_improved_color_segmentation(self, rgb_image):
        """Improved color-based segmentation for shape detection"""
        shapes = []
        
        try:
            # Convert to multiple color spaces for better detection
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
            
            # Enhanced color ranges with more coverage
            color_ranges = [
                # Red ranges
                (np.array([0, 30, 30]), np.array([10, 255, 255])),
                (np.array([170, 30, 30]), np.array([180, 255, 255])),
                # Blue ranges
                (np.array([100, 30, 30]), np.array([130, 255, 255])),
                # Green ranges
                (np.array([35, 30, 30]), np.array([85, 255, 255])),
                # Yellow ranges
                (np.array([15, 30, 30]), np.array([35, 255, 255])),
                # Purple/Magenta ranges
                (np.array([130, 30, 30]), np.array([170, 255, 255])),
                # Orange ranges
                (np.array([5, 50, 50]), np.array([20, 255, 255])),
                # Cyan ranges
                (np.array([85, 30, 30]), np.array([100, 255, 255])),
            ]
            
            for i, (lower, upper) in enumerate(color_ranges):
                try:
                    # Create mask
                    mask = cv2.inRange(hsv, lower, upper)
                    
                    # Improve mask quality
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                    
                    # Remove small noise
                    mask = cv2.medianBlur(mask, 5)
                    
                    # Find contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        processed_shapes = self._process_contours(contours, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), f"color_{i}")
                        shapes.extend(processed_shapes)
                        
                except Exception as e:
                    continue
            
            # Additional intensity-based segmentation
            try:
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                
                # OTSU thresholding for intensity-based shapes
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Also try inverse
                _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                for binary_img, name in [(binary, "otsu"), (binary_inv, "otsu_inv")]:
                    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        processed_shapes = self._process_contours(contours, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), name)
                        shapes.extend(processed_shapes)
                        
            except Exception as e:
                pass
            
        except Exception as e:
            print(f"Color segmentation error: {e}")
        
        return shapes
    
    def _detect_with_contour_analysis(self, image):
        """Comprehensive contour-based detection"""
        shapes = []
        
        try:
            # Multiple threshold approaches
            thresholding_methods = [
                # Standard binary threshold with different values
                (cv2.THRESH_BINARY, [127, 100, 150, 80, 180]),
                (cv2.THRESH_BINARY_INV, [127, 100, 150, 80, 180]),
            ]
            
            for thresh_type, values in thresholding_methods:
                for thresh_val in values:
                    try:
                        _, binary = cv2.threshold(image, thresh_val, 255, thresh_type)
                        
                        # Find contours
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            processed_shapes = self._process_contours(contours, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), f"threshold_{thresh_val}")
                            shapes.extend(processed_shapes)
                            
                    except Exception as e:
                        continue
            
        except Exception as e:
            print(f"Contour analysis error: {e}")
        
        return shapes
    
    def _basic_fallback_detection(self, image):
        """Basic fallback detection when advanced methods fail"""
        try:
            # Simple grayscale conversion and basic threshold
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Basic threshold
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shapes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    # Basic shape info
                    x, y, w, h = cv2.boundingRect(contour)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w//2, y + h//2
                    
                    shape = {
                        'type': 'shape',
                        'contour': contour,
                        'area': area,
                        'center': (cx, cy),
                        'bounding_rect': (x, y, w, h),
                        'perimeter': cv2.arcLength(contour, True),
                        'detection_method': 'fallback',
                        'dominant_color': [128, 128, 128],
                        'hex_color': '#808080',
                        'brightness': 128,
                        'contrast': 0,
                        'pixel_count': int(area)
                    }
                    shapes.append(shape)
            
            return shapes
            
        except Exception as e:
            print(f"Fallback detection failed: {e}")
            return []
    
    def _process_contours(self, contours, image, method):
        """Process contours and extract shape information with error handling"""
        shapes = []
        
        for contour in contours:
            try:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                # Check if contour is valid
                if len(contour) < 3:
                    continue
                
                # Approximate contour to polygon
                epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center with error handling
                try:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w//2, y + h//2
                except:
                    cx, cy = x + w//2, y + h//2
                
                # Classify shape with advanced analysis
                try:
                    shape_info = self._classify_shape_advanced(approx, area, contour, image)
                except Exception as e:
                    # Fallback to basic classification
                    vertices = len(approx)
                    if vertices == 3:
                        shape_type = 'triangle'
                    elif vertices == 4:
                        shape_type = 'rectangle'
                    elif vertices > 8:
                        shape_type = 'circle'
                    else:
                        shape_type = 'polygon'
                    
                    shape_info = {
                        'type': shape_type,
                        'points': approx.reshape(-1, 2).tolist() if len(approx) > 0 else []
                    }
                
                # Add common properties
                shape_info.update({
                    'contour': contour,
                    'approx': approx,
                    'area': float(area),
                    'center': (int(cx), int(cy)),
                    'bounding_rect': (int(x), int(y), int(w), int(h)),
                    'perimeter': float(cv2.arcLength(contour, True)),
                    'detection_method': method
                })
                
                shapes.append(shape_info)
                
            except Exception as e:
                # Skip problematic contours
                continue
        
        return shapes
    
    def _classify_shape_advanced(self, approx, area, contour, image):
        """Advanced shape classification with more precision and error handling"""
        try:
            vertices = len(approx)
            
            # Calculate additional geometric properties with error handling
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Fit ellipse for better circle/ellipse detection
            aspect_ratio_ellipse = 1
            angle = 0
            ellipse = None
            
            try:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (center, axes, angle) = ellipse
                    aspect_ratio_ellipse = max(axes) / min(axes) if min(axes) > 0 else 1
            except:
                pass
            
            # More precise shape classification with better thresholds
            if vertices == 3:
                return self._analyze_triangle_advanced(approx, area)
            elif vertices == 4:
                return self._analyze_quadrilateral_advanced(approx, area)
            elif vertices > 8 and circularity > 0.7:
                if aspect_ratio_ellipse < 1.3:
                    return self._analyze_circle_advanced(contour, area, ellipse)
                else:
                    return self._analyze_ellipse_advanced(contour, area, ellipse)
            elif 5 <= vertices <= 8:
                return self._analyze_polygon_advanced(approx, area, vertices)
            else:
                return self._analyze_complex_shape(contour, area)
                
        except Exception as e:
            # Fallback classification
            vertices = len(approx) if approx is not None else 0
            
            if vertices == 3:
                return {'type': 'triangle', 'points': approx.reshape(-1, 2).tolist()}
            elif vertices == 4:
                return {'type': 'rectangle', 'points': approx.reshape(-1, 2).tolist()}
            elif vertices > 8:
                return {'type': 'circle', 'center': (0, 0), 'radius': int(math.sqrt(area / math.pi))}
            else:
                return {'type': 'polygon', 'vertices': vertices, 'points': approx.reshape(-1, 2).tolist() if approx is not None else []}
    
    def _analyze_triangle_advanced(self, approx, area):
        """Advanced triangle analysis"""
        points = approx.reshape(-1, 2)
        
        # Calculate sides and angles
        sides = []
        angles = []
        for i in range(3):
            p1 = points[i]
            p2 = points[(i + 1) % 3]
            p3 = points[(i + 2) % 3]
            
            # Side length
            side_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(side_length)
            
            # Angle calculation
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            angles.append(angle)
        
        # Classify triangle type
        sides_sorted = sorted(sides)
        if abs(sides_sorted[0] - sides_sorted[1]) < 5 and abs(sides_sorted[1] - sides_sorted[2]) < 5:
            triangle_type = "equilateral"
        elif abs(sides_sorted[0] - sides_sorted[1]) < 5 or abs(sides_sorted[1] - sides_sorted[2]) < 5:
            triangle_type = "isosceles"
        elif any(abs(angle - 90) < 5 for angle in angles):
            triangle_type = "right"
        else:
            triangle_type = "scalene"
        
        return {
            'type': 'triangle',
            'triangle_type': triangle_type,
            'points': points.tolist(),
            'sides': sides,
            'angles': angles,
            'is_acute': all(angle < 90 for angle in angles),
            'is_obtuse': any(angle > 90 for angle in angles)
        }
    
    def _analyze_quadrilateral_advanced(self, approx, area):
        """Advanced quadrilateral analysis"""
        points = approx.reshape(-1, 2)
        
        # Calculate all side lengths
        sides = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(length)
        
        # Calculate angles
        angles = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            angles.append(angle)
        
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h > 0 else 1
        
        # Classify quadrilateral
        is_rectangle = all(abs(angle - 90) < 10 for angle in angles)
        sides_equal = all(abs(sides[i] - sides[(i+1)%4]) < 10 for i in range(4))
        opposite_sides_equal = (abs(sides[0] - sides[2]) < 10 and abs(sides[1] - sides[3]) < 10)
        
        if is_rectangle:
            if 0.9 <= aspect_ratio <= 1.1 and sides_equal:
                shape_type = "square"
            else:
                shape_type = "rectangle"
        elif sides_equal:
            shape_type = "rhombus"
        elif opposite_sides_equal:
            shape_type = "parallelogram"
        else:
            shape_type = "quadrilateral"
        
        return {
            'type': shape_type,
            'points': points.tolist(),
            'sides': sides,
            'angles': angles,
            'aspect_ratio': aspect_ratio,
            'width': w,
            'height': h,
            'is_regular': sides_equal and is_rectangle
        }
    
    def _analyze_circle_advanced(self, contour, area, ellipse):
        """Advanced circle analysis"""
        (center, axes, angle) = ellipse
        radius = (axes[0] + axes[1]) / 4  # Average radius
        
        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        return {
            'type': 'circle',
            'center': (int(center[0]), int(center[1])),
            'radius': int(radius),
            'circularity': circularity,
            'fitted_ellipse': ellipse,
            'is_perfect_circle': circularity > 0.9
        }
    
    def _analyze_ellipse_advanced(self, contour, area, ellipse):
        """Advanced ellipse analysis"""
        (center, axes, angle) = ellipse
        major_axis = max(axes) / 2
        minor_axis = min(axes) / 2
        
        return {
            'type': 'ellipse',
            'center': (int(center[0]), int(center[1])),
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'angle': angle,
            'eccentricity': math.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0,
            'fitted_ellipse': ellipse
        }
    
    def _analyze_polygon_advanced(self, approx, area, vertices):
        """Advanced polygon analysis"""
        points = approx.reshape(-1, 2)
        
        # Check if it's a regular polygon
        center = np.mean(points, axis=0)
        distances = [np.linalg.norm(point - center) for point in points]
        is_regular = all(abs(d - distances[0]) < 10 for d in distances)
        
        # Calculate internal angles
        angles = []
        for i in range(vertices):
            p1 = points[i]
            p2 = points[(i + 1) % vertices]
            p3 = points[(i + 2) % vertices]
            
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            angles.append(angle)
        
        # Specific polygon names
        polygon_names = {
            5: "pentagon",
            6: "hexagon", 
            7: "heptagon",
            8: "octagon",
            9: "nonagon",
            10: "decagon"
        }
        
        polygon_type = polygon_names.get(vertices, f"{vertices}-gon")
        if is_regular:
            polygon_type = f"regular {polygon_type}"
        
        return {
            'type': 'polygon',
            'polygon_type': polygon_type,
            'vertices': vertices,
            'points': points.tolist(),
            'is_regular': is_regular,
            'angles': angles,
            'center': center.tolist()
        }
    
    def _analyze_complex_shape(self, contour, area):
        """Analyze complex or irregular shapes"""
        # Calculate convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Calculate extent
        x, y, w, h = cv2.boundingRect(contour)
        extent = area / (w * h) if w * h > 0 else 0
        
        return {
            'type': 'complex_shape',
            'solidity': solidity,
            'extent': extent,
            'convex_hull': hull,
            'is_convex': cv2.isContourConvex(contour),
            'complexity_score': 1 - solidity
        }
    
    def _extract_visual_properties(self, image, shape):
        """Extract color, texture, and visual properties with error handling"""
        try:
            # Create mask for the shape
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Ensure contour is valid
            contour = shape.get('contour')
            if contour is None or len(contour) < 3:
                raise ValueError("Invalid contour")
            
            cv2.fillPoly(mask, [contour], 255)
            
            # Extract colors
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            
            # Get pixels within the mask
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) == 0:
                raise ValueError("No pixels in mask")
            
            # Extract pixel values
            pixels = image[mask_indices]
            
            if len(pixels) > 0:
                # Calculate dominant color (average)
                dominant_color = np.mean(pixels, axis=0).astype(int)
                
                # Ensure RGB format
                if len(dominant_color) == 3:
                    # Convert BGR to RGB for display
                    dominant_color_rgb = [int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0])]
                else:
                    dominant_color_rgb = [128, 128, 128]
                
                # Convert to hex
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    dominant_color_rgb[0], dominant_color_rgb[1], dominant_color_rgb[2]
                )
                
                # Calculate brightness and contrast
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                gray_pixels = gray_image[mask_indices]
                
                brightness = float(np.mean(gray_pixels)) if len(gray_pixels) > 0 else 128.0
                contrast = float(np.std(gray_pixels)) if len(gray_pixels) > 0 else 0.0
                
            else:
                dominant_color_rgb = [128, 128, 128]
                hex_color = "#808080"
                brightness = 128.0
                contrast = 0.0
            
            return {
                'dominant_color': dominant_color_rgb,
                'hex_color': hex_color,
                'brightness': brightness,
                'contrast': contrast,
                'pixel_count': int(np.sum(mask > 0))
            }
            
        except Exception as e:
            # Return default values on error
            return {
                'dominant_color': [128, 128, 128],
                'hex_color': '#808080',
                'brightness': 128.0,
                'contrast': 0.0,
                'pixel_count': int(shape.get('area', 100))
            }
    
    def _filter_and_merge_shapes(self, shapes):
        """Filter duplicates and merge similar shapes"""
        if not shapes:
            return []
        
        filtered = []
        
        for shape in shapes:
            # Check if this shape is similar to any existing shape
            is_duplicate = False
            
            for existing in filtered:
                # Check center distance
                center_dist = np.linalg.norm(
                    np.array(shape['center']) - np.array(existing['center'])
                )
                
                # Check area similarity
                area_ratio = min(shape['area'], existing['area']) / max(shape['area'], existing['area'])
                
                # If shapes are very similar, skip this one
                if center_dist < 20 and area_ratio > 0.8 and shape['type'] == existing['type']:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(shape)
        
        # Sort by area (largest first)
        filtered.sort(key=lambda x: x['area'], reverse=True)
        
        return filtered
    
    def draw_advanced_contours(self, image, shapes):
        """Draw shapes with advanced visualization"""
        result = image.copy()
        
        for i, shape in enumerate(shapes):
            # Use shape-specific colors
            color = self._get_advanced_color(shape['type'])
            
            # Draw filled shape with transparency
            overlay = result.copy()
            cv2.fillPoly(overlay, [shape['contour']], color)
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
            
            # Draw contour
            cv2.drawContours(result, [shape['contour']], -1, color, 2)
            
            # Draw center point
            cv2.circle(result, shape['center'], 5, (255, 255, 255), -1)
            cv2.circle(result, shape['center'], 5, color, 2)
            
            # Add detailed label
            label = f"{shape['type']} {i+1}"
            if shape['type'] == 'circle':
                label += f" (r={shape.get('radius', 0)})"
            elif shape['type'] in ['rectangle', 'square']:
                label += f" ({shape.get('width', 0)}x{shape.get('height', 0)})"
            
            # Position label
            x, y = shape['center']
            cv2.putText(result, label, (x - 40, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(result, label, (x - 40, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def _get_advanced_color(self, shape_type):
        """Get advanced colors for different shape types"""
        colors = {
            'circle': (0, 255, 0),           # Green
            'ellipse': (0, 255, 255),        # Cyan
            'rectangle': (255, 0, 0),        # Blue
            'square': (255, 0, 255),         # Magenta
            'triangle': (0, 165, 255),       # Orange
            'polygon': (255, 255, 0),        # Yellow
            'quadrilateral': (128, 0, 128),  # Purple
            'rhombus': (255, 20, 147),       # Deep Pink
            'parallelogram': (50, 205, 50),  # Lime Green
            'complex_shape': (169, 169, 169) # Dark Gray
        }
        return colors.get(shape_type, (128, 128, 128))