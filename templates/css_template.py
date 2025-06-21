class CSSTemplate:
    def __init__(self):
        """Initialize CSS template generator"""
        pass
    
    def generate(self, shapes, width, height):
        """
        Generate CSS code for shapes
        
        Args:
            shapes: List of detected shapes
            width: Canvas width
            height: Canvas height
            
        Returns:
            CSS code as string
        """
        css_rules = []
        
        # Base container styles
        css_rules.append(f'''.shapes-container {{
  position: relative;
  width: {width}px;
  height: {height}px;
  background-color: #f8f9fa;
  border: 1px solid #dee2e6;
  margin: 20px auto;
}}''')
        
        # Base shape styles
        css_rules.append('''.shape {
  position: absolute;
  border: 2px solid #007bff;
  box-sizing: border-box;
}''')
        
        for i, shape in enumerate(shapes):
            rule = self._create_css_rule(shape, i)
            if rule:
                css_rules.append(rule)
        
        return "\n\n".join(css_rules)
    
    def _create_css_rule(self, shape, index):
        """Create CSS rule for a specific shape"""
        shape_type = shape['type']
        
        if shape_type == 'circle':
            return self._create_circle_css(shape, index)
        elif shape_type in ['rectangle', 'square']:
            return self._create_rectangle_css(shape, index)
        elif shape_type == 'triangle':
            return self._create_triangle_css(shape, index)
        elif shape_type in ['polygon', 'quadrilateral', 'rhombus']:
            return self._create_polygon_css(shape, index)
        elif shape_type == 'ellipse':
            return self._create_ellipse_css(shape, index)
        
        return None
    
    def _create_circle_css(self, shape, index):
        """Create CSS for circle"""
        cx, cy = shape['center']
        radius = shape['radius']
        
        left = cx - radius
        top = cy - radius
        diameter = radius * 2
        
        return f'''.circle-{index} {{
  left: {left}px;
  top: {top}px;
  width: {diameter}px;
  height: {diameter}px;
  border-radius: 50%;
  background-color: rgba(0, 123, 255, 0.1);
}}'''
    
    def _create_rectangle_css(self, shape, index):
        """Create CSS for rectangle"""
        x, y, w, h = shape['bounding_rect']
        
        return f'''.rectangle-{index} {{
  left: {x}px;
  top: {y}px;
  width: {w}px;
  height: {h}px;
  background-color: rgba(40, 167, 69, 0.1);
}}'''
    
    def _create_triangle_css(self, shape, index):
        """Create CSS for triangle using clip-path"""
        points = shape['points']
        x, y, w, h = shape['bounding_rect']
        
        # Convert points to percentages relative to bounding rect
        clip_points = []
        for point in points:
            px = ((point[0] - x) / w) * 100 if w > 0 else 0
            py = ((point[1] - y) / h) * 100 if h > 0 else 0
            clip_points.append(f"{px}% {py}%")
        
        clip_path = "polygon(" + ", ".join(clip_points) + ")"
        
        return f'''.triangle-{index} {{
  left: {x}px;
  top: {y}px;
  width: {w}px;
  height: {h}px;
  background-color: rgba(255, 193, 7, 0.1);
  clip-path: {clip_path};
}}'''
    
    def _create_polygon_css(self, shape, index):
        """Create CSS for polygon using clip-path"""
        points = shape.get('points', [])
        if not points:
            return None
            
        x, y, w, h = shape['bounding_rect']
        
        # Convert points to percentages relative to bounding rect
        clip_points = []
        for point in points:
            px = ((point[0] - x) / w) * 100 if w > 0 else 0
            py = ((point[1] - y) / h) * 100 if h > 0 else 0
            clip_points.append(f"{px}% {py}%")
        
        clip_path = "polygon(" + ", ".join(clip_points) + ")"
        
        return f'''.polygon-{index} {{
  left: {x}px;
  top: {y}px;
  width: {w}px;
  height: {h}px;
  background-color: rgba(220, 53, 69, 0.1);
  clip-path: {clip_path};
}}'''
    
    def _create_ellipse_css(self, shape, index):
        """Create CSS for ellipse"""
        cx, cy = shape['center']
        radius = shape['radius']
        
        left = cx - radius
        top = cy - int(radius * 0.7)
        width = radius * 2
        height = int(radius * 1.4)
        
        return f'''.ellipse-{index} {{
  left: {left}px;
  top: {top}px;
  width: {width}px;
  height: {height}px;
  border-radius: 50%;
  background-color: rgba(0, 123, 255, 0.1);
}}'''
