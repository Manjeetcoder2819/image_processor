class SVGTemplate:
    def __init__(self):
        """Initialize SVG template generator"""
        pass
    
    def generate(self, shapes, width, height):
        """
        Generate SVG code for shapes
        
        Args:
            shapes: List of detected shapes
            width: Canvas width
            height: Canvas height
            
        Returns:
            SVG code as string
        """
        svg_elements = []
        
        # SVG header
        svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .shape {{ fill: none; stroke: #007bff; stroke-width: 2; }}
      .circle {{ fill: rgba(0, 123, 255, 0.1); }}
      .rectangle {{ fill: rgba(40, 167, 69, 0.1); }}
      .triangle {{ fill: rgba(255, 193, 7, 0.1); }}
      .polygon {{ fill: rgba(220, 53, 69, 0.1); }}
    </style>
  </defs>'''
        
        svg_elements.append(svg_header)
        
        for i, shape in enumerate(shapes):
            element = self._create_svg_element(shape, i)
            if element:
                svg_elements.append(f"  {element}")
        
        svg_elements.append("</svg>")
        
        return "\n".join(svg_elements)
    
    def _create_svg_element(self, shape, index):
        """Create SVG element for a specific shape"""
        shape_type = shape['type']
        
        if shape_type == 'circle':
            return self._create_circle(shape, index)
        elif shape_type in ['rectangle', 'square']:
            return self._create_rectangle(shape, index)
        elif shape_type == 'triangle':
            return self._create_triangle(shape, index)
        elif shape_type in ['polygon', 'quadrilateral', 'rhombus']:
            return self._create_polygon(shape, index)
        elif shape_type == 'ellipse':
            return self._create_ellipse(shape, index)
        
        return None
    
    def _create_circle(self, shape, index):
        """Create SVG circle element"""
        cx, cy = shape['center']
        radius = shape['radius']
        
        return f'''<circle cx="{cx}" cy="{cy}" r="{radius}" 
                    class="shape circle" id="circle-{index}"/>'''
    
    def _create_rectangle(self, shape, index):
        """Create SVG rectangle element"""
        x, y, w, h = shape['bounding_rect']
        
        return f'''<rect x="{x}" y="{y}" width="{w}" height="{h}" 
                   class="shape rectangle" id="rect-{index}"/>'''
    
    def _create_triangle(self, shape, index):
        """Create SVG triangle (polygon) element"""
        points = shape['points']
        points_str = " ".join([f"{p[0]},{p[1]}" for p in points])
        
        return f'''<polygon points="{points_str}" 
                   class="shape triangle" id="triangle-{index}"/>'''
    
    def _create_polygon(self, shape, index):
        """Create SVG polygon element"""
        points = shape.get('points', [])
        if not points:
            return None
            
        points_str = " ".join([f"{p[0]},{p[1]}" for p in points])
        
        return f'''<polygon points="{points_str}" 
                   class="shape polygon" id="polygon-{index}"/>'''
    
    def _create_ellipse(self, shape, index):
        """Create SVG ellipse element"""
        cx, cy = shape['center']
        radius = shape['radius']
        
        # For ellipse, we approximate with circle for now
        return f'''<ellipse cx="{cx}" cy="{cy}" rx="{radius}" ry="{radius * 0.7}" 
                   class="shape circle" id="ellipse-{index}"/>'''
