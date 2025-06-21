def get_mime_type(format_type):
    """
    Get MIME type for different file formats
    
    Args:
        format_type: File format (SVG, CSS, HTML)
        
    Returns:
        MIME type string
    """
    mime_types = {
        'SVG': 'image/svg+xml',
        'CSS': 'text/css',
        'HTML': 'text/html',
        'HTML+CSS+JS': 'text/html',
        'Java': 'text/x-java-source',
        'Python': 'text/x-python'
    }
    
    return mime_types.get(format_type.upper(), 'text/plain')

def validate_image_format(file_extension):
    """
    Validate if image format is supported
    
    Args:
        file_extension: File extension
        
    Returns:
        Boolean indicating if format is supported
    """
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    return file_extension.lower() in supported_formats

def calculate_shape_complexity(shape):
    """
    Calculate complexity score for a shape
    
    Args:
        shape: Shape dictionary
        
    Returns:
        Complexity score (0-1)
    """
    shape_type = shape['type']
    
    if shape_type == 'circle':
        # Circles are simple
        return 0.1
    elif shape_type in ['rectangle', 'square']:
        # Rectangles are simple
        return 0.2
    elif shape_type == 'triangle':
        # Triangles are moderately simple
        return 0.3
    elif shape_type in ['polygon', 'quadrilateral']:
        # Complexity based on number of vertices
        vertices = len(shape.get('points', []))
        return min(0.1 * vertices, 1.0)
    else:
        return 0.5

def format_shape_info(shape):
    """
    Format shape information for display
    
    Args:
        shape: Shape dictionary
        
    Returns:
        Formatted string with shape information
    """
    shape_type = shape['type']
    info_lines = [f"Type: {shape_type.title()}"]
    
    if 'center' in shape:
        info_lines.append(f"Center: ({shape['center'][0]}, {shape['center'][1]})")
    
    if 'area' in shape:
        info_lines.append(f"Area: {shape['area']:.0f} pixels²")
    
    if 'perimeter' in shape:
        info_lines.append(f"Perimeter: {shape['perimeter']:.1f} pixels")
    
    if shape_type == 'circle' and 'radius' in shape:
        info_lines.append(f"Radius: {shape['radius']} pixels")
        if 'circularity' in shape:
            info_lines.append(f"Circularity: {shape['circularity']:.2f}")
    
    if shape_type in ['rectangle', 'square'] and 'aspect_ratio' in shape:
        info_lines.append(f"Aspect Ratio: {shape['aspect_ratio']:.2f}")
        info_lines.append(f"Dimensions: {shape['width']}×{shape['height']} pixels")
    
    if shape_type == 'triangle' and 'triangle_type' in shape:
        info_lines.append(f"Triangle Type: {shape['triangle_type'].title()}")
    
    return "\n".join(info_lines)
