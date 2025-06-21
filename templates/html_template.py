from .css_template import CSSTemplate

class HTMLTemplate:
    def __init__(self):
        """Initialize HTML template generator"""
        self.css_template = CSSTemplate()
    
    def generate(self, shapes, width, height):
        """
        Generate HTML code for shapes
        
        Args:
            shapes: List of detected shapes
            width: Canvas width
            height: Canvas height
            
        Returns:
            HTML code as string
        """
        # Generate CSS
        css_code = self.css_template.generate(shapes, width, height)
        
        # Generate HTML structure
        html_elements = []
        
        for i, shape in enumerate(shapes):
            element = self._create_html_element(shape, i)
            if element:
                html_elements.append(f"  {element}")
        
        html_body = "\n".join(html_elements)
        
        # Complete HTML document
        html_document = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detected Shapes</title>
    <style>
{css_code}
    </style>
</head>
<body>
    <div class="shapes-container">
{html_body}
    </div>
    
    <div style="margin: 20px; font-family: Arial, sans-serif;">
        <h2>Shape Information</h2>
        <ul>
{self._create_shape_list(shapes)}
        </ul>
    </div>
</body>
</html>'''
        
        return html_document
    
    def _create_html_element(self, shape, index):
        """Create HTML element for a specific shape"""
        shape_type = shape['type']
        class_name = f"{shape_type}-{index}"
        
        return f'<div class="shape {class_name}" title="{shape_type.title()} {index + 1}"></div>'
    
    def _create_shape_list(self, shapes):
        """Create HTML list of shape information"""
        list_items = []
        
        for i, shape in enumerate(shapes):
            shape_type = shape['type']
            info = []
            
            if shape_type == 'circle':
                info.append(f"Radius: {shape['radius']}px")
                info.append(f"Center: ({shape['center'][0]}, {shape['center'][1]})")
            elif shape_type in ['rectangle', 'square']:
                info.append(f"Size: {shape['width']}×{shape['height']}px")
                info.append(f"Aspect Ratio: {shape['aspect_ratio']:.2f}")
            elif shape_type == 'triangle':
                info.append(f"Type: {shape['triangle_type']}")
                info.append(f"Sides: {[f'{s:.1f}' for s in shape['sides']]}")
            elif shape_type in ['polygon', 'quadrilateral']:
                info.append(f"Vertices: {len(shape.get('points', []))}")
                info.append(f"Area: {shape['area']:.0f}px²")
            
            info_str = ", ".join(info) if info else "Basic shape"
            list_items.append(f"            <li><strong>{shape_type.title()} {i + 1}:</strong> {info_str}</li>")
        
        return "\n".join(list_items)
    
    def generate_interactive(self, shapes, width, height):
        """
        Generate interactive HTML with CSS and JavaScript
        
        Args:
            shapes: List of detected shapes
            width: Canvas width
            height: Canvas height
            
        Returns:
            Complete HTML with CSS and JavaScript as string
        """
        # Generate CSS
        css_code = self.css_template.generate(shapes, width, height)
        
        # Generate JavaScript for interactivity
        js_code = self._generate_javascript(shapes)
        
        # Generate HTML elements
        html_elements = []
        for i, shape in enumerate(shapes):
            element = self._create_interactive_element(shape, i)
            if element:
                html_elements.append(f"  {element}")
        
        html_body = "\n".join(html_elements)
        
        # Complete interactive HTML document
        html_document = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Shape Detection</title>
    <style>
{css_code}

/* Interactive styles */
.shape {{
    cursor: pointer;
    transition: all 0.3s ease;
    opacity: 0.8;
}}

.shape:hover {{
    opacity: 1;
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    z-index: 10;
}}

.shape.selected {{
    opacity: 1;
    transform: scale(1.1);
    box-shadow: 0 6px 12px rgba(0,123,255,0.4);
    z-index: 20;
}}

.info-panel {{
    position: fixed;
    top: 20px;
    right: 20px;
    width: 300px;
    background: white;
    border: 2px solid #007bff;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    font-family: Arial, sans-serif;
    display: none;
}}

.info-panel h3 {{
    margin: 0 0 10px 0;
    color: #007bff;
}}

.controls {{
    position: fixed;
    bottom: 20px;
    left: 20px;
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    font-family: Arial, sans-serif;
}}

.controls button {{
    margin: 5px;
    padding: 8px 16px;
    border: 1px solid #007bff;
    background: #007bff;
    color: white;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.3s;
}}

.controls button:hover {{
    background: #0056b3;
}}

.controls button.active {{
    background: #28a745;
    border-color: #28a745;
}}
    </style>
</head>
<body>
    <div class="shapes-container" id="shapesContainer">
{html_body}
    </div>
    
    <div class="info-panel" id="infoPanel">
        <h3>Shape Information</h3>
        <div id="shapeInfo"></div>
    </div>
    
    <div class="controls">
        <h4 style="margin: 0 0 10px 0;">Controls</h4>
        <button onclick="toggleAnimation()" id="animateBtn">Start Animation</button>
        <button onclick="highlightAll()">Highlight All</button>
        <button onclick="resetShapes()">Reset</button>
        <button onclick="toggleInfo()" id="infoBtn">Show Info Panel</button>
    </div>

    <script>
{js_code}
    </script>
</body>
</html>'''
        
        return html_document
    
    def _create_interactive_element(self, shape, index):
        """Create interactive HTML element for a specific shape"""
        shape_type = shape['type']
        class_name = f"{shape_type}-{index}"
        
        # Add data attributes for JavaScript interaction
        data_attrs = f'data-index="{index}" data-type="{shape_type}"'
        
        return f'<div class="shape {class_name}" {data_attrs} onclick="selectShape({index})" title="Click to select {shape_type.title()} {index + 1}"></div>'
    
    def _generate_javascript(self, shapes):
        """Generate JavaScript code for interactivity"""
        
        # Create shape data for JavaScript
        shape_data = []
        for i, shape in enumerate(shapes):
            shape_info = {
                'index': i,
                'type': shape['type'],
                'center': shape.get('center', [0, 0]),
                'area': shape.get('area', 0)
            }
            
            if shape['type'] == 'circle':
                shape_info['radius'] = shape.get('radius', 0)
                shape_info['circularity'] = shape.get('circularity', 0)
            elif shape['type'] in ['rectangle', 'square']:
                shape_info['width'] = shape.get('width', 0)
                shape_info['height'] = shape.get('height', 0)
                shape_info['aspect_ratio'] = shape.get('aspect_ratio', 1)
            elif shape['type'] == 'triangle':
                shape_info['triangle_type'] = shape.get('triangle_type', 'unknown')
                shape_info['sides'] = shape.get('sides', [])
            
            shape_data.append(shape_info)
        
        js_code = f'''
// Shape data
const shapeData = {shape_data};
let selectedShape = null;
let animationRunning = false;
let animationInterval = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {{
    console.log('Interactive shapes loaded:', shapeData.length);
}});

// Select shape function
function selectShape(index) {{
    // Remove previous selection
    if (selectedShape !== null) {{
        const prevElement = document.querySelector(`[data-index="${{selectedShape}}"]`);
        if (prevElement) prevElement.classList.remove('selected');
    }}
    
    // Select new shape
    selectedShape = index;
    const element = document.querySelector(`[data-index="${{index}}"]`);
    if (element) {{
        element.classList.add('selected');
        showShapeInfo(shapeData[index]);
    }}
}}

// Show shape information
function showShapeInfo(shape) {{
    const infoPanel = document.getElementById('infoPanel');
    const shapeInfo = document.getElementById('shapeInfo');
    
    let infoHTML = `
        <p><strong>Type:</strong> ${{shape.type.charAt(0).toUpperCase() + shape.type.slice(1)}}</p>
        <p><strong>Center:</strong> (${{shape.center[0]}}, ${{shape.center[1]}})</p>
        <p><strong>Area:</strong> ${{Math.round(shape.area)}} pixels²</p>
    `;
    
    if (shape.type === 'circle') {{
        infoHTML += `<p><strong>Radius:</strong> ${{shape.radius}} pixels</p>`;
        if (shape.circularity) {{
            infoHTML += `<p><strong>Circularity:</strong> ${{shape.circularity.toFixed(2)}}</p>`;
        }}
    }} else if (shape.type === 'rectangle' || shape.type === 'square') {{
        infoHTML += `<p><strong>Dimensions:</strong> ${{shape.width}}×${{shape.height}} pixels</p>`;
        infoHTML += `<p><strong>Aspect Ratio:</strong> ${{shape.aspect_ratio.toFixed(2)}}</p>`;
    }} else if (shape.type === 'triangle') {{
        infoHTML += `<p><strong>Type:</strong> ${{shape.triangle_type}}</p>`;
        if (shape.sides && shape.sides.length === 3) {{
            infoHTML += `<p><strong>Sides:</strong> ${{shape.sides.map(s => s.toFixed(1)).join(', ')}}</p>`;
        }}
    }}
    
    shapeInfo.innerHTML = infoHTML;
    infoPanel.style.display = 'block';
}}

// Toggle animation
function toggleAnimation() {{
    const btn = document.getElementById('animateBtn');
    
    if (animationRunning) {{
        clearInterval(animationInterval);
        animationRunning = false;
        btn.textContent = 'Start Animation';
        btn.classList.remove('active');
    }} else {{
        animationInterval = setInterval(animateShapes, 2000);
        animationRunning = true;
        btn.textContent = 'Stop Animation';
        btn.classList.add('active');
    }}
}}

// Animate shapes
function animateShapes() {{
    const shapes = document.querySelectorAll('.shape');
    let index = 0;
    
    function highlightNext() {{
        // Remove previous highlight
        shapes.forEach(shape => shape.classList.remove('selected'));
        
        // Highlight current shape
        if (index < shapes.length) {{
            shapes[index].classList.add('selected');
            const shapeIndex = parseInt(shapes[index].getAttribute('data-index'));
            showShapeInfo(shapeData[shapeIndex]);
            index++;
        }} else {{
            index = 0; // Reset to beginning
        }}
    }}
    
    highlightNext();
}}

// Highlight all shapes
function highlightAll() {{
    const shapes = document.querySelectorAll('.shape');
    shapes.forEach((shape, index) => {{
        setTimeout(() => {{
            shape.classList.add('selected');
            setTimeout(() => shape.classList.remove('selected'), 1000);
        }}, index * 200);
    }});
}}

// Reset all shapes
function resetShapes() {{
    const shapes = document.querySelectorAll('.shape');
    shapes.forEach(shape => shape.classList.remove('selected'));
    selectedShape = null;
    
    const infoPanel = document.getElementById('infoPanel');
    infoPanel.style.display = 'none';
    
    if (animationRunning) {{
        toggleAnimation();
    }}
}}

// Toggle info panel
function toggleInfo() {{
    const infoPanel = document.getElementById('infoPanel');
    const btn = document.getElementById('infoBtn');
    
    if (infoPanel.style.display === 'none' || infoPanel.style.display === '') {{
        infoPanel.style.display = 'block';
        btn.textContent = 'Hide Info Panel';
        btn.classList.add('active');
    }} else {{
        infoPanel.style.display = 'none';
        btn.textContent = 'Show Info Panel';
        btn.classList.remove('active');
    }}
}}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {{
    switch(e.key) {{
        case 'Escape':
            resetShapes();
            break;
        case ' ':
            e.preventDefault();
            toggleAnimation();
            break;
        case 'i':
        case 'I':
            toggleInfo();
            break;
        case 'h':
        case 'H':
            highlightAll();
            break;
    }}
}});
'''
        
        return js_code
