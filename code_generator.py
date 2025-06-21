from templates.svg_template import SVGTemplate
from templates.css_template import CSSTemplate
from templates.html_template import HTMLTemplate
from templates.java_template import JavaTemplate
from templates.python_template import PythonTemplate

class CodeGenerator:
    def __init__(self):
        """Initialize code generator with templates"""
        self.svg_template = SVGTemplate()
        self.css_template = CSSTemplate()
        self.html_template = HTMLTemplate()
        self.java_template = JavaTemplate()
        self.python_template = PythonTemplate()
    
    def generate_svg(self, shapes, image_shape):
        """
        Generate SVG code for detected shapes
        
        Args:
            shapes: List of detected shapes
            image_shape: Original image dimensions (height, width, channels)
            
        Returns:
            SVG code as string
        """
        height, width = image_shape[:2]
        return self.svg_template.generate(shapes, width, height)
    
    def generate_css(self, shapes, image_shape):
        """
        Generate CSS code for detected shapes
        
        Args:
            shapes: List of detected shapes
            image_shape: Original image dimensions
            
        Returns:
            CSS code as string
        """
        height, width = image_shape[:2]
        return self.css_template.generate(shapes, width, height)
    
    def generate_html(self, shapes, image_shape):
        """
        Generate HTML code for detected shapes
        
        Args:
            shapes: List of detected shapes
            image_shape: Original image dimensions
            
        Returns:
            HTML code as string
        """
        height, width = image_shape[:2]
        return self.html_template.generate(shapes, width, height)
    
    def generate_interactive_html(self, shapes, image_shape):
        """
        Generate interactive HTML with CSS and JavaScript
        
        Args:
            shapes: List of detected shapes
            image_shape: Original image dimensions
            
        Returns:
            Complete HTML code with CSS and JavaScript
        """
        height, width = image_shape[:2]
        return self.html_template.generate_interactive(shapes, width, height)
    
    def generate_java(self, shapes, image_shape):
        """
        Generate Java code for detected shapes
        
        Args:
            shapes: List of detected shapes
            image_shape: Original image dimensions
            
        Returns:
            Java code as string
        """
        height, width = image_shape[:2]
        return self.java_template.generate(shapes, width, height)
    
    def generate_python(self, shapes, image_shape):
        """
        Generate Python code for detected shapes
        
        Args:
            shapes: List of detected shapes
            image_shape: Original image dimensions
            
        Returns:
            Python code as string
        """
        height, width = image_shape[:2]
        return self.python_template.generate(shapes, width, height)
