class PythonTemplate:
    def __init__(self):
        """Initialize Python template generator"""
        pass
    
    def generate(self, shapes, width, height):
        """
        Generate Python code for shapes using matplotlib/PIL
        
        Args:
            shapes: List of detected shapes
            width: Canvas width
            height: Canvas height
            
        Returns:
            Python code as string
        """
        
        python_code = f'''#!/usr/bin/env python3
"""
Generated Python code for shape detection results
This code reproduces the detected shapes with exact colors, sizes, and positions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Circle, Rectangle, Ellipse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys

class ShapeRenderer:
    def __init__(self, width={width}, height={height}):
        self.width = width
        self.height = height
        self.shapes_data = self._initialize_shapes()
    
    def _initialize_shapes(self):
        """Initialize shape data from detection results"""
        shapes = []
{self._generate_shape_data(shapes)}
        return shapes
    
    def render_with_matplotlib(self, save_path="detected_shapes.png", show_plot=True):
        """Render shapes using matplotlib"""
        fig, ax = plt.subplots(1, 1, figsize=(self.width/100, self.height/100), dpi=100)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match image coordinate system
        
        for i, shape_data in enumerate(self.shapes_data):
            self._draw_matplotlib_shape(ax, shape_data, i)
        
        ax.set_title('Detected Shapes Reproduction', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Shape visualization saved to: {{save_path}}")
        
        if show_plot:
            plt.show()
        
        return fig, ax
    
    def render_with_pil(self, save_path="detected_shapes_pil.png"):
        """Render shapes using PIL for exact pixel reproduction"""
        # Create image
        img = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        for i, shape_data in enumerate(self.shapes_data):
            self._draw_pil_shape(draw, shape_data, i, font)
        
        if save_path:
            img.save(save_path)
            print(f"Exact pixel reproduction saved to: {{save_path}}")
        
        return img
    
    def _draw_matplotlib_shape(self, ax, shape_data, index):
        """Draw shape using matplotlib"""
        shape_type = shape_data['type']
        color = tuple(c/255.0 for c in shape_data['fill_color'][:3])
        edge_color = tuple(c/255.0 for c in shape_data['border_color'][:3])
        
        if shape_type == 'circle':
            circle = Circle(
                (shape_data['center'][0], shape_data['center'][1]), 
                shape_data['radius'],
                facecolor=color, 
                edgecolor=edge_color, 
                linewidth=2,
                alpha=0.7
            )
            ax.add_patch(circle)
            
        elif shape_type in ['rectangle', 'square']:
            rect = Rectangle(
                (shape_data['x'], shape_data['y']), 
                shape_data['width'], shape_data['height'],
                facecolor=color, 
                edgecolor=edge_color, 
                linewidth=2,
                alpha=0.7
            )
            ax.add_patch(rect)
            
        elif shape_type == 'ellipse':
            ellipse = Ellipse(
                (shape_data['center'][0], shape_data['center'][1]),
                shape_data['width'], shape_data['height'],
                facecolor=color, 
                edgecolor=edge_color, 
                linewidth=2,
                alpha=0.7
            )
            ax.add_patch(ellipse)
            
        elif shape_type in ['triangle', 'polygon']:
            if 'points' in shape_data and shape_data['points']:
                polygon = Polygon(
                    shape_data['points'], 
                    facecolor=color, 
                    edgecolor=edge_color, 
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(polygon)
        
        # Add label
        label_x, label_y = shape_data.get('center', [shape_data.get('x', 0), shape_data.get('y', 0)])
        ax.text(label_x, label_y, f"{{shape_type}} {{index+1}}", 
               ha='center', va='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _draw_pil_shape(self, draw, shape_data, index, font):
        """Draw shape using PIL for exact reproduction"""
        shape_type = shape_data['type']
        fill_color = tuple(shape_data['fill_color'][:3])
        outline_color = tuple(shape_data['border_color'][:3])
        
        if shape_type == 'circle':
            center = shape_data['center']
            radius = shape_data['radius']
            bbox = [center[0] - radius, center[1] - radius, 
                   center[0] + radius, center[1] + radius]
            draw.ellipse(bbox, fill=fill_color, outline=outline_color, width=2)
            
        elif shape_type in ['rectangle', 'square']:
            bbox = [shape_data['x'], shape_data['y'], 
                   shape_data['x'] + shape_data['width'], 
                   shape_data['y'] + shape_data['height']]
            draw.rectangle(bbox, fill=fill_color, outline=outline_color, width=2)
            
        elif shape_type == 'ellipse':
            center = shape_data['center']
            w, h = shape_data['width'], shape_data['height']
            bbox = [center[0] - w//2, center[1] - h//2, 
                   center[0] + w//2, center[1] + h//2]
            draw.ellipse(bbox, fill=fill_color, outline=outline_color, width=2)
            
        elif shape_type in ['triangle', 'polygon']:
            if 'points' in shape_data and shape_data['points']:
                # Convert points to flat list for PIL
                point_list = []
                for point in shape_data['points']:
                    point_list.extend([point[0], point[1]])
                draw.polygon(point_list, fill=fill_color, outline=outline_color)
        
        # Add text label
        label_x, label_y = shape_data.get('center', [shape_data.get('x', 0), shape_data.get('y', 0)])
        text = f"{{shape_type}} {{index+1}}"
        
        # Get text size for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw text background
        text_bg = [label_x - text_width//2 - 2, label_y - text_height//2 - 2,
                  label_x + text_width//2 + 2, label_y + text_height//2 + 2]
        draw.rectangle(text_bg, fill=(255, 255, 255, 200))
        
        # Draw text
        draw.text((label_x - text_width//2, label_y - text_height//2), 
                 text, fill=(0, 0, 0), font=font)
    
    def generate_analysis_report(self):
        """Generate detailed analysis report"""
        report = f"""
Shape Detection Analysis Report
===============================
Canvas Size: {{self.width}} x {{self.height}} pixels
Total Shapes Detected: {{len(self.shapes_data)}}

Shape Details:
"""
        
        for i, shape in enumerate(self.shapes_data):
            report += f"""
Shape {{i+1}}: {{shape['type'].title()}}
  Position: {{shape.get('center', 'N/A')}}
  Area: {{shape.get('area', 0):.2f}} pixels²
  Color: RGB{{tuple(shape['fill_color'][:3])}} ({{shape.get('hex_color', 'N/A')}})
  Brightness: {{shape.get('brightness', 0):.1f}}
  Contrast: {{shape.get('contrast', 0):.1f}}"""
            
            if shape['type'] == 'circle':
                report += f"\\n  Radius: {{shape.get('radius', 0)}} pixels"
            elif shape['type'] in ['rectangle', 'square']:
                report += f"\\n  Dimensions: {{shape.get('width', 0)}}x{{shape.get('height', 0)}} pixels"
            elif shape['type'] == 'triangle':
                report += f"\\n  Triangle Type: {{shape.get('triangle_type', 'unknown')}}"
        
        return report
    
    def save_shape_data_json(self, filename="shape_data.json"):
        """Save shape data as JSON for further processing"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = []
        for shape in self.shapes_data:
            json_shape = {{}}
            for key, value in shape.items():
                if isinstance(value, np.ndarray):
                    json_shape[key] = value.tolist()
                else:
                    json_shape[key] = value
            json_data.append(json_shape)
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Shape data saved to: {{filename}}")

def main():
    """Main function to demonstrate shape rendering"""
    renderer = ShapeRenderer()
    
    print("=== Shape Detection Results ===")
    print(f"Canvas: {{renderer.width}}x{{renderer.height}} pixels")
    print(f"Shapes detected: {{len(renderer.shapes_data)}}")
    print()
    
    # Render with matplotlib
    print("Rendering with matplotlib...")
    renderer.render_with_matplotlib("shapes_matplotlib.png")
    
    # Render with PIL for exact reproduction
    print("Rendering with PIL for exact pixel reproduction...")
    renderer.render_with_pil("shapes_exact.png")
    
    # Generate analysis report
    print("Generating analysis report...")
    report = renderer.generate_analysis_report()
    print(report)
    
    # Save shape data
    renderer.save_shape_data_json("detected_shapes.json")
    
    print("\\nAll outputs generated successfully!")
    print("Files created:")
    print("  - shapes_matplotlib.png (matplotlib visualization)")
    print("  - shapes_exact.png (exact pixel reproduction)")
    print("  - detected_shapes.json (shape data)")

if __name__ == "__main__":
    main()
'''
        
        return python_code
    
    def _generate_shape_data(self, shapes):
        """Generate Python shape data initialization"""
        shape_init_code = []
        
        for i, shape in enumerate(shapes):
            shape_type = shape['type']
            
            # Get colors
            fill_color = shape.get('dominant_color', [128, 128, 128])
            border_color = [max(0, c - 40) for c in fill_color[:3]]  # Darker border
            
            # Get position and size
            center = shape.get('center', [0, 0])
            area = shape.get('area', 0)
            perimeter = shape.get('perimeter', 0)
            
            shape_dict = {
                'type': shape_type,
                'center': center,
                'area': area,
                'perimeter': perimeter,
                'fill_color': fill_color,
                'border_color': border_color,
                'hex_color': shape.get('hex_color', '#808080'),
                'brightness': shape.get('brightness', 128),
                'contrast': shape.get('contrast', 0)
            }
            
            if shape_type == 'circle':
                shape_dict.update({
                    'radius': shape.get('radius', 50),
                    'circularity': shape.get('circularity', 1.0)
                })
            elif shape_type in ['rectangle', 'square']:
                x, y, w, h = shape.get('bounding_rect', (0, 0, 100, 100))
                shape_dict.update({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'aspect_ratio': shape.get('aspect_ratio', 1.0)
                })
            elif shape_type == 'ellipse':
                shape_dict.update({
                    'width': shape.get('radius', 50) * 2,
                    'height': int(shape.get('radius', 50) * 1.4),
                    'major_axis': shape.get('major_axis', 50),
                    'minor_axis': shape.get('minor_axis', 35)
                })
            
            # Add points for polygons
            if 'points' in shape and shape['points']:
                shape_dict['points'] = shape['points']
            
            # Add triangle-specific data
            if shape_type == 'triangle':
                shape_dict.update({
                    'triangle_type': shape.get('triangle_type', 'scalene'),
                    'angles': shape.get('angles', [60, 60, 60]),
                    'sides': shape.get('sides', [100, 100, 100])
                })
            
            # Convert to Python dict format
            dict_str = self._dict_to_python_string(shape_dict)
            shape_init_code.append(f"        shapes.append({dict_str})")
        
        return '\n'.join(shape_init_code)
    
    def _dict_to_python_string(self, d, indent=0):
        """Convert dictionary to properly formatted Python string"""
        items = []
        for key, value in d.items():
            if isinstance(value, str):
                items.append(f"'{key}': '{value}'")
            elif isinstance(value, list):
                if value and isinstance(value[0], list):
                    # List of points
                    points_str = str(value).replace("'", "")
                    items.append(f"'{key}': {points_str}")
                else:
                    items.append(f"'{key}': {value}")
            else:
                items.append(f"'{key}': {value}")
        
        return "{\n            " + ",\n            ".join(items) + "\n        }"