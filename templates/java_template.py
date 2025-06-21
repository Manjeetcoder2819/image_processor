class JavaTemplate:
    def __init__(self):
        """Initialize Java template generator"""
        pass
    
    def generate(self, shapes, width, height):
        """
        Generate Java code for shapes using Java Swing/AWT
        
        Args:
            shapes: List of detected shapes
            width: Canvas width
            height: Canvas height
            
        Returns:
            Java code as string
        """
        # Generate main class
        java_code = f'''import java.awt.*;
import java.awt.geom.*;
import javax.swing.*;
import java.util.ArrayList;
import java.util.List;

public class ShapeRenderer extends JPanel {{
    private static final int CANVAS_WIDTH = {width};
    private static final int CANVAS_HEIGHT = {height};
    private List<ShapeData> shapes;
    
    public ShapeRenderer() {{
        setPreferredSize(new Dimension(CANVAS_WIDTH, CANVAS_HEIGHT));
        setBackground(Color.WHITE);
        initializeShapes();
    }}
    
    private void initializeShapes() {{
        shapes = new ArrayList<>();
{self._generate_shape_data(shapes)}
    }}
    
    @Override
    protected void paintComponent(Graphics g) {{
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        
        for (ShapeData shapeData : shapes) {{
            drawShape(g2d, shapeData);
        }}
    }}
    
    private void drawShape(Graphics2D g2d, ShapeData shapeData) {{
        // Set color
        g2d.setColor(shapeData.fillColor);
        
        // Set stroke
        g2d.setStroke(new BasicStroke(2.0f));
        
        switch (shapeData.type) {{
            case "circle":
                drawCircle(g2d, shapeData);
                break;
            case "rectangle":
            case "square":
                drawRectangle(g2d, shapeData);
                break;
            case "triangle":
                drawTriangle(g2d, shapeData);
                break;
            case "polygon":
                drawPolygon(g2d, shapeData);
                break;
            case "ellipse":
                drawEllipse(g2d, shapeData);
                break;
            default:
                drawPolygon(g2d, shapeData);
        }}
        
        // Draw border
        g2d.setColor(shapeData.borderColor);
        switch (shapeData.type) {{
            case "circle":
                g2d.drawOval(shapeData.x - shapeData.width/2, shapeData.y - shapeData.height/2, 
                           shapeData.width, shapeData.height);
                break;
            case "rectangle":
            case "square":
                g2d.drawRect(shapeData.x, shapeData.y, shapeData.width, shapeData.height);
                break;
            default:
                if (shapeData.points != null && shapeData.points.length > 0) {{
                    g2d.drawPolygon(shapeData.points[0], shapeData.points[1], shapeData.points[0].length);
                }}
        }}
    }}
    
    private void drawCircle(Graphics2D g2d, ShapeData shapeData) {{
        g2d.fillOval(shapeData.x - shapeData.width/2, shapeData.y - shapeData.height/2, 
                    shapeData.width, shapeData.height);
    }}
    
    private void drawRectangle(Graphics2D g2d, ShapeData shapeData) {{
        g2d.fillRect(shapeData.x, shapeData.y, shapeData.width, shapeData.height);
    }}
    
    private void drawTriangle(Graphics2D g2d, ShapeData shapeData) {{
        if (shapeData.points != null && shapeData.points.length == 2) {{
            g2d.fillPolygon(shapeData.points[0], shapeData.points[1], 3);
        }}
    }}
    
    private void drawPolygon(Graphics2D g2d, ShapeData shapeData) {{
        if (shapeData.points != null && shapeData.points.length == 2) {{
            g2d.fillPolygon(shapeData.points[0], shapeData.points[1], shapeData.points[0].length);
        }}
    }}
    
    private void drawEllipse(Graphics2D g2d, ShapeData shapeData) {{
        g2d.fillOval(shapeData.x - shapeData.width/2, shapeData.y - (int)(shapeData.height*0.7)/2, 
                    shapeData.width, (int)(shapeData.height*1.4));
    }}
    
    // Shape data class
    private static class ShapeData {{
        String type;
        int x, y, width, height;
        Color fillColor, borderColor;
        int[][] points; // [x_coords, y_coords]
        double area, perimeter;
        
        public ShapeData(String type, int x, int y, int width, int height, 
                        Color fillColor, Color borderColor, int[][] points, 
                        double area, double perimeter) {{
            this.type = type;
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.fillColor = fillColor;
            this.borderColor = borderColor;
            this.points = points;
            this.area = area;
            this.perimeter = perimeter;
        }}
    }}
    
    public static void main(String[] args) {{
        SwingUtilities.invokeLater(() -> {{
            JFrame frame = new JFrame("Shape Detection Result");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(new ShapeRenderer());
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
            
            // Print shape information
            System.out.println("=== Detected Shapes Information ===");
{self._generate_shape_info(shapes)}
        }});
    }}
}}'''
        
        return java_code
    
    def _generate_shape_data(self, shapes):
        """Generate Java code for shape initialization"""
        shape_init_code = []
        
        for i, shape in enumerate(shapes):
            shape_type = shape['type']
            
            # Get position and size
            if shape_type == 'circle':
                cx, cy = shape['center']
                radius = shape.get('radius', 50)
                x, y, width, height = cx, cy, radius * 2, radius * 2
            else:
                x, y, w, h = shape.get('bounding_rect', (0, 0, 100, 100))
                width, height = w, h
            
            # Get colors
            fill_color = self._get_java_color(shape.get('dominant_color', [128, 128, 128]))
            border_color = self._get_java_color_variant(shape.get('dominant_color', [128, 128, 128]))
            
            # Get points for polygons
            points_code = "null"
            if 'points' in shape and shape['points']:
                points = shape['points']
                x_coords = [str(p[0]) for p in points]
                y_coords = [str(p[1]) for p in points]
                points_code = f"new int[][]{{{{new int[]{{{', '.join(x_coords)}}}, new int[]{{{', '.join(y_coords)}}}}}"
            
            area = shape.get('area', 0)
            perimeter = shape.get('perimeter', 0)
            
            shape_init_code.append(f'''        shapes.add(new ShapeData(
            "{shape_type}", {x}, {y}, {width}, {height},
            {fill_color}, {border_color}, {points_code},
            {area:.2f}, {perimeter:.2f}
        ));''')
        
        return '\n'.join(shape_init_code)
    
    def _generate_shape_info(self, shapes):
        """Generate Java code for printing shape information"""
        info_code = []
        
        for i, shape in enumerate(shapes):
            info_code.append(f'''            System.out.println("Shape {i+1}: {shape['type'].title()}");
            System.out.println("  Area: {shape.get('area', 0):.2f} pixels²");
            System.out.println("  Center: ({shape.get('center', [0, 0])[0]}, {shape.get('center', [0, 0])[1]})");''')
            
            if 'hex_color' in shape:
                info_code.append(f'''            System.out.println("  Color: {shape['hex_color']}");''')
            
            info_code.append(f'''            System.out.println();''')
        
        return '\n'.join(info_code)
    
    def _get_java_color(self, rgb_color):
        """Convert RGB color to Java Color"""
        r, g, b = rgb_color[:3]
        return f"new Color({r}, {g}, {b}, 180)"  # Semi-transparent
    
    def _get_java_color_variant(self, rgb_color):
        """Get darker variant for border"""
        r, g, b = rgb_color[:3]
        # Make darker for border
        r = max(0, r - 40)
        g = max(0, g - 40)
        b = max(0, b - 40)
        return f"new Color({r}, {g}, {b})"