import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import io
import time
import datetime
from shape_detector import ShapeDetector
from advanced_detector import AdvancedShapeDetector
from video_processor import VideoShapeProcessor
from code_generator import CodeGenerator
from database import DatabaseManager
import utils

def main():
    st.set_page_config(
        page_title="Industrial Shape Detection & Code Generator",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Industrial Shape Detection & Code Generator")
    st.markdown("Upload an image with shapes to automatically detect them and generate HTML/CSS/SVG code")
    
    # Initialize database
    try:
        db = DatabaseManager()
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.stop()
    
    # Sidebar for settings and database operations
    with st.sidebar:
        st.header("Settings")
        min_area = st.slider("Minimum Shape Area", 100, 5000, 1000)
        epsilon_factor = st.slider("Contour Approximation Factor", 0.01, 0.1, 0.02, 0.01)
        output_format = st.selectbox("Output Format", ["SVG", "CSS", "HTML", "HTML+CSS+JS", "Java", "Python"])
        show_debug = st.checkbox("Show Debug Information", False)
        use_advanced = st.checkbox("Use Advanced Detection", True)
        detection_mode = st.selectbox("Detection Mode", ["Image", "Video"])
        
        st.divider()
        st.header("Database")
        
        # Save session option
        save_to_db = st.checkbox("Save results to database", False)
        if save_to_db:
            session_name = st.text_input("Session Name", value=f"Detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            session_notes = st.text_area("Notes (optional)", height=80)
        
        # Database operations
        if st.button("View Detection History"):
            st.session_state.show_history = True
        
        if st.button("Show Statistics"):
            st.session_state.show_stats = True
    
    # Initialize components
    detector = ShapeDetector(min_area=min_area, epsilon_factor=epsilon_factor)
    advanced_detector = AdvancedShapeDetector(min_area=min_area, epsilon_factor=epsilon_factor)
    video_processor = VideoShapeProcessor(min_area=min_area, epsilon_factor=epsilon_factor)
    generator = CodeGenerator()
    
    # File upload based on mode
    if detection_mode == "Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Upload an image containing geometric shapes, icons, drawings, or any visual content"
        )
    else:  # Video mode
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to extract shapes from frames"
        )
    
    if uploaded_file is not None:
        try:
            if detection_mode == "Image":
                # Process single image
                image = Image.open(uploaded_file)
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)
                
                # Process image with selected detector
                with st.spinner("Detecting shapes..."):
                    start_time = time.time()
                    if use_advanced:
                        shapes = advanced_detector.detect_shapes_advanced(cv_image)
                        processed_image = advanced_detector.draw_advanced_contours(cv_image.copy(), shapes)
                    else:
                        shapes = detector.detect_shapes(cv_image)
                        processed_image = detector.draw_contours(cv_image.copy(), shapes)
                    processing_time = time.time() - start_time
                
                with col2:
                    st.subheader("Detected Shapes")
                    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    st.image(processed_rgb, use_container_width=True)
                
                # Display results
                if shapes:
                    st.success(f"Found {len(shapes)} shape(s) in {processing_time:.2f} seconds")
                    
                    # Save to database if requested
                    if save_to_db and session_name:
                        try:
                            detection_settings = {
                                'min_area': min_area,
                                'epsilon_factor': epsilon_factor,
                                'output_format': output_format,
                                'use_advanced': use_advanced,
                                'detection_mode': detection_mode
                            }
                            
                            session_id = db.save_detection_session(
                                session_name=session_name,
                                image_filename=uploaded_file.name,
                                image_shape=cv_image.shape,
                                shapes=shapes,
                                detection_settings=detection_settings,
                                processing_time=processing_time,
                                notes=session_notes if 'session_notes' in locals() else None
                            )
                            
                            st.success(f"Results saved to database (Session ID: {session_id})")
                            
                        except Exception as e:
                            st.error(f"Failed to save to database: {str(e)}")
                    
                    # Enhanced shape information
                    if show_debug:
                        st.subheader("Shape Details")
                        for i, shape in enumerate(shapes):
                            with st.expander(f"Shape {i+1} - {shape['type'].title()}"):
                                # Basic info
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.json({k: v for k, v in shape.items() if k not in ['contour', 'approx']})
                                with col2:
                                    if 'hex_color' in shape:
                                        st.color_picker(f"Dominant Color", shape['hex_color'], disabled=True)
                                    if 'brightness' in shape:
                                        st.metric("Brightness", f"{shape['brightness']:.1f}")
                                    if 'contrast' in shape:
                                        st.metric("Contrast", f"{shape['contrast']:.1f}")
                    
                    # Code generation
                    st.subheader(f"Generated {output_format} Code")
                    
                    if output_format == "SVG":
                        code = generator.generate_svg(shapes, cv_image.shape)
                    elif output_format == "CSS":
                        code = generator.generate_css(shapes, cv_image.shape)
                    elif output_format == "HTML":
                        code = generator.generate_html(shapes, cv_image.shape)
                    elif output_format == "HTML+CSS+JS":
                        code = generator.generate_interactive_html(shapes, cv_image.shape)
                    elif output_format == "Java":
                        code = generator.generate_java(shapes, cv_image.shape)
                    else:  # Python
                        code = generator.generate_python(shapes, cv_image.shape)
                    
                    # Code display with syntax highlighting
                    if output_format == "SVG":
                        st.code(code, language="xml")
                    elif output_format == "CSS":
                        st.code(code, language="css")
                    elif output_format == "HTML":
                        st.code(code, language="html")
                    elif output_format == "HTML+CSS+JS":
                        st.code(code, language="html")
                    elif output_format == "Java":
                        st.code(code, language="java")
                    else:  # Python
                        st.code(code, language="python")
                    
                    # Download button
                    file_extension = self._get_file_extension(output_format)
                    st.download_button(
                        label=f"Download {output_format} Code",
                        data=code,
                        file_name=f"shapes.{file_extension}",
                        mime=utils.get_mime_type(output_format)
                    )
                    
                    # Preview (for HTML and SVG)
                    if output_format in ["HTML", "SVG", "HTML+CSS+JS"]:
                        st.subheader("Code Preview")
                        st.components.v1.html(code, height=400, scrolling=True)
                    
                else:
                    st.warning("No shapes detected. Try adjusting the settings or using advanced detection.")
                    st.info("Tips: Advanced detection works better with complex images, icons, and varied content.")
            
            else:  # Video mode
                st.subheader("Video Processing")
                
                with st.spinner("Processing video frames..."):
                    start_time = time.time()
                    
                    # Process video
                    sample_rate = st.sidebar.slider("Frame Sample Rate", 1, 60, 30, 
                                                   help="Process every Nth frame")
                    
                    video_result = video_processor.process_video_file(uploaded_file, sample_rate)
                    processing_time = time.time() - start_time
                
                if video_result['processed_frames']:
                    st.success(f"Processed {video_result['total_processed']} frames with shapes in {processing_time:.2f} seconds")
                    
                    # Video summary
                    summary = video_processor.generate_video_summary_report(video_result)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Shapes", summary['video_stats']['total_shapes_detected'])
                    with col2:
                        st.metric("Shape Types", summary['video_stats']['unique_shape_types'])
                    with col3:
                        st.metric("Frames with Shapes", summary['video_stats']['frames_with_shapes'])
                    
                    # Display key frames
                    st.subheader("Key Frames with Detected Shapes")
                    
                    key_frames = summary['key_frames'][:6]  # Show top 6 frames
                    
                    cols = st.columns(3)
                    for i, frame_data in enumerate(key_frames):
                        with cols[i % 3]:
                            # Convert BGR to RGB for display
                            display_frame = cv2.cvtColor(frame_data['processed_frame'], cv2.COLOR_BGR2RGB)
                            st.image(display_frame, caption=f"Frame {frame_data['frame_number']} - {frame_data['shape_count']} shapes")
                            
                            if st.button(f"Generate Code for Frame {frame_data['frame_number']}", key=f"frame_{i}"):
                                # Generate code for this frame
                                frame_shapes = frame_data['shapes']
                                frame_image_shape = frame_data['original_frame'].shape
                                
                                if output_format == "SVG":
                                    frame_code = generator.generate_svg(frame_shapes, frame_image_shape)
                                elif output_format == "CSS":
                                    frame_code = generator.generate_css(frame_shapes, frame_image_shape)
                                elif output_format == "HTML":
                                    frame_code = generator.generate_html(frame_shapes, frame_image_shape)
                                elif output_format == "HTML+CSS+JS":
                                    frame_code = generator.generate_interactive_html(frame_shapes, frame_image_shape)
                                elif output_format == "Java":
                                    frame_code = generator.generate_java(frame_shapes, frame_image_shape)
                                else:  # Python
                                    frame_code = generator.generate_python(frame_shapes, frame_image_shape)
                                
                                # Display code with appropriate syntax highlighting
                                if output_format == "CSS":
                                    st.code(frame_code, language="css")
                                elif output_format == "Java":
                                    st.code(frame_code, language="java")
                                elif output_format == "Python":
                                    st.code(frame_code, language="python")
                                elif output_format == "SVG":
                                    st.code(frame_code, language="xml")
                                else:
                                    st.code(frame_code, language="html")
                                
                                # Download button for frame
                                file_extension = self._get_file_extension(output_format)
                                st.download_button(
                                    label=f"Download Frame {frame_data['frame_number']} Code",
                                    data=frame_code,
                                    file_name=f"frame_{frame_data['frame_number']}_shapes.{file_extension}",
                                    mime=utils.get_mime_type(output_format),
                                    key=f"download_frame_{i}"
                                )
                    
                    # Shape distribution chart
                    if summary['shape_distribution']:
                        st.subheader("Shape Distribution Across Video")
                        
                        shape_data = []
                        for shape_type, data in summary['shape_distribution'].items():
                            shape_data.append({
                                'Shape Type': shape_type.title(),
                                'Count': data['count'],
                                'Avg Area': data['avg_area']
                            })
                        
                        import pandas as pd
                        df = pd.DataFrame(shape_data)
                        st.dataframe(df, use_container_width=True)
                    
                    # Save video results to database
                    if save_to_db and session_name:
                        try:
                            # Save key frame data
                            for frame_data in key_frames[:3]:  # Save top 3 frames
                                frame_session_name = f"{session_name}_frame_{frame_data['frame_number']}"
                                
                                detection_settings = {
                                    'min_area': min_area,
                                    'epsilon_factor': epsilon_factor,
                                    'output_format': output_format,
                                    'use_advanced': use_advanced,
                                    'detection_mode': detection_mode,
                                    'video_frame': frame_data['frame_number'],
                                    'sample_rate': sample_rate
                                }
                                
                                session_id = db.save_detection_session(
                                    session_name=frame_session_name,
                                    image_filename=f"{uploaded_file.name}_frame_{frame_data['frame_number']}",
                                    image_shape=frame_data['original_frame'].shape,
                                    shapes=frame_data['shapes'],
                                    detection_settings=detection_settings,
                                    processing_time=processing_time / len(key_frames),
                                    notes=f"Video frame analysis. {session_notes if 'session_notes' in locals() else ''}"
                                )
                            
                            st.success(f"Video results saved to database ({len(key_frames[:3])} key frames)")
                            
                        except Exception as e:
                            st.error(f"Failed to save video results: {str(e)}")
                
                else:
                    st.warning("No shapes detected in video frames. Try adjusting sample rate or detection settings.")
                    st.info("Tips: Videos with clear geometric shapes, animations, or graphics work best.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please try uploading a different file or check the file format.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Upload an image to get started")
        
        with st.expander("How to use this tool"):
            st.markdown("""
            **Steps:**
            1. Upload an image containing geometric shapes (PNG, JPG, etc.)
            2. Adjust detection settings in the sidebar if needed
            3. View detected shapes and generated code
            4. Download the generated code
            
            **Supported Shapes:**
            - Circles
            - Rectangles/Squares
            - Triangles
            - Polygons
            
            **Detection Modes:**
            - **Image**: Process single images, photos, icons, drawings
            - **Video**: Extract shapes from video frames automatically
            
            **Detection Methods:**
            - **Standard**: Fast detection for simple geometric shapes
            - **Advanced**: Precise detection for complex images, icons, and varied content
            
            **Output Formats:**
            - **SVG**: Scalable vector graphics with exact shape reproduction
            - **CSS**: CSS-based shape creation with precise positioning  
            - **HTML**: Complete HTML with inline styles and shape information
            - **HTML+CSS+JS**: Interactive HTML with animations and controls
            - **Java**: Complete Java Swing application with exact visual reproduction
            - **Python**: Python code using matplotlib and PIL for precise rendering
            
            **Database Features:**
            - Save detection results for later analysis
            - View detection history and statistics
            - Search and filter previous sessions
            - Export data to CSV format
            """)

    # Database views
    if 'show_history' in st.session_state and st.session_state.show_history:
        st.subheader("Detection History")
        
        try:
            sessions = db.get_detection_sessions()
            
            if sessions:
                # Search and filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    search_query = st.text_input("Search sessions")
                with col2:
                    shape_filter = st.selectbox("Filter by shape", ["All", "circle", "rectangle", "square", "triangle", "polygon"])
                with col3:
                    if st.button("Clear Filters"):
                        st.session_state.show_history = False
                        st.rerun()
                
                # Display sessions table
                for session in sessions[:20]:  # Limit to 20 recent sessions
                    with st.expander(f"{session['session_name']} - {session['shapes_detected']} shapes"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Created:** {session['created_at']}")
                            st.write(f"**Image:** {session['image_filename']}")
                        with col2:
                            st.write(f"**Shapes:** {session['shapes_detected']}")
                            st.write(f"**Size:** {session['image_size']}")
                        with col3:
                            if session['processing_time']:
                                st.write(f"**Time:** {session['processing_time']:.2f}s")
                            
                            if st.button(f"Load Session {session['id']}", key=f"load_{session['id']}"):
                                # Load session details
                                details = db.get_session_details(session['id'])
                                if details:
                                    st.session_state.loaded_session = details
                                    st.success("Session loaded!")
                            
                            if st.button(f"Delete", key=f"delete_{session['id']}", type="secondary"):
                                if st.button(f"Confirm Delete {session['id']}", key=f"confirm_delete_{session['id']}"):
                                    db.delete_session(session['id'])
                                    st.success("Session deleted!")
                                    st.rerun()
            else:
                st.info("No detection sessions found.")
                
        except Exception as e:
            st.error(f"Error loading history: {str(e)}")
        
        if st.button("Close History"):
            st.session_state.show_history = False
            st.rerun()
    
    # Statistics view
    if 'show_stats' in st.session_state and st.session_state.show_stats:
        st.subheader("Detection Statistics")
        
        try:
            stats = db.get_shape_statistics()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Sessions", stats['session_summary']['total_sessions'])
                st.metric("Total Shapes Detected", stats['session_summary']['total_shapes'])
                if stats['session_summary']['avg_processing_time'] > 0:
                    st.metric("Average Processing Time", f"{stats['session_summary']['avg_processing_time']:.2f}s")
            
            with col2:
                if stats['shape_distribution']:
                    st.subheader("Shape Distribution")
                    for shape_stat in stats['shape_distribution']:
                        st.write(f"**{shape_stat['shape_type'].title()}:** {shape_stat['count']} detected")
                        st.write(f"  - Average area: {shape_stat['avg_area']:.0f} pixels²")
                        
                    # Export option
                    if st.button("Export All Data to CSV"):
                        try:
                            df = db.export_data_to_csv()
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"shape_detection_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
                            
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")
        
        if st.button("Close Statistics"):
            st.session_state.show_stats = False
            st.rerun()
    
    # Display loaded session details
    if 'loaded_session' in st.session_state:
        st.subheader("Loaded Session Details")
        session_data = st.session_state.loaded_session
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Session:** {session_data['session']['session_name']}")
            st.write(f"**Date:** {session_data['session']['created_at']}")
            st.write(f"**Image:** {session_data['session']['image_filename']}")
            st.write(f"**Shapes:** {session_data['session']['shapes_detected']}")
        
        with col2:
            if session_data['session']['detection_settings']:
                st.write("**Settings:**")
                st.json(session_data['session']['detection_settings'])
            
            if session_data['session']['notes']:
                st.write(f"**Notes:** {session_data['session']['notes']}")
        
        # Regenerate code from saved session
        if session_data['session']['shapes_data']:
            st.subheader("Regenerate Code")
            regenerate_format = st.selectbox("Format", ["SVG", "CSS", "HTML", "HTML+CSS+JS", "Java", "Python"], key="regenerate_format")
            
            if st.button("Generate Code from Session"):
                try:
                    generator = CodeGenerator()
                    shapes = session_data['session']['shapes_data']
                    image_shape = (session_data['session']['image_size_height'], 
                                 session_data['session']['image_size_width'], 3)
                    
                    if regenerate_format == "SVG":
                        code = generator.generate_svg(shapes, image_shape)
                        st.code(code, language="xml")
                    elif regenerate_format == "CSS":
                        code = generator.generate_css(shapes, image_shape)
                        st.code(code, language="css")
                    elif regenerate_format == "HTML":
                        code = generator.generate_html(shapes, image_shape)
                        st.code(code, language="html")
                    elif regenerate_format == "HTML+CSS+JS":
                        code = generator.generate_interactive_html(shapes, image_shape)
                        st.code(code, language="html")
                    elif regenerate_format == "Java":
                        code = generator.generate_java(shapes, image_shape)
                        st.code(code, language="java")
                    else:  # Python
                        code = generator.generate_python(shapes, image_shape)
                        st.code(code, language="python")
                    
                    # Download button
                    file_extension = self._get_file_extension(regenerate_format)
                    st.download_button(
                        label=f"Download {regenerate_format} Code",
                        data=code,
                        file_name=f"session_{session_data['session']['id']}_shapes.{file_extension}",
                        mime=utils.get_mime_type(regenerate_format)
                    )
                    
                except Exception as e:
                    st.error(f"Code generation failed: {str(e)}")
        
        if st.button("Clear Loaded Session"):
            del st.session_state.loaded_session
            st.rerun()

def _get_file_extension(output_format):
    """Get appropriate file extension for output format"""
    extensions = {
        "SVG": "svg",
        "CSS": "css", 
        "HTML": "html",
        "HTML+CSS+JS": "html",
        "Java": "java",
        "Python": "py"
    }
    return extensions.get(output_format, "txt")

if __name__ == "__main__":
    main()
