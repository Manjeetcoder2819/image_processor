import os
import json
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
import pandas as pd

Base = declarative_base()

class DetectionSession(Base):
    __tablename__ = 'detection_sessions'
    
    id = Column(Integer, primary_key=True)
    session_name = Column(String(200), nullable=False)
    image_filename = Column(String(500))
    image_size_width = Column(Integer)
    image_size_height = Column(Integer)
    shapes_detected = Column(Integer, default=0)
    detection_settings = Column(JSONB)
    shapes_data = Column(JSONB)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    processing_time = Column(Float)
    notes = Column(Text)

class ShapeRecord(Base):
    __tablename__ = 'shape_records'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, nullable=False)
    shape_type = Column(String(50), nullable=False)
    shape_index = Column(Integer, nullable=False)
    center_x = Column(Float)
    center_y = Column(Float)
    area = Column(Float)
    perimeter = Column(Float)
    bounding_rect_x = Column(Float)
    bounding_rect_y = Column(Float)
    bounding_rect_width = Column(Float)
    bounding_rect_height = Column(Float)
    properties = Column(JSONB)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        """Initialize database connection"""
        self.database_url = os.environ.get('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
        
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        Base.metadata.create_all(self.engine)
    
    def save_detection_session(self, session_name, image_filename, image_shape, shapes, 
                             detection_settings, processing_time=None, notes=None):
        """
        Save a detection session to the database
        
        Args:
            session_name: Name for the session
            image_filename: Original image filename
            image_shape: Image dimensions (height, width, channels)
            shapes: List of detected shapes
            detection_settings: Detection parameters used
            processing_time: Time taken for processing
            notes: Optional notes
            
        Returns:
            Session ID
        """
        session = self.Session()
        try:
            # Create detection session record
            detection_session = DetectionSession(
                session_name=session_name,
                image_filename=image_filename,
                image_size_width=image_shape[1] if len(image_shape) > 1 else None,
                image_size_height=image_shape[0] if len(image_shape) > 0 else None,
                shapes_detected=len(shapes),
                detection_settings=detection_settings,
                shapes_data=shapes,
                processing_time=processing_time,
                notes=notes
            )
            
            session.add(detection_session)
            session.flush()  # Get the ID
            session_id = detection_session.id
            
            # Create individual shape records
            for i, shape in enumerate(shapes):
                shape_record = ShapeRecord(
                    session_id=session_id,
                    shape_type=shape.get('type', 'unknown'),
                    shape_index=i,
                    center_x=shape.get('center', [0, 0])[0],
                    center_y=shape.get('center', [0, 0])[1],
                    area=shape.get('area', 0),
                    perimeter=shape.get('perimeter', 0),
                    bounding_rect_x=shape.get('bounding_rect', [0, 0, 0, 0])[0],
                    bounding_rect_y=shape.get('bounding_rect', [0, 0, 0, 0])[1],
                    bounding_rect_width=shape.get('bounding_rect', [0, 0, 0, 0])[2],
                    bounding_rect_height=shape.get('bounding_rect', [0, 0, 0, 0])[3],
                    properties=shape
                )
                session.add(shape_record)
            
            session.commit()
            return session_id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_detection_sessions(self, limit=50):
        """Get list of detection sessions"""
        session = self.Session()
        try:
            sessions = session.query(DetectionSession).order_by(
                DetectionSession.created_at.desc()
            ).limit(limit).all()
            
            return [{
                'id': s.id,
                'session_name': s.session_name,
                'image_filename': s.image_filename,
                'shapes_detected': s.shapes_detected,
                'created_at': s.created_at,
                'processing_time': s.processing_time,
                'image_size': f"{s.image_size_width}x{s.image_size_height}" if s.image_size_width else "Unknown"
            } for s in sessions]
            
        finally:
            session.close()
    
    def get_session_details(self, session_id):
        """Get detailed information about a specific session"""
        session = self.Session()
        try:
            detection_session = session.query(DetectionSession).filter(
                DetectionSession.id == session_id
            ).first()
            
            if not detection_session:
                return None
            
            shapes = session.query(ShapeRecord).filter(
                ShapeRecord.session_id == session_id
            ).order_by(ShapeRecord.shape_index).all()
            
            return {
                'session': {
                    'id': detection_session.id,
                    'session_name': detection_session.session_name,
                    'image_filename': detection_session.image_filename,
                    'image_size_width': detection_session.image_size_width,
                    'image_size_height': detection_session.image_size_height,
                    'shapes_detected': detection_session.shapes_detected,
                    'detection_settings': detection_session.detection_settings,
                    'shapes_data': detection_session.shapes_data,
                    'created_at': detection_session.created_at,
                    'processing_time': detection_session.processing_time,
                    'notes': detection_session.notes
                },
                'shapes': [{
                    'id': s.id,
                    'shape_type': s.shape_type,
                    'shape_index': s.shape_index,
                    'center': [s.center_x, s.center_y],
                    'area': s.area,
                    'perimeter': s.perimeter,
                    'bounding_rect': [s.bounding_rect_x, s.bounding_rect_y, 
                                    s.bounding_rect_width, s.bounding_rect_height],
                    'properties': s.properties,
                    'created_at': s.created_at
                } for s in shapes]
            }
            
        finally:
            session.close()
    
    def delete_session(self, session_id):
        """Delete a detection session and all related shape records"""
        session = self.Session()
        try:
            # Delete shape records first
            session.query(ShapeRecord).filter(
                ShapeRecord.session_id == session_id
            ).delete()
            
            # Delete detection session
            session.query(DetectionSession).filter(
                DetectionSession.id == session_id
            ).delete()
            
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_shape_statistics(self):
        """Get statistics about detected shapes"""
        session = self.Session()
        try:
            # Shape type distribution
            shape_stats = session.query(
                ShapeRecord.shape_type,
                session.func.count(ShapeRecord.id).label('count'),
                session.func.avg(ShapeRecord.area).label('avg_area'),
                session.func.max(ShapeRecord.area).label('max_area'),
                session.func.min(ShapeRecord.area).label('min_area')
            ).group_by(ShapeRecord.shape_type).all()
            
            # Session stats
            session_stats = session.query(
                session.func.count(DetectionSession.id).label('total_sessions'),
                session.func.sum(DetectionSession.shapes_detected).label('total_shapes'),
                session.func.avg(DetectionSession.processing_time).label('avg_processing_time')
            ).first()
            
            return {
                'shape_distribution': [{
                    'shape_type': stat.shape_type,
                    'count': stat.count,
                    'avg_area': float(stat.avg_area) if stat.avg_area else 0,
                    'max_area': float(stat.max_area) if stat.max_area else 0,
                    'min_area': float(stat.min_area) if stat.min_area else 0
                } for stat in shape_stats],
                'session_summary': {
                    'total_sessions': session_stats.total_sessions or 0,
                    'total_shapes': session_stats.total_shapes or 0,
                    'avg_processing_time': float(session_stats.avg_processing_time) if session_stats.avg_processing_time else 0
                }
            }
            
        finally:
            session.close()
    
    def search_sessions(self, query=None, shape_type=None, date_from=None, date_to=None):
        """Search detection sessions with filters"""
        session = self.Session()
        try:
            query_obj = session.query(DetectionSession)
            
            if query:
                query_obj = query_obj.filter(
                    DetectionSession.session_name.ilike(f'%{query}%')
                )
            
            if shape_type:
                # Join with shape records to filter by shape type
                query_obj = query_obj.join(ShapeRecord).filter(
                    ShapeRecord.shape_type == shape_type
                ).distinct()
            
            if date_from:
                query_obj = query_obj.filter(
                    DetectionSession.created_at >= date_from
                )
            
            if date_to:
                query_obj = query_obj.filter(
                    DetectionSession.created_at <= date_to
                )
            
            sessions = query_obj.order_by(
                DetectionSession.created_at.desc()
            ).all()
            
            return [{
                'id': s.id,
                'session_name': s.session_name,
                'image_filename': s.image_filename,
                'shapes_detected': s.shapes_detected,
                'created_at': s.created_at,
                'processing_time': s.processing_time
            } for s in sessions]
            
        finally:
            session.close()
    
    def export_data_to_csv(self, session_ids=None):
        """Export detection data to CSV format"""
        session = self.Session()
        try:
            query = session.query(
                DetectionSession.id.label('session_id'),
                DetectionSession.session_name,
                DetectionSession.image_filename,
                DetectionSession.created_at.label('session_date'),
                ShapeRecord.shape_type,
                ShapeRecord.shape_index,
                ShapeRecord.center_x,
                ShapeRecord.center_y,
                ShapeRecord.area,
                ShapeRecord.perimeter,
                ShapeRecord.bounding_rect_width,
                ShapeRecord.bounding_rect_height
            ).join(ShapeRecord)
            
            if session_ids:
                query = query.filter(DetectionSession.id.in_(session_ids))
            
            # Convert to pandas DataFrame
            df = pd.read_sql(query.statement, self.engine)
            return df
            
        finally:
            session.close()