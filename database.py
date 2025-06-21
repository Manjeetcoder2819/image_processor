import os
import json
import datetime
from typing import List, Dict, Optional, Any
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class DetectionSession(Base):
    """Database model for detection sessions"""
    __tablename__ = 'detection_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_name = Column(String(200), nullable=False, index=True)
    image_filename = Column(String(500))
    image_size_width = Column(Integer)
    image_size_height = Column(Integer)
    shapes_detected = Column(Integer, default=0)
    detection_settings = Column(JSONB)
    shapes_data = Column(JSONB)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    processing_time = Column(Float)
    notes = Column(Text)
    is_active = Column(Boolean, default=True)
    
    # Relationship to shape records
    shapes = relationship("ShapeRecord", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DetectionSession(id={self.id}, name='{self.session_name}', shapes={self.shapes_detected})>"

class ShapeRecord(Base):
    """Database model for individual shape records"""
    __tablename__ = 'shape_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('detection_sessions.id'), nullable=False, index=True)
    shape_type = Column(String(50), nullable=False, index=True)
    shape_index = Column(Integer, nullable=False)
    center_x = Column(Float)
    center_y = Column(Float)
    area = Column(Float, index=True)
    perimeter = Column(Float)
    bounding_rect_x = Column(Float)
    bounding_rect_y = Column(Float)
    bounding_rect_width = Column(Float)
    bounding_rect_height = Column(Float)
    confidence_score = Column(Float)
    properties = Column(JSONB)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship back to session
    session = relationship("DetectionSession", back_populates="shapes")
    
    def __repr__(self):
        return f"<ShapeRecord(id={self.id}, type='{self.shape_type}', area={self.area})>"

class DatabaseManager:
    """Enhanced database manager for shape detection data"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database connection
        
        Args:
            database_url: Database connection string. If None, uses DATABASE_URL environment variable
        """
        self.database_url = database_url or os.environ.get('DATABASE_URL')
        if not self.database_url:
            # Fallback to SQLite for development
            self.database_url = 'sqlite:///shape_detection.db'
            logger.warning("No DATABASE_URL found, using SQLite database: shape_detection.db")
        
        try:
            self.engine = create_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,
                pool_recycle=300
            )
            self.Session = sessionmaker(bind=self.engine)
            self.create_tables()
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    def save_detection_session(self, 
                             session_name: str, 
                             image_filename: str, 
                             image_shape: tuple, 
                             shapes: List[Dict[str, Any]], 
                             detection_settings: Dict[str, Any],
                             processing_time: Optional[float] = None, 
                             notes: Optional[str] = None) -> int:
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
                    center_x=shape.get('center', [0, 0])[0] if shape.get('center') else None,
                    center_y=shape.get('center', [0, 0])[1] if shape.get('center') else None,
                    area=shape.get('area', 0),
                    perimeter=shape.get('perimeter', 0),
                    bounding_rect_x=shape.get('bounding_rect', [0, 0, 0, 0])[0] if shape.get('bounding_rect') else None,
                    bounding_rect_y=shape.get('bounding_rect', [0, 0, 0, 0])[1] if shape.get('bounding_rect') else None,
                    bounding_rect_width=shape.get('bounding_rect', [0, 0, 0, 0])[2] if shape.get('bounding_rect') else None,
                    bounding_rect_height=shape.get('bounding_rect', [0, 0, 0, 0])[3] if shape.get('bounding_rect') else None,
                    confidence_score=shape.get('confidence', 0.0),
                    properties=shape
                )
                session.add(shape_record)
            
            session.commit()
            logger.info(f"Successfully saved session '{session_name}' with {len(shapes)} shapes")
            return session_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save detection session: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_detection_sessions(self, limit: int = 50, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """Get list of detection sessions"""
        session = self.Session()
        try:
            query = session.query(DetectionSession)
            
            if not include_inactive:
                query = query.filter(DetectionSession.is_active == True)
            
            sessions = query.order_by(
                DetectionSession.created_at.desc()
            ).limit(limit).all()
            
            return [{
                'id': s.id,
                'session_name': s.session_name,
                'image_filename': s.image_filename,
                'shapes_detected': s.shapes_detected,
                'created_at': s.created_at,
                'updated_at': s.updated_at,
                'processing_time': s.processing_time,
                'image_size': f"{s.image_size_width}x{s.image_size_height}" if s.image_size_width else "Unknown",
                'is_active': s.is_active
            } for s in sessions]
            
        except Exception as e:
            logger.error(f"Failed to get detection sessions: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_session_details(self, session_id: int) -> Optional[Dict[str, Any]]:
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
                    'updated_at': detection_session.updated_at,
                    'processing_time': detection_session.processing_time,
                    'notes': detection_session.notes,
                    'is_active': detection_session.is_active
                },
                'shapes': [{
                    'id': s.id,
                    'shape_type': s.shape_type,
                    'shape_index': s.shape_index,
                    'center': [s.center_x, s.center_y] if s.center_x is not None else None,
                    'area': s.area,
                    'perimeter': s.perimeter,
                    'bounding_rect': [s.bounding_rect_x, s.bounding_rect_y, 
                                    s.bounding_rect_width, s.bounding_rect_height] if s.bounding_rect_x is not None else None,
                    'confidence_score': s.confidence_score,
                    'properties': s.properties,
                    'created_at': s.created_at
                } for s in shapes]
            }
            
        except Exception as e:
            logger.error(f"Failed to get session details for ID {session_id}: {str(e)}")
            raise
        finally:
            session.close()
    
    def update_session(self, session_id: int, **kwargs) -> bool:
        """Update a detection session"""
        session = self.Session()
        try:
            detection_session = session.query(DetectionSession).filter(
                DetectionSession.id == session_id
            ).first()
            
            if not detection_session:
                return False
            
            # Update allowed fields
            allowed_fields = ['session_name', 'notes', 'is_active']
            for field, value in kwargs.items():
                if field in allowed_fields:
                    setattr(detection_session, field, value)
            
            detection_session.updated_at = datetime.datetime.utcnow()
            session.commit()
            logger.info(f"Successfully updated session {session_id}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update session {session_id}: {str(e)}")
            raise
        finally:
            session.close()
    
    def delete_session(self, session_id: int, soft_delete: bool = True) -> bool:
        """Delete a detection session and all related shape records"""
        session = self.Session()
        try:
            if soft_delete:
                # Soft delete - mark as inactive
                detection_session = session.query(DetectionSession).filter(
                    DetectionSession.id == session_id
                ).first()
                
                if detection_session:
                    detection_session.is_active = False
                    detection_session.updated_at = datetime.datetime.utcnow()
                    session.commit()
                    logger.info(f"Successfully soft-deleted session {session_id}")
                    return True
                return False
            else:
                # Hard delete - remove from database
                # Delete shape records first (cascade should handle this, but being explicit)
                shape_count = session.query(ShapeRecord).filter(
                    ShapeRecord.session_id == session_id
                ).delete()
                
                # Delete detection session
                session_count = session.query(DetectionSession).filter(
                    DetectionSession.id == session_id
                ).delete()
                
                session.commit()
                logger.info(f"Successfully hard-deleted session {session_id} with {shape_count} shapes")
                return session_count > 0
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete session {session_id}: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_shape_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about detected shapes"""
        session = self.Session()
        try:
            # Shape type distribution
            shape_stats = session.query(
                ShapeRecord.shape_type,
                session.func.count(ShapeRecord.id).label('count'),
                session.func.avg(ShapeRecord.area).label('avg_area'),
                session.func.max(ShapeRecord.area).label('max_area'),
                session.func.min(ShapeRecord.area).label('min_area'),
                session.func.avg(ShapeRecord.perimeter).label('avg_perimeter'),
                session.func.avg(ShapeRecord.confidence_score).label('avg_confidence')
            ).group_by(ShapeRecord.shape_type).all()
            
            # Session stats
            session_stats = session.query(
                session.func.count(DetectionSession.id).label('total_sessions'),
                session.func.sum(DetectionSession.shapes_detected).label('total_shapes'),
                session.func.avg(DetectionSession.processing_time).label('avg_processing_time'),
                session.func.count(session.case(
                    [(DetectionSession.is_active == True, 1)]
                )).label('active_sessions')
            ).first()
            
            # Recent activity (last 30 days)
            thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
            recent_stats = session.query(
                session.func.count(DetectionSession.id).label('recent_sessions')
            ).filter(DetectionSession.created_at >= thirty_days_ago).first()
            
            return {
                'shape_distribution': [{
                    'shape_type': stat.shape_type,
                    'count': stat.count,
                    'avg_area': float(stat.avg_area) if stat.avg_area else 0,
                    'max_area': float(stat.max_area) if stat.max_area else 0,
                    'min_area': float(stat.min_area) if stat.min_area else 0,
                    'avg_perimeter': float(stat.avg_perimeter) if stat.avg_perimeter else 0,
                    'avg_confidence': float(stat.avg_confidence) if stat.avg_confidence else 0
                } for stat in shape_stats],
                'session_summary': {
                    'total_sessions': session_stats.total_sessions or 0,
                    'active_sessions': session_stats.active_sessions or 0,
                    'total_shapes': session_stats.total_shapes or 0,
                    'avg_processing_time': float(session_stats.avg_processing_time) if session_stats.avg_processing_time else 0,
                    'recent_sessions_30d': recent_stats.recent_sessions or 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get shape statistics: {str(e)}")
            raise
        finally:
            session.close()
    
    def search_sessions(self, 
                       query: Optional[str] = None, 
                       shape_type: Optional[str] = None, 
                       date_from: Optional[datetime.datetime] = None, 
                       date_to: Optional[datetime.datetime] = None,
                       min_shapes: Optional[int] = None,
                       max_shapes: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search detection sessions with comprehensive filters"""
        session = self.Session()
        try:
            query_obj = session.query(DetectionSession).filter(
                DetectionSession.is_active == True
            )
            
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
            
            if min_shapes is not None:
                query_obj = query_obj.filter(
                    DetectionSession.shapes_detected >= min_shapes
                )
            
            if max_shapes is not None:
                query_obj = query_obj.filter(
                    DetectionSession.shapes_detected <= max_shapes
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
                'updated_at': s.updated_at,
                'processing_time': s.processing_time,
                'image_size': f"{s.image_size_width}x{s.image_size_height}" if s.image_size_width else "Unknown"
            } for s in sessions]
            
        except Exception as e:
            logger.error(f"Failed to search sessions: {str(e)}")
            raise
        finally:
            session.close()
    
    def export_data_to_csv(self, session_ids: Optional[List[int]] = None, output_file: Optional[str] = None) -> pd.DataFrame:
        """Export detection data to CSV format"""
        session = self.Session()
        try:
            query = session.query(
                DetectionSession.id.label('session_id'),
                DetectionSession.session_name,
                DetectionSession.image_filename,
                DetectionSession.created_at.label('session_date'),
                DetectionSession.processing_time,
                ShapeRecord.shape_type,
                ShapeRecord.shape_index,
                ShapeRecord.center_x,
                ShapeRecord.center_y,
                ShapeRecord.area,
                ShapeRecord.perimeter,
                ShapeRecord.bounding_rect_width,
                ShapeRecord.bounding_rect_height,
                ShapeRecord.confidence_score
            ).join(ShapeRecord).filter(DetectionSession.is_active == True)
            
            if session_ids:
                query = query.filter(DetectionSession.id.in_(session_ids))
            
            # Convert to pandas DataFrame
            df = pd.read_sql(query.statement, self.engine)
            
            if output_file:
                df.to_csv(output_file, index=False)
                logger.info(f"Data exported to {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to export data: {str(e)}")
            raise
        finally:
            session.close()
    
    def backup_database(self, backup_file: str):
        """Create a backup of all detection data"""
        try:
            sessions_data = self.get_detection_sessions(limit=None, include_inactive=True)
            detailed_data = []
            
            for session_info in sessions_data:
                session_details = self.get_session_details(session_info['id'])
                if session_details:
                    detailed_data.append(session_details)
            
            backup_data = {
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'sessions_count': len(detailed_data),
                'data': detailed_data
            }
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Database backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database"""
        session = self.Session()
        try:
            # Get table sizes
            session_count = session.query(DetectionSession).count()
            shape_count = session.query(ShapeRecord).count()
            active_sessions = session.query(DetectionSession).filter(
                DetectionSession.is_active == True
            ).count()
            
            # Get date range
            oldest_session = session.query(DetectionSession.created_at).order_by(
                DetectionSession.created_at.asc()
            ).first()
            
            newest_session = session.query(DetectionSession.created_at).order_by(
                DetectionSession.created_at.desc()
            ).first()
            
            return {
                'database_url': self.database_url.split('@')[-1] if '@' in self.database_url else 'Local SQLite',
                'total_sessions': session_count,
                'active_sessions': active_sessions,
                'total_shapes': shape_count,
                'oldest_session': oldest_session[0] if oldest_session else None,
                'newest_session': newest_session[0] if newest_session else None,
                'avg_shapes_per_session': round(shape_count / session_count, 2) if session_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get database info: {str(e)}")
            raise
        finally:
            session.close()
    
    def close(self):
        """Close database connections"""
        try:
            self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        # Initialize database manager
        db = DatabaseManager()
        
        # Example shape data
        sample_shapes = [
            {
                'type': 'circle',
                'center': [100, 150],
                'area': 314.159,
                'perimeter': 62.83,
                'bounding_rect': [75, 125, 50, 50],
                'confidence': 0.95
            },
            {
                'type': 'rectangle',
                'center': [200, 100],
                'area': 2400,
                'perimeter': 200,
                'bounding_rect': [150, 50, 100, 100],
                'confidence': 0.88
            }
        ]
        
        sample_settings = {
            'threshold': 0.5,
            'algorithm': 'contour_detection',
            'blur_kernel': 5
        }
        
        # Save a sample session
        session_id = db.save_detection_session(
            session_name="Test Session",
            image_filename="test_image.jpg",
            image_shape=(480, 640, 3),
            shapes=sample_shapes,
            detection_settings=sample_settings,
            processing_time=2.5,
            notes="This is a test session"
        )
        
        print(f"Created session with ID: {session_id}")
        
        # Get database info
        info = db.get_database_info()
        print(f"Database info: {info}")
        
        # Get statistics
        stats = db.get_shape_statistics()
        print(f"Shape statistics: {stats}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'db' in locals():
            db.close()