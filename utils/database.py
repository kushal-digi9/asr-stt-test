import os
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Float, Time, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, time as dt_time

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://voiceai:voiceai_password@localhost:5432/voiceai_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Medical Receptionist Tables
class Department(Base):
    __tablename__ = "departments"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    name_hindi = Column(String)  # Hindi name
    description = Column(Text)
    floor = Column(String)
    contact_number = Column(String)
    is_active = Column(Boolean, default=True)


class Doctor(Base):
    __tablename__ = "doctors"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    name_hindi = Column(String)
    specialization = Column(String)
    department_id = Column(Integer, ForeignKey("departments.id"))
    qualification = Column(String)
    consultation_fee = Column(Float)
    availability_days = Column(String)  # e.g., "Monday,Wednesday,Friday"
    available_from = Column(Time)
    available_to = Column(Time)
    is_active = Column(Boolean, default=True)
    
    department = relationship("Department")


class Service(Base):
    __tablename__ = "services"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    name_hindi = Column(String)
    description = Column(Text)
    price = Column(Float)
    department_id = Column(Integer, ForeignKey("departments.id"))
    is_active = Column(Boolean, default=True)
    
    department = relationship("Department")


class Appointment(Base):
    __tablename__ = "appointments"
    
    id = Column(Integer, primary_key=True)
    patient_name = Column(String, nullable=False)
    patient_phone = Column(String)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    appointment_date = Column(DateTime)
    appointment_time = Column(Time)
    status = Column(String)  # scheduled, completed, cancelled
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    doctor = relationship("Doctor")


class HospitalInfo(Base):
    __tablename__ = "hospital_info"
    
    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, nullable=False)  # e.g., "opening_hours", "address", "contact"
    value = Column(Text, nullable=False)
    value_hindi = Column(Text)  # Hindi translation
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pipeline Session (for logging)
class PipelineSession(Base):
    __tablename__ = "pipeline_sessions"
    
    id = Column(String, primary_key=True)
    request_id = Column(String, unique=True, nullable=False)
    audio_filename = Column(String)
    transcribed_text = Column(Text)
    llm_response = Column(Text)
    db_context_used = Column(Text)  # What DB data was retrieved
    audio_input_path = Column(String)
    audio_output_path = Column(String)
    audio_duration_seconds = Column(Float)
    latency_asr_ms = Column(Float)
    latency_llm_ms = Column(Float)
    latency_tts_ms = Column(Float)
    total_latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String, default="completed")


def init_db():
    """Initialize database tables."""
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Database initialization warning (tables may already exist): {e}")
        # Don't raise - allow service to continue


def get_db():
    """Database session generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()