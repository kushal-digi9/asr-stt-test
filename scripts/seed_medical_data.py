#!/usr/bin/env python3
"""
Seed PostgreSQL with dummy medical receptionist data.
Run this after database is initialized.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SessionLocal, init_db, Department, Doctor, Service, Appointment, HospitalInfo
from datetime import datetime, time, timedelta

def seed_data():
    """Seed database with dummy medical data."""
    db = SessionLocal()
    
    try:
        print("🌱 Seeding medical receptionist data...")
        
        # 1. Hospital Info
        hospital_info = [
            {"key": "opening_hours", "value": "8:00 AM - 8:00 PM", "value_hindi": "सुबह 8 बजे से रात 8 बजे तक"},
            {"key": "emergency_hours", "value": "24/7", "value_hindi": "24 घंटे"},
            {"key": "contact_phone", "value": "+91-1234567890", "value_hindi": "+91-1234567890"},
            {"key": "address", "value": "123 Medical Street, City", "value_hindi": "123 मेडिकल स्ट्रीट, शहर"},
            {"key": "email", "value": "info@hospital.com", "value_hindi": "info@hospital.com"}
        ]
        
        for info in hospital_info:
            existing = db.query(HospitalInfo).filter(HospitalInfo.key == info["key"]).first()
            if not existing:
                db.add(HospitalInfo(**info))
        
        # 2. Departments
        departments_data = [
            {"name": "General Medicine", "name_hindi": "सामान्य चिकित्सा", "floor": "1st Floor", "contact_number": "101"},
            {"name": "Cardiology", "name_hindi": "हृदय रोग विभाग", "floor": "2nd Floor", "contact_number": "201"},
            {"name": "Pediatrics", "name_hindi": "बाल रोग विभाग", "floor": "1st Floor", "contact_number": "102"},
            {"name": "Orthopedics", "name_hindi": "अस्थि रोग विभाग", "floor": "3rd Floor", "contact_number": "301"},
            {"name": "Gynecology", "name_hindi": "स्त्री रोग विभाग", "floor": "2nd Floor", "contact_number": "202"}
        ]
        
        departments = []
        for dept_data in departments_data:
            existing = db.query(Department).filter(Department.name == dept_data["name"]).first()
            if not existing:
                dept = Department(**dept_data)
                db.add(dept)
                departments.append(dept)
            else:
                departments.append(existing)
        
        db.commit()
        
        # 3. Doctors
        doctors_data = [
            {"name": "Dr. Rajesh Kumar", "name_hindi": "डॉ. राजेश कुमार", "specialization": "General Physician", 
             "department_id": departments[0].id, "qualification": "MD", "consultation_fee": 500.0,
             "availability_days": "Monday,Wednesday,Friday", "available_from": time(9, 0), "available_to": time(13, 0)},
            
            {"name": "Dr. Priya Sharma", "name_hindi": "डॉ. प्रिया शर्मा", "specialization": "Cardiologist",
             "department_id": departments[1].id, "qualification": "DM Cardiology", "consultation_fee": 1000.0,
             "availability_days": "Tuesday,Thursday", "available_from": time(10, 0), "available_to": time(14, 0)},
            
            {"name": "Dr. Amit Singh", "name_hindi": "डॉ. अमित सिंह", "specialization": "Pediatrician",
             "department_id": departments[2].id, "qualification": "MD Pediatrics", "consultation_fee": 600.0,
             "availability_days": "Monday,Tuesday,Wednesday,Thursday,Friday", "available_from": time(9, 0), "available_to": time(17, 0)},
            
            {"name": "Dr. Sunita Patel", "name_hindi": "डॉ. सुनीता पटेल", "specialization": "Gynecologist",
             "department_id": departments[4].id, "qualification": "MD Gynecology", "consultation_fee": 800.0,
             "availability_days": "Monday,Wednesday,Friday", "available_from": time(11, 0), "available_to": time(15, 0)}
        ]
        
        for doc_data in doctors_data:
            existing = db.query(Doctor).filter(Doctor.name == doc_data["name"]).first()
            if not existing:
                db.add(Doctor(**doc_data))
        
        db.commit()
        
        # 4. Services
        services_data = [
            {"name": "Blood Test", "name_hindi": "रक्त परीक्षण", "price": 300.0, "department_id": departments[0].id},
            {"name": "ECG", "name_hindi": "ईसीजी", "price": 500.0, "department_id": departments[1].id},
            {"name": "X-Ray", "name_hindi": "एक्स-रे", "price": 400.0, "department_id": departments[3].id},
            {"name": "Ultrasound", "name_hindi": "अल्ट्रासाउंड", "price": 800.0, "department_id": departments[4].id}
        ]
        
        for svc_data in services_data:
            existing = db.query(Service).filter(Service.name == svc_data["name"]).first()
            if not existing:
                db.add(Service(**svc_data))
        
        db.commit()
        
        # 5. Sample Appointments
        doctors = db.query(Doctor).all()
        if doctors:
            appointments_data = [
                {"patient_name": "Ramesh Kumar", "patient_phone": "9876543210",
                 "doctor_id": doctors[0].id, "appointment_date": datetime.now() + timedelta(days=1),
                 "appointment_time": time(10, 0), "status": "scheduled"},
                
                {"patient_name": "Meera Devi", "patient_phone": "9876543211",
                 "doctor_id": doctors[1].id, "appointment_date": datetime.now() + timedelta(days=2),
                 "appointment_time": time(11, 0), "status": "scheduled"}
            ]
            
            for apt_data in appointments_data:
                existing = db.query(Appointment).filter(
                    Appointment.patient_name == apt_data["patient_name"],
                    Appointment.appointment_date == apt_data["appointment_date"]
                ).first()
                if not existing:
                    db.add(Appointment(**apt_data))
        
        db.commit()
        
        print("✅ Medical data seeded successfully!")
        print(f"  - Departments: {len(departments)}")
        print(f"  - Doctors: {len(doctors_data)}")
        print(f"  - Services: {len(services_data)}")
        print(f"  - Hospital Info: {len(hospital_info)}")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error seeding data: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    seed_data()