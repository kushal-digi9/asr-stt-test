import logging
import re
from typing import Dict, Any, List, Optional
from utils.database import SessionLocal, Department, Doctor, Service, Appointment, HospitalInfo
from datetime import datetime

logger = logging.getLogger(__name__)


class MedicalDBQuery:
    """Query PostgreSQL database for medical receptionist information."""
    
    def __init__(self):
        try:
            self.db = SessionLocal()
        except Exception as e:
            logger.error(f"❌ Failed to create database session: {e}")
            raise
    
    def query_by_user_question(self, question: str) -> Dict[str, Any]:
        """
        Intelligently query database based on user's question.
        Returns relevant context for LLM.
        """
        question_lower = question.lower()
        context = {
            "departments": [],
            "doctors": [],
            "services": [],
            "appointments": [],
            "hospital_info": {},
            "query_type": None
        }
        
        try:
            # Detect query intent
            if any(word in question_lower for word in ["department", "विभाग", "dept", "speciality", "विशेषज्ञता"]):
                context["query_type"] = "departments"
                context["departments"] = self.get_all_departments()
            
            elif any(word in question_lower for word in ["doctor", "डॉक्टर", "physician", "specialist", "consultant"]):
                context["query_type"] = "doctors"
                # First get all departments to check if department is mentioned
                all_depts = self.get_all_departments()
                
                # Check if specific department is mentioned in question
                department_id = None
                for dept in all_depts:
                    if dept["name"] and dept["name"].lower() in question_lower:
                        department_id = dept["id"]
                        break
                    if dept["name_hindi"] and dept["name_hindi"].lower() in question_lower:
                        department_id = dept["id"]
                        break
                
                # Get doctors - filtered by department if mentioned
                if department_id:
                    context["doctors"] = self.get_doctors_by_department(department_id)
                else:
                    context["doctors"] = self.get_all_doctors()
            
            elif any(word in question_lower for word in ["appointment", "appoint", "booking", "schedule", "अपॉइंटमेंट", "बुक"]):
                context["query_type"] = "appointments"
                # Get upcoming appointments
                context["appointments"] = self.get_upcoming_appointments(limit=10)
            
            elif any(word in question_lower for word in ["service", "service", "test", "परीक्षण", "सेवा"]):
                context["query_type"] = "services"
                context["services"] = self.get_all_services()
            
            elif any(word in question_lower for word in ["time", "timing", "hours", "open", "close", "समय", "खुला", "बंद"]):
                context["query_type"] = "hospital_info"
                context["hospital_info"] = self.get_hospital_info()
            
            elif any(word in question_lower for word in ["contact", "phone", "number", "call", "संपर्क", "फोन"]):
                context["query_type"] = "hospital_info"
                context["hospital_info"] = self.get_hospital_info()
            
            else:
                # Default: Get general hospital info
                context["query_type"] = "general"
                context["departments"] = self.get_all_departments()
                context["hospital_info"] = self.get_hospital_info()
            
            logger.info(f"🔍 Query type detected: {context['query_type']}")
            logger.info(f"🔍 Retrieved: {len(context['departments'])} departments, {len(context['doctors'])} doctors")
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Error querying database: {e}")
            return context
    
    def get_all_departments(self) -> List[Dict[str, Any]]:
        """Get all active departments."""
        depts = self.db.query(Department).filter(Department.is_active == True).all()
        return [{
            "id": d.id,
            "name": d.name,
            "name_hindi": d.name_hindi,
            "description": d.description,
            "floor": d.floor,
            "contact": d.contact_number
        } for d in depts]
    
    def get_doctors_by_department(self, department_id: int) -> List[Dict[str, Any]]:
        """Get doctors in a specific department."""
        doctors = self.db.query(Doctor).filter(
            Doctor.department_id == department_id,
            Doctor.is_active == True
        ).all()
        return [{
            "id": doc.id,
            "name": doc.name,
            "name_hindi": doc.name_hindi,
            "specialization": doc.specialization,
            "qualification": doc.qualification,
            "fee": doc.consultation_fee,
            "availability": doc.availability_days,
            "timings": f"{doc.available_from} - {doc.available_to}" if doc.available_from else None
        } for doc in doctors]
    
    def get_all_doctors(self) -> List[Dict[str, Any]]:
        """Get all active doctors."""
        doctors = self.db.query(Doctor).filter(Doctor.is_active == True).all()
        return [{
            "id": doc.id,
            "name": doc.name,
            "name_hindi": doc.name_hindi,
            "specialization": doc.specialization,
            "department": doc.department.name if doc.department else None,
            "fee": doc.consultation_fee,
            "availability": doc.availability_days
        } for doc in doctors]
    
    def get_all_services(self) -> List[Dict[str, Any]]:
        """Get all active services."""
        services = self.db.query(Service).filter(Service.is_active == True).all()
        return [{
            "id": s.id,
            "name": s.name,
            "name_hindi": s.name_hindi,
            "description": s.description,
            "price": s.price,
            "department": s.department.name if s.department else None
        } for s in services]
    
    def get_upcoming_appointments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get upcoming appointments."""
        now = datetime.now()
        appointments = self.db.query(Appointment).filter(
            Appointment.appointment_date >= now,
            Appointment.status == "scheduled"
        ).order_by(Appointment.appointment_date).limit(limit).all()
        
        return [{
            "id": apt.id,
            "patient_name": apt.patient_name,
            "doctor": apt.doctor.name if apt.doctor else None,
            "date": apt.appointment_date.strftime("%Y-%m-%d") if apt.appointment_date else None,
            "time": apt.appointment_time.strftime("%H:%M") if apt.appointment_time else None,
            "status": apt.status
        } for apt in appointments]
    
    def get_hospital_info(self) -> Dict[str, Any]:
        """Get general hospital information."""
        info_records = self.db.query(HospitalInfo).all()
        return {record.key: {
            "value": record.value,
            "value_hindi": record.value_hindi
        } for record in info_records}
    
    def format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Format database context as text for LLM prompt."""
        lines = []
        
        if context.get("hospital_info"):
            lines.append("=== अस्पताल की जानकारी ===")
            for key, info in context["hospital_info"].items():
                lines.append(f"{key}: {info.get('value_hindi', info.get('value', ''))}")
        
        if context.get("departments"):
            lines.append("\n=== विभाग (Departments) ===")
            for dept in context["departments"]:
                dept_name = dept.get("name_hindi") or dept.get("name", "")
                lines.append(f"- {dept_name}")
                if dept.get("description"):
                    lines.append(f"  {dept['description']}")
        
        if context.get("doctors"):
            lines.append("\n=== डॉक्टर (Doctors) ===")
            for doc in context["doctors"][:5]:  # Limit to 5
                doc_name = doc.get("name_hindi") or doc.get("name", "")
                spec = doc.get("specialization", "")
                dept = doc.get("department", "")
                lines.append(f"- {doc_name} ({spec})")
                if dept:
                    lines.append(f"  विभाग: {dept}")
        
        if context.get("services"):
            lines.append("\n=== सेवाएं (Services) ===")
            for svc in context["services"][:5]:  # Limit to 5
                svc_name = svc.get("name_hindi") or svc.get("name", "")
                price = svc.get("price", "")
                lines.append(f"- {svc_name}")
                if price:
                    lines.append(f"  मूल्य: ₹{price}")
        
        return "\n".join(lines) if lines else ""
    
    def close(self):
        """Close database connection."""
        self.db.close()