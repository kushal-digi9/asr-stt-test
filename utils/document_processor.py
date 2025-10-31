import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pypdf
import docx
from PIL import Image
import pytesseract
import io

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process various document formats and extract text."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"📄 Document processor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document file and extract text.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dict with extracted text and metadata
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"📄 Processing file: {path.name}")
        
        # Route to appropriate processor
        if path.suffix.lower() == '.pdf':
            return self._process_pdf(file_path)
        elif path.suffix.lower() in ['.docx', '.doc']:
            return self._process_docx(file_path)
        elif path.suffix.lower() == '.txt':
            return self._process_txt(file_path)
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF."""
        logger.info(f"📄 Processing PDF: {file_path}")
        
        text_content = []
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():
                    text_content.append(text)
        
        full_text = "\n\n".join(text_content)
        
        return {
            "text": full_text,
            "metadata": {
                "source": file_path,
                "type": "pdf",
                "pages": num_pages,
                "char_count": len(full_text)
            }
        }
    
    def _process_docx(self, file_path: str) -> Dict[str, Any]:
        """Extract text from DOCX."""
        logger.info(f"📄 Processing DOCX: {file_path}")
        
        doc = docx.Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = "\n\n".join(paragraphs)
        
        return {
            "text": full_text,
            "metadata": {
                "source": file_path,
                "type": "docx",
                "paragraphs": len(paragraphs),
                "char_count": len(full_text)
            }
        }
    
    def _process_txt(self, file_path: str) -> Dict[str, Any]:
        """Extract text from TXT."""
        logger.info(f"📄 Processing TXT: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            full_text = file.read()
        
        return {
            "text": full_text,
            "metadata": {
                "source": file_path,
                "type": "txt",
                "char_count": len(full_text)
            }
        }
    
    def _process_image(self, file_path: str) -> Dict[str, Any]:
        """Extract text from image using OCR."""
        logger.info(f"📄 Processing image with OCR: {file_path}")
        
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang='hin+eng')
        
        return {
            "text": text,
            "metadata": {
                "source": file_path,
                "type": "image_ocr",
                "image_size": image.size,
                "char_count": len(text)
            }
        }
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunks with metadata
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for delimiter in ['. ', '। ', '\n\n', '\n']:
                    last_delim = text.rfind(delimiter, start, end)
                    if last_delim != -1:
                        end = last_delim + len(delimiter)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_data = {
                    "text": chunk_text,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": len(chunks),
                        "start_char": start,
                        "end_char": end
                    }
                }
                chunks.append(chunk_data)
            
            start = end - self.chunk_overlap
        
        logger.info(f"📄 Created {len(chunks)} chunks from text")
        return chunks
    
    def process_and_chunk(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a file and return chunked text.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of text chunks with metadata
        """
        # Extract text
        result = self.process_file(file_path)
        
        # Chunk text
        chunks = self.chunk_text(result["text"], result["metadata"])
        
        logger.info(f"✅ Processed {file_path}: {len(chunks)} chunks")
        return chunks
