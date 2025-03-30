import streamlit as st
import json
import docx
import pdfplumber
from typing import Tuple, Optional, List, Dict, Any, Union


class FileHandler:
    """
    Handles file uploads and content extraction for various file types.
    """
    
    # Supported file types and their descriptions
    SUPPORTED_TYPES = {
        "txt": "Text files",
        "json": "JSON files",
        "docx": "Word documents",
        "pdf": "PDF documents"
    }
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """Get a list of supported file extensions."""
        return list(FileHandler.SUPPORTED_TYPES.keys())
    
    @staticmethod
    def get_file_uploader(label: str = "Upload a file for additional context") -> Optional[Any]:
        """
        Create a Streamlit file uploader widget for supported file types.
        
        Args:
            label: The label to display for the file uploader.
            
        Returns:
            The uploaded file object, or None if no file was uploaded.
        """
        return st.file_uploader(
            label,
            type=FileHandler.get_supported_types(),
            help=f"Supported formats: {', '.join(FileHandler.SUPPORTED_TYPES.values())}"
        )
    
    @staticmethod
    def extract_content(uploaded_file: Any) -> str:
        """
        Extract content from an uploaded file based on its type.
        
        Args:
            uploaded_file: The uploaded file object.
            
        Returns:
            The extracted content as a string.
        """
        if uploaded_file is None:
            return ""
            
        file_type = uploaded_file.type
        
        try:
            # Plain text files
            if file_type == "text/plain":
                return uploaded_file.read().decode("cp1252", errors="replace")
                
            # JSON files
            elif file_type == "application/json":
                return json.dumps(json.load(uploaded_file))
                
            # Word documents
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                return "\n".join(para.text for para in doc.paragraphs)
                
            # PDF documents
            elif file_type == "application/pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)
            
            # Unsupported file types
            else:
                st.warning(f"Unsupported file type: {file_type}")
                return ""
                
        except Exception as e:
            st.error(f"Error extracting content from file: {str(e)}")
            return ""
    
    @staticmethod
    def upload_and_process() -> Tuple[Optional[Any], str]:
        """
        Handle file upload and content extraction in a single method.
        
        Returns:
            A tuple containing the uploaded file object and the extracted content.
        """
        uploaded_file = FileHandler.get_file_uploader()
        
        if uploaded_file is not None:
            file_content = FileHandler.extract_content(uploaded_file)
            return uploaded_file, file_content
        
        return None, "" 