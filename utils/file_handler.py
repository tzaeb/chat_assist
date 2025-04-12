import streamlit as st
import json
import docx
import pdfplumber
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any, Union


class FileHandler:
    """
    Handles file uploads and content extraction for various file types.
    """

    # Supported file types and their descriptions
    SUPPORTED_TYPES = {
        "txt": "Text files",
        "yml": "YAML files",
        "py": "Python files",
        "ini": "INI files",
        "md": "Markdown files",
        "yaml": "YAML files",
        "json": "JSON files",
        "docx": "Word documents",
        "pdf": "PDF documents",
        "csv": "CSV spreadsheets",
        "xlsx": "Excel spreadsheets",
    }

    @staticmethod
    def get_supported_types() -> List[str]:
        """Get a list of supported file extensions."""
        return list(FileHandler.SUPPORTED_TYPES.keys())

    @staticmethod
    def get_file_uploader(
        label: str = "Upload a file for additional context",
    ) -> Optional[Any]:
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
            help=f"Supported formats: {', '.join(FileHandler.SUPPORTED_TYPES.values())}",
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
        file_name = uploaded_file.name

        try:
            # Plain text files
            if file_type == "text/plain":
                return uploaded_file.read().decode("cp1252", errors="replace")

            # JSON files
            elif file_type == "application/json":
                return json.dumps(json.load(uploaded_file))

            # Word documents
            elif (
                file_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                doc = docx.Document(uploaded_file)
                return "\n".join(para.text for para in doc.paragraphs)

            # PDF documents
            elif file_type == "application/pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)

            # CSV files
            elif file_type == "text/csv" or file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                # Show a preview in the UI
                with st.expander("CSV Preview", expanded=False):
                    st.dataframe(df.head(10))
                # Return a string representation
                return df.to_string(index=False)

            # Excel files
            elif (
                file_type
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                or file_name.endswith(".xlsx")
            ):
                # Read all sheets
                sheet_to_df_map = pd.read_excel(uploaded_file, sheet_name=None)

                # Show a preview in the UI with tabs for each sheet
                with st.expander("Excel Preview", expanded=False):
                    if len(sheet_to_df_map) > 1:
                        sheet_tabs = st.tabs(list(sheet_to_df_map.keys()))
                        for i, (sheet_name, df) in enumerate(sheet_to_df_map.items()):
                            with sheet_tabs[i]:
                                st.write(f"Sheet: {sheet_name}")
                                st.dataframe(df.head(10))
                    else:
                        # Just one sheet
                        sheet_name = list(sheet_to_df_map.keys())[0]
                        st.write(f"Sheet: {sheet_name}")
                        st.dataframe(sheet_to_df_map[sheet_name].head(10))

                # Return a string representation of all sheets
                result = []
                for sheet_name, df in sheet_to_df_map.items():
                    result.append(f"--- Sheet: {sheet_name} ---")
                    result.append(df.to_string(index=False))
                    result.append("\n")

                return "\n".join(result)

            # Unsupported file types
            else:
                # Attempt to read as plain text as a last resort if type is unknown but extension is supported
                if any(
                    file_name.endswith(f".{ext}") for ext in FileHandler.SUPPORTED_TYPES
                ):
                    try:
                        return uploaded_file.read().decode("utf-8", errors="replace")
                    except Exception as decode_error:
                        st.warning(
                            f"Could not decode file {file_name} as text: {decode_error}"
                        )
                        return ""
                else:
                    st.warning(
                        f"Unsupported file type: {file_type} for file {file_name}"
                    )
                    return ""

        except Exception as e:
            st.error(f"Error extracting content from file {file_name}: {str(e)}")
            return ""

    @staticmethod
    def get_buffered_content(uploaded_file: Any) -> str:
        """
        Get buffered content for the uploaded file.
        Only uses buffered content if the exact same file is uploaded again.
        Resets buffer when no file is uploaded or a different file is uploaded.

        Args:
            uploaded_file: The uploaded file object.

        Returns:
            The buffered content as a string.
        """
        # Initialize session state variables if they don't exist
        if "file_metadata" not in st.session_state:
            st.session_state.file_metadata = {"name": None, "size": None}
            st.session_state.file_content = ""

        # If no file is uploaded, reset buffer and return empty string
        if uploaded_file is None:
            st.session_state.file_metadata = {"name": None, "size": None}
            st.session_state.file_content = ""
            return ""

        file_name = uploaded_file.name
        file_size = uploaded_file.size

        # If it's a different file, update buffer with new content
        if (
            st.session_state.file_metadata["name"] != file_name
            or st.session_state.file_metadata["size"] != file_size
        ):
            st.session_state.file_metadata = {"name": file_name, "size": file_size}
            st.session_state.file_content = FileHandler.extract_content(uploaded_file)

        return st.session_state.file_content

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
