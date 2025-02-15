import re

def extract_image_urls(text):
    return re.findall(r"(https?://\S+\.(?:png|jpg|jpeg|gif))", text)


def format_file_context(file_name, file_content):
    return f"""Additional context:\n
[file name]: {file_name}
[file content begin]
{file_content}
[file content end]
"""


def chunk_text_by_sections(text, max_tokens=256):
    """
    Splits text into chunks by first dividing it into paragraphs (using double newlines)
    and then grouping paragraphs together while ensuring each chunk stays within max_tokens.
    
    A paragraph that begins with a heading indicator (e.g. "###", "**", or a capitalized title)
    starts a new chunk. Section dividers (three or more dashes) are used to flush the current chunk.
    
    If the document starts with a title (e.g. bold text) and the following chunk starts with a section
    heading, the title is removed.
    """
    # Split text into paragraphs by detecting double newlines (allowing for spaces)
    paragraphs = re.split(r'\r?\n\s*\r?\n', text)
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    # Patterns to detect headings and dividers.
    heading_pattern = re.compile(r'^(###|\*\*|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)')
    divider_pattern = re.compile(r'^-{3,}$')
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If the paragraph is a section divider, flush the current chunk.
        if divider_pattern.fullmatch(para):
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            continue
        
        # Count tokens approximated by whitespace splitting.
        tokens = para.split()
        token_count = len(tokens)
        
        # If the paragraph starts with a heading, treat it as a boundary.
        if heading_pattern.match(para):
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            current_chunk.append(para)
            current_token_count += token_count
        else:
            # If adding this paragraph would exceed the max_tokens, flush the current chunk.
            if current_token_count + token_count > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [para]
                current_token_count = token_count
            else:
                current_chunk.append(para)
                current_token_count += token_count
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Optional: Remove a leading title if it's a standalone paragraph
    # and the following chunk starts with a section heading.
    if (chunks and len(chunks) > 1 and 
        not chunks[0].lstrip().startswith("###") and 
        heading_pattern.match(chunks[1].lstrip())):
        chunks.pop(0)
    
    return chunks
