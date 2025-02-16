import re

def extract_image_urls(text):
    return re.findall(r"(https?://\S+\.(?:png|jpg|jpeg|gif))", text)


def format_file_context(file_name, file_content):
    return f"""This is some additional context:
[file name]: {file_name}
[file content begin]
{file_content}
[file content end]"""
