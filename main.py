from markitdown import MarkItDown
from ray import serve
from fastapi import FastAPI, UploadFile, Form, HTTPException, BackgroundTasks
from docling.document_converter import DocumentConverter

import os
import ray
from ray import serve
from fastapi import FastAPI, UploadFile, Form, HTTPException
import time
from typing import Optional
from io import BytesIO
import re
import tempfile
from fastapi.responses import FileResponse, Response
# ----------------------------------------
# Init Ray Serve
rest_port = 8000  # Define the port
ray.init()
serve.start(http_options={"host": "0.0.0.0","port": rest_port})
app = FastAPI()

allowed_exts = ["docx", "pdf", "txt", "xlsx", "xls", "csv", "pptx", "ppt"]

def get_file_extension(filename: str) -> str:
    """Get the file extension from a filename."""
    return filename.rsplit(".",1)[-1].lower() if "." in filename else ""

def is_youtube_url(url):
    """Check if a URL is a valid YouTube URL."""
    youtube_regex = r"^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$"
    return bool(re.match(youtube_regex, url))

def extract_youtube_id(url):
    """Extract the YouTube video ID from a URL."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  # Standard YouTube URLs
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",  # Short YouTube URLs
        r"(?:embed\/)([0-9A-Za-z_-]{11})",  # Embedded YouTube URLs
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None

def check_extension_allowed(filename: str, allowed_exts: list) -> bool:
    """Check if the file extension is in the allowed list."""
    ext = get_file_extension(filename)
    return ext in allowed_exts
# ----------------------------------------
class ConverterWorker:
    def __init__(self):
        self.markitdown = MarkItDown(enable_plugins=False)
        self.docling = DocumentConverter()
        print("✅ Converter already")
        
    def convert_markdown(self, type: str, file_path: str):
        """Convert a file to markdown using MarkItDown or Docling."""
        if type == "markitdown":
            return self.markitdown.convert(file_path).text_content
        else:
            return self.docling.convert(file_path).document.export_to_markdown()
# ----------------------------------------
@serve.deployment( num_replicas=3,ray_actor_options={"num_gpus": 1/3})
@serve.ingress(app)
class ConverterDeployment:
    def __init__(self):
        self.worker = ConverterWorker()
        print("✅ WhisperDeployment ready")

    @app.post("/convert")
    async def convert(self, file: UploadFile, type: Optional[str] = Form(None), background_tasks: BackgroundTasks = None):
        if not type or type not in ["markitdown", "docling"]:
            raise HTTPException(status_code=400, detail="Invalid type. Must be 'markitdown' or 'docling'.")
        # check file extension
        filename = file.filename.lower()
        ext = get_file_extension(filename)
        if not ext in allowed_exts:
            raise HTTPException(status_code=400, detail=f"Supported {[x for x in allowed_exts]} files are allowed.")
        file_data = await file.read()
        if not file_data:
            raise HTTPException(status_code=400, detail="Empty file or upload failed.")
        # Using pydub calculation duration
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
                tmp_file.write(file_data)
                tmp_file_path = tmp_file.name
            
            # Convert file to markdown
            result = self.worker.convert_markdown(type, tmp_file_path)
            os.unlink(tmp_file_path)  # Clean up the input temporary file
            
            # Create a temporary markdown file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as md_file:
                md_file.write(result.encode('utf-8'))
                md_file_path = md_file.name
            
            # Return FileResponse with background task to clean up the file
            if background_tasks:
                background_tasks.add_task(cleanup_file, md_file_path)
            
            response = FileResponse(
                path=md_file_path,
                filename=f"{os.path.splitext(filename)[0]}.md",
                media_type="text/markdown"
            )
            return response
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed convert file: {str(e)}")

    # Alternative approach - return content directly without temporary files
    @app.post("/convert-direct")
    async def convert_direct(self, file: UploadFile, type: Optional[str] = Form(None)):
        """Alternative endpoint that returns content directly without temporary files."""
        if not type or type not in ["markitdown", "docling"]:
            raise HTTPException(status_code=400, detail="Invalid type. Must be 'markitdown' or 'docling'.")
        
        # check file extension
        filename = file.filename.lower()
        ext = get_file_extension(filename)
        print(f"File extension: {ext}")
        if not ext in allowed_exts:
            raise HTTPException(status_code=400, detail=f"Supported {[x for x in allowed_exts]} files are allowed.")
        
        file_data = await file.read()
        if not file_data:
            raise HTTPException(status_code=400, detail="Empty file or upload failed.")
        
        try:
            # Create temporary file for input
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
                tmp_file.write(file_data)
                tmp_file_path = tmp_file.name
            
            # Convert file to markdown
            result = self.worker.convert_markdown(type, tmp_file_path)
            os.unlink(tmp_file_path)  # Clean up the input temporary file immediately
            
            # Return markdown content directly
            return Response(
                content=result.encode('utf-8'),
                media_type="text/markdown",
                headers={
                    "Content-Disposition": f"attachment; filename={os.path.splitext(filename)[0]}.md"
                }
            )
        except Exception as e:
            # Clean up in case of error
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            raise HTTPException(status_code=400, detail=f"Failed convert file: {str(e)}")

def cleanup_file(file_path: str):
    """Clean up temporary files."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Warning: Could not delete temporary file {file_path}: {e}")

# ----------------------------------------
serve.run(ConverterDeployment.bind(),route_prefix="/v1")
print(f"✅ Ray Serve FastAPI running at http://0.0.0.0:{rest_port}/v1/convert-direct")
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("⏹️ Stopped by user.")