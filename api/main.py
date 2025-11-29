"""
FastAPI backend for EthoScore Article Analysis
Version: 2.3.0 - Added /analyze/url and /analyze/text endpoints
"""

import os
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Try to import gdown for Google Drive downloads (handles virus scan automatically)
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    logging.warning("gdown not available, will use fallback method")

# Import our model analyzer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_analyzer import ArticleFramingAnalyzer, ModelLoadingError, ModelInferenceError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global analyzer and dataset instances
analyzer: Optional[ArticleFramingAnalyzer] = None
dataset_loaded: bool = False
_dataset_path: Optional[Path] = None


def check_dataset_exists() -> bool:
    """
    Check if the dataset file exists (without loading it into memory).
    """
    global dataset_loaded, _dataset_path

    base_dir = Path(__file__).parent.parent
    dataset_path = base_dir / "Dataset-framing_annotations-Llama-3.3-70B-Instruct-Turbo.csv"
    _dataset_path = dataset_path

    if dataset_path.exists():
        file_size = dataset_path.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info(f"Dataset file found: {dataset_path} ({file_size:.1f} MB)")
        dataset_loaded = True
        return True
    else:
        logger.warning(f"Dataset file not found at {dataset_path}")
        dataset_loaded = False
        return False


def search_dataset_streaming(
    topic: str,
    limit: int = 3,
    label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search the dataset using streaming CSV reading (memory efficient).
    
    Uses chunked reading to avoid loading the entire 810MB file into memory.
    Searches in concept, title, and body columns.
    Filters by FRAMING_CLASS if label is provided.
    """
    import csv
    global _dataset_path

    if not topic:
        return []

    if _dataset_path is None or not _dataset_path.exists():
        if not check_dataset_exists():
            logger.warning("Dataset not available for topic search")
            return []

    topic_str = str(topic).strip()
    topic_lower = topic_str.lower()
    label_upper = str(label).strip().upper() if label else None

    logger.info(f"Streaming search for topic '{topic_str}'" + (f" with label '{label_upper}'" if label_upper else ""))

    results: List[Dict[str, Any]] = []

    try:
        with open(_dataset_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if len(results) >= limit:
                    break

                # Check if topic matches in concept, title, or body
                concept = str(row.get('concept', '')).lower()
                title = str(row.get('title', '')).lower()
                body = str(row.get('body', '')).lower()
                
                topic_match = (
                    topic_lower in concept or 
                    topic_lower in title or 
                    topic_lower in body
                )
                
                if not topic_match:
                    continue

                # Check framing label filter if provided
                if label_upper:
                    framing_class = str(row.get('FRAMING_CLASS', '')).strip().upper()
                    if framing_class != label_upper:
                        continue

                # Build result record
                record: Dict[str, Any] = {}
                
                # Title
                raw_title = row.get('title', '').strip()
                record["title"] = raw_title if raw_title else topic_str
                
                # Body preview (first 2000 chars)
                raw_body = row.get('body', '')
                if raw_body:
                    record["body_preview"] = raw_body[:2000]
                
                # Source
                raw_source = row.get('source', '')
                if raw_source:
                    # Handle if source is a JSON string (dict format)
                    if raw_source.startswith('{'):
                        try:
                            import json
                            source_dict = json.loads(raw_source.replace("'", '"'))
                            record["source"] = source_dict.get('title') or source_dict.get('uri', raw_source)
                        except:
                            record["source"] = raw_source
                    else:
                        record["source"] = raw_source
                
                # Concept/topic category
                raw_concept = row.get('concept', '').strip()
                if raw_concept:
                    record["concept"] = raw_concept
                
                # URL if available
                raw_url = row.get('url', '').strip()
                if raw_url:
                    record["source_url"] = raw_url
                
                # Framing label
                framing_class = row.get('FRAMING_CLASS', '').strip()
                if framing_class:
                    framing_label_display = framing_class.title()
                    record["ordinal_analysis"] = {"predicted_label": framing_label_display}
                    record["classification_analysis"] = {"predicted_label": framing_label_display}
                    record["framing_class"] = framing_label_display
                
                results.append(record)

        logger.info(f"Topic search for '{topic_str}' returning {len(results)} results (limit={limit})")
        return results

    except Exception as e:
        logger.error(f"Error during streaming search: {e}")
        return []


# Alias for backward compatibility
def search_dataset(
    topic: str,
    limit: int = 3,
    label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search dataset using memory-efficient streaming."""
    return search_dataset_streaming(topic, limit, label)

def download_file_from_google_drive_gdown(file_id: str, destination: str) -> bool:
    """
    Download a file from Google Drive using gdown library (handles virus scan automatically).
    
    Args:
        file_id: Google Drive file ID
        destination: Local file path to save to
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not GDOWN_AVAILABLE:
        logger.warning("gdown not available, cannot use this download method")
        return False
        
    try:
        logger.info(f"Downloading with gdown: {file_id} -> {destination}")
        
        # gdown URL format
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Download with gdown (it handles virus scan automatically)
        output = gdown.download(url, destination, quiet=False, fuzzy=True)
        
        if output is None:
            logger.error("gdown returned None - download failed")
            return False
        
        # Verify file exists and has reasonable size
        if not os.path.exists(destination):
            logger.error(f"File not found after download: {destination}")
            return False
        
        file_size = os.path.getsize(destination)
        logger.info(f"Successfully downloaded with gdown: {destination} ({file_size / (1024*1024):.2f} MB)")
        
        # Verify file is not too small (likely an error)
        if destination.endswith('.safetensors') and file_size < 1000000:  # Less than 1MB
            logger.error(f"Downloaded file is suspiciously small ({file_size} bytes)")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading with gdown: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def download_file_from_google_drive(file_id: str, destination: str) -> bool:
    """
    Download a file from Google Drive, handling large file virus scan warnings.
    Uses gdown if available (preferred), falls back to manual method.
    
    Args:
        file_id: Google Drive file ID
        destination: Local file path to save to
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Try gdown first (it handles virus scan automatically)
    if GDOWN_AVAILABLE:
        logger.info("Attempting download with gdown (preferred method)...")
        if download_file_from_google_drive_gdown(file_id, destination):
            return True
        logger.warning("gdown download failed, trying fallback method...")
    
    # Fallback to manual method
    try:
        logger.info(f"Downloading file from Google Drive (fallback): {file_id} -> {destination}")
        
        # Google Drive download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        session = requests.Session()
        
        # First request to get the page/file
        response = session.get(url, stream=True)
        
        # Check if we got a virus scan warning page
        content_type = response.headers.get('content-type', '')
        
        if 'text/html' in content_type:
            logger.info("Received HTML page (likely virus scan warning), extracting confirmation...")
            
            # Read the HTML content to extract the confirmation token
            html_content = response.text
            
            # Try multiple patterns to extract the confirm token
            import re
            
            patterns = [
                r'confirm=([a-zA-Z0-9_-]+)',
                r'id="download-form"[^>]*action="[^"]*confirm=([a-zA-Z0-9_-]+)',
                r'"downloadUrl":"[^"]*confirm=([a-zA-Z0-9_-]+)',
            ]
            
            confirm_token = None
            for pattern in patterns:
                match = re.search(pattern, html_content)
                if match:
                    confirm_token = match.group(1)
                    logger.info(f"Found confirmation token using pattern")
                    break
            
            # Check cookies as well
            if not confirm_token:
                for key, value in response.cookies.items():
                    if 'download_warning' in key.lower() or 'confirm' in key.lower():
                        confirm_token = value
                        logger.info(f"Found confirmation token in cookies")
                        break
            
            if confirm_token:
                # Make a new request with the confirmation token
                logger.info(f"Using confirmation token...")
                params = {'id': file_id, 'confirm': confirm_token}
                response = session.get(url, params=params, stream=True)
            else:
                # Try confirm=t (works for some files)
                logger.info("No token found, trying confirm=t...")
                params = {'id': file_id, 'confirm': 't'}
                response = session.get(url, params=params, stream=True)
        
        # Check if successful
        if response.status_code != 200:
            logger.error(f"Failed to download file. Status code: {response.status_code}")
            return False
        
        # Verify we're getting binary content, not HTML error page
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            logger.error(f"Received HTML instead of binary file. Download may have failed.")
            return False
        
        # Save file in chunks
        downloaded_size = 0
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
        
        file_size = os.path.getsize(destination)
        logger.info(f"Downloaded {destination} ({file_size / (1024*1024):.2f} MB)")
        
        # Verify file is not too small (likely an error page)
        if destination.endswith('.safetensors') and file_size < 1000000:  # Less than 1MB
            logger.error(f"Downloaded file is suspiciously small ({file_size} bytes). May be an error page.")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return False

def download_models() -> bool:
    """
    Download model files from Google Drive if they don't exist locally.
    Uses environment variables for Google Drive file IDs or URLs.
    
    Supported env variable formats:
    - ORDINAL_MODEL_ID / ORDINAL_MODEL_URL / MODEL_URL_ORDINAL_MODEL_BEST_CHECKPOINT_SAFETENSORS
    - CLASS_3_MODEL_ID / CLASS_3_MODEL_URL / MODEL_URL_3CLASS_MODEL_BEST_CHECKPOINT_SAFETENSORS
    - DATASET_ID / DATASET_URL / MODEL_URL_DATASET_FRAMING_ANNOTATIONS_LLAMA_3_3_70B_INSTRUCT_TURBO_CSV
    
    Returns:
        bool: True if all models are available, False otherwise
    """
    try:
        base_dir = Path(__file__).parent.parent
        
        models_to_download = [
            {
                'name': 'ordinal_model_best_checkpoint.safetensors',
                'env_vars': [
                    'ORDINAL_MODEL_ID',
                    'ORDINAL_MODEL_URL',
                    'MODEL_URL_ORDINAL_MODEL_BEST_CHECKPOINT_SAFETENSORS'
                ],
            },
            {
                'name': '3class_model_best_checkpoint.safetensors',
                'env_vars': [
                    'CLASS_3_MODEL_ID',
                    'CLASS_3_MODEL_URL',
                    'MODEL_URL_3CLASS_MODEL_BEST_CHECKPOINT_SAFETENSORS'
                ],
            },
            {
                'name': 'Dataset-framing_annotations-Llama-3.3-70B-Instruct-Turbo.csv',
                'env_vars': [
                    'DATASET_ID',
                    'DATASET_URL',
                    'MODEL_URL_DATASET_FRAMING_ANNOTATIONS_LLAMA_3_3_70B_INSTRUCT_TURBO_CSV'
                ],
            }
        ]
        
        all_successful = True
        
        for model_info in models_to_download:
            file_path = base_dir / model_info['name']
            
            # Skip if file already exists and is not empty
            if file_path.exists() and file_path.stat().st_size > 1000:
                logger.info(f"Model file already exists: {model_info['name']}")
                continue
            
            # Try all possible environment variable names
            file_id = None
            file_url = None
            
            for env_var in model_info['env_vars']:
                value = os.getenv(env_var)
                if value:
                    logger.info(f"Found environment variable {env_var} for {model_info['name']}")
                    if 'http' in value or 'drive.google.com' in value:
                        file_url = value
                    else:
                        file_id = value
                    break
            
            if file_id:
                logger.info(f"Downloading {model_info['name']} using file ID: {file_id[:20]}...")
                success = download_file_from_google_drive(file_id, str(file_path))
                if not success:
                    logger.error(f"Failed to download {model_info['name']}")
                    all_successful = False
            elif file_url:
                # Extract file ID from Google Drive URL
                extracted_id = None
                
                # Handle different URL formats
                if 'drive.google.com' in file_url:
                    # Format 1: https://drive.google.com/file/d/FILE_ID/view
                    if '/d/' in file_url:
                        extracted_id = file_url.split('/d/')[1].split('/')[0]
                    # Format 2: https://drive.google.com/uc?export=download&id=FILE_ID
                    elif 'id=' in file_url:
                        extracted_id = file_url.split('id=')[1].split('&')[0]
                
                if extracted_id:
                    logger.info(f"Extracted file ID from URL: {extracted_id[:20]}...")
                    success = download_file_from_google_drive(extracted_id, str(file_path))
                    if not success:
                        logger.error(f"Failed to download {model_info['name']}")
                        all_successful = False
                else:
                    logger.error(f"Could not parse Google Drive URL: {file_url}")
                    all_successful = False
            else:
                logger.warning(f"No environment variable found for {model_info['name']} "
                             f"(tried {', '.join(model_info['env_vars'])})")
                all_successful = False
        
        return all_successful
        
    except Exception as e:
        logger.error(f"Error in download_models: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global analyzer, dataset_loaded
    
    logger.info("[startup] Starting EthoScore API v2.2.0 (memory-efficient streaming)...")
    
    # Try to download models
    logger.info("[startup] Checking for model files...")
    models_available = download_models()
    
    if not models_available:
        logger.warning("[startup] WARNING: Could not download model files. Models may not work.")
    
    # Try to initialize models
    try:
        base_dir = Path(__file__).parent.parent
        ordinal_path = base_dir / "ordinal_model_best_checkpoint.safetensors"
        class_path = base_dir / "3class_model_best_checkpoint.safetensors"
        dataset_path = base_dir / "Dataset-framing_annotations-Llama-3.3-70B-Instruct-Turbo.csv"
        
        if ordinal_path.exists() and class_path.exists():
            logger.info("[startup] Initializing models...")
            analyzer = ArticleFramingAnalyzer(
                ordinal_checkpoint=str(ordinal_path),
                class_checkpoint=str(class_path),
                device_map="auto"
            )
            analyzer.initialize_models()
            logger.info("[startup] Models initialized successfully!")
        else:
            logger.error(f"[startup] Model files not found: ordinal={ordinal_path.exists()}, class={class_path.exists()}")
            
        # Check dataset exists (don't load into memory - use streaming search)
        if dataset_path.exists():
            file_size = dataset_path.stat().st_size / (1024 * 1024)
            logger.info(f"[startup] Dataset file found ({file_size:.1f} MB) - will use streaming search")
            check_dataset_exists()
        else:
            logger.warning("[startup] Dataset file not found")
            
    except Exception as e:
        logger.error(f"[startup] Model loading failed: {str(e)}")
    
    logger.info("[startup] Server startup complete")
    
    yield
    
    logger.info("[shutdown] Shutting down EthoScore API...")

# Create FastAPI app
app = FastAPI(
    title="EthoScore API",
    description="Article Framing Analysis API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AnalyzeRequest(BaseModel):
    title: str = Field(..., description="Article title")
    body: str = Field(..., description="Article body text")

class AnalyzeURLRequest(BaseModel):
    url: str = Field(..., description="URL of article to analyze")

class AnalyzeTextRequest(BaseModel):
    title: str = Field("", description="Article title (optional)")
    body: str = Field(..., description="Article body text")
    text: Optional[str] = Field(None, description="Alias for body text")
    
    def get_body(self) -> str:
        """Get the body text, preferring body over text"""
        return self.body or self.text or ""

class AnalyzeResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TopicRequest(BaseModel):
    keyword: Optional[str] = Field(None, description="Keyword to search for in concept, title, and body")
    topic: Optional[str] = Field(None, description="Topic to explore (alias for keyword)")
    limit: Optional[int] = Field(3, description="Number of results to return (default: 3)")
    label: Optional[str] = Field(None, description="Filter by framing label: NEUTRAL, LOADED, or ALARMIST")
    
    def get_topic(self) -> str:
        """Get the topic/keyword value, preferring keyword over topic"""
        return self.keyword or self.topic or ""

class HealthResponse(BaseModel):
    ok: bool
    models: Dict[str, bool]
    dataset_loaded: bool

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "EthoScore API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global analyzer, dataset_loaded
    
    return HealthResponse(
        ok=analyzer is not None and analyzer.is_initialized,
        models={
            "is_initialized": analyzer is not None and analyzer.is_initialized
        },
        dataset_loaded=dataset_loaded
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_article(request: AnalyzeRequest):
    """Analyze an article for framing bias"""
    global analyzer
    
    if analyzer is None or not analyzer.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Models not initialized. Please check server logs and ensure model files are available."
        )
    
    try:
        result = analyzer.analyze_article(request.title, request.body)
        return AnalyzeResponse(success=True, data=result)
    except ModelInferenceError as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/analyze/url", response_model=AnalyzeResponse)
async def analyze_url(request: AnalyzeURLRequest):
    """Analyze an article from URL for framing bias"""
    global analyzer
    
    if analyzer is None or not analyzer.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Models not initialized. Please check server logs and ensure model files are available."
        )
    
    try:
        # Import article processor for URL extraction
        from src.article_processor import extract_article_from_url, ArticleExtractionError
        
        logger.info(f"Extracting article from URL: {request.url}")
        title, body, metadata = extract_article_from_url(request.url)
        
        logger.info(f"Analyzing extracted article: '{title[:50]}...'")
        result = analyzer.analyze_article(title, body)
        
        # Add metadata to result
        result["metadata"] = metadata
        result["source_url"] = request.url
        
        return AnalyzeResponse(success=True, data=result)
    except ArticleExtractionError as e:
        logger.error(f"Article extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract article: {str(e)}")
    except ModelInferenceError as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")


@app.post("/analyze/text", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeTextRequest):
    """Analyze pasted text for framing bias"""
    global analyzer
    
    if analyzer is None or not analyzer.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Models not initialized. Please check server logs and ensure model files are available."
        )
    
    body = request.get_body()
    if not body or len(body.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Text body is required and must be at least 10 characters"
        )
    
    title = request.title.strip() if request.title else "Untitled Article"
    
    try:
        logger.info(f"Analyzing pasted text: '{title[:50]}...' ({len(body)} chars)")
        result = analyzer.analyze_article(title, body)
        return AnalyzeResponse(success=True, data=result)
    except ModelInferenceError as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/explore/topic")
async def explore_topic(request: TopicRequest):
    """Explore articles related to a topic"""
    global analyzer, dataset_loaded
    
    if analyzer is None or not analyzer.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Models not initialized"
        )
    
    if not dataset_loaded:
        raise HTTPException(
            status_code=503,
            detail="Dataset not loaded"
        )
    
    topic = request.get_topic()
    if not topic:
        raise HTTPException(
            status_code=400,
            detail="Either 'keyword' or 'topic' must be provided"
        )

    # Use the precomputed dataset for topic exploration
    limit = request.limit or 3
    label_filter = request.label

    results = search_dataset(topic, limit=limit, label=label_filter)

    return {
        "success": True,
        "topic": topic,
        "results": results,
        "count": len(results),
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    global analyzer
    
    if analyzer is None:
        return {
            "initialized": False,
            "error": "Models not initialized"
        }
    
    return analyzer.get_model_info()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
