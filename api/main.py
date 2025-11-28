"""
FastAPI backend for EthoScore Article Analysis
"""

import os
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
from src.article_processor import extract_article_from_url, ArticleExtractionError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global analyzer instance
analyzer: Optional[ArticleFramingAnalyzer] = None
dataset_loaded = False

def download_file_from_google_drive_gdown(file_id: str, destination: str) -> bool:
    """
    Download a file from Google Drive using gdown library (handles virus scan automatically).
    
    Args:
        file_id: Google Drive file ID
        destination: Local file path to save to
        
    Returns:
        bool: True if successful, False otherwise
    """
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
    
    Args:
        file_id: Google Drive file ID
        destination: Local file path to save to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading file from Google Drive: {file_id} -> {destination}")
        
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
            
            # Pattern 1: Look for confirm parameter in forms or links
            patterns = [
                r'confirm=([a-zA-Z0-9_-]+)',
                r'id="download-form"[^>]*action="[^"]*confirm=([a-zA-Z0-9_-]+)',
                r'"downloadUrl":"[^"]*confirm=([a-zA-Z0-9_-]+)',
                r'confirm=([^&"\s]+)',
            ]
            
            confirm_token = None
            for pattern in patterns:
                match = re.search(pattern, html_content)
                if match:
                    confirm_token = match.group(1)
                    logger.info(f"Found confirmation token using pattern: {pattern[:30]}...")
                    break
            
            # Check cookies as well
            if not confirm_token:
                for key, value in response.cookies.items():
                    if 'download_warning' in key.lower() or 'confirm' in key.lower():
                        confirm_token = value
                        logger.info(f"Found confirmation token in cookies: {key}")
                        break
            
            if confirm_token:
                # Make a new request with the confirmation token
                logger.info(f"Using confirmation token: {confirm_token[:20]}...")
                params = {'id': file_id, 'confirm': confirm_token}
                response = session.get(url, params=params, stream=True)
            else:
                # Try the alternative method: use confirm=t (works for some files)
                logger.info("No token found, trying confirm=t...")
                params = {'id': file_id, 'confirm': 't'}
                response = session.get(url, params=params, stream=True)
        
        # Check if successful
        if response.status_code != 200:
            logger.error(f"Failed to download file. Status code: {response.status_code}")
            return False
        
        # Double-check we're not still getting HTML
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            logger.error(f"Still receiving HTML after confirmation attempt.")
            # Try one more time with confirm=t
            logger.info("Final attempt with confirm=t...")
            params = {'id': file_id, 'confirm': 't', 'uuid': ''}
            response = session.get(url, params=params, stream=True)
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                logger.error(f"Unable to bypass virus scan warning. File may need different sharing settings.")
                return False
        
        # Save file in chunks
        downloaded_size = 0
        logger.info("Starting file download...")
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Log progress for large files
                    if downloaded_size % (10 * 1024 * 1024) == 0:  # Every 10MB
                        logger.info(f"Downloaded {downloaded_size / (1024*1024):.0f} MB...")
        
        file_size = os.path.getsize(destination)
        logger.info(f"Successfully downloaded {destination} ({file_size / (1024*1024):.2f} MB)")
        
        # Verify file is not too small (likely an error page)
        if destination.endswith('.safetensors') and file_size < 1000000:  # Less than 1MB
            logger.error(f"Downloaded file is suspiciously small ({file_size} bytes). May be an error page.")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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
                # Try gdown first if available
                if GDOWN_AVAILABLE:
                    success = download_file_from_google_drive_gdown(file_id, str(file_path))
                else:
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
                    # Try gdown first if available
                    if GDOWN_AVAILABLE:
                        success = download_file_from_google_drive_gdown(extracted_id, str(file_path))
                    else:
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
    
    logger.info("[startup] Starting EthoScore API...")
    
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
            
        # Check dataset
        if dataset_path.exists():
            dataset_loaded = True
            logger.info("[startup] Dataset file found")
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

class AnalyzeUrlRequest(BaseModel):
    url: str = Field(..., description="URL to analyze")

class AnalyzeTextRequest(BaseModel):
    title: str = Field(..., description="Article title")
    body: str = Field(..., description="Article body text")

class AnalyzeResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TopicRequest(BaseModel):
    keyword: Optional[str] = Field(None, description="Keyword to search for")
    topic: Optional[str] = Field(None, description="Topic to explore")
    limit: Optional[int] = Field(3, description="Number of results to return")
    label: Optional[str] = Field(None, description="Filter by framing label")
    
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

@app.post("/analyze/url")
async def analyze_url(request: AnalyzeUrlRequest):
    """Extract article from URL and analyze it"""
    global analyzer
    
    if analyzer is None or not analyzer.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Models not initialized. Please check server logs and ensure model files are available."
        )
    
    try:
        # Extract article from URL
        logger.info(f"Extracting article from URL: {request.url}")
        title, body, metadata = extract_article_from_url(request.url)
        
        # Analyze the article
        logger.info(f"Analyzing extracted article: '{title[:50]}...'")
        analysis_result = analyzer.analyze_article(title, body)
        
        # Format response for frontend
        return {
            "title": title,
            "body": body,
            "body_preview": body[:500] if len(body) > 500 else body,
            "source": metadata.get("source", ""),
            "source_url": metadata.get("source_url", request.url),
            "publish_date": metadata.get("publish_date"),
            "analysis": {
                "ordinal_analysis": analysis_result.get("ordinal_analysis", {}),
                "classification_analysis": analysis_result.get("classification_analysis", {})
            },
            "ordinal_analysis": analysis_result.get("ordinal_analysis", {}),
            "classification_analysis": analysis_result.get("classification_analysis", {})
        }
    except ArticleExtractionError as e:
        logger.error(f"Article extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except ModelInferenceError as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze/text")
async def analyze_text(request: AnalyzeTextRequest):
    """Analyze manually provided text"""
    global analyzer
    
    if analyzer is None or not analyzer.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Models not initialized. Please check server logs and ensure model files are available."
        )
    
    try:
        # Analyze the article
        logger.info(f"Analyzing text: '{request.title[:50]}...'")
        analysis_result = analyzer.analyze_article(request.title, request.body)
        
        # Format response for frontend
        return {
            "title": request.title,
            "body": request.body,
            "body_preview": request.body[:500] if len(request.body) > 500 else request.body,
            "analysis": {
                "ordinal_analysis": analysis_result.get("ordinal_analysis", {}),
                "classification_analysis": analysis_result.get("classification_analysis", {})
            },
            "ordinal_analysis": analysis_result.get("ordinal_analysis", {}),
            "classification_analysis": analysis_result.get("classification_analysis", {})
        }
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
    
    try:
        import csv
        base_dir = Path(__file__).parent.parent
        dataset_path = base_dir / "Dataset-framing_annotations-Llama-3.3-70B-Instruct-Turbo.csv"
        
        if not dataset_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Dataset file not found"
            )
        
        logger.info(f"Streaming search of dataset {dataset_path} for '{topic}'")
        keyword_lower = topic.lower()
        limit = request.limit or 3
        label_filter = (request.label or "").strip().lower() or None
        
        results = []
        
        with open(dataset_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Stop once we have enough results
                if len(results) >= limit:
                    break
                
                # Extract title and body text from common columns
                title = ""
                body = ""
                
                for col in ["title", "headline"]:
                    if col in reader.fieldnames and row.get(col):
                        title = str(row[col])
                        break
                
                for col in ["text", "body", "article_text", "content"]:
                    if col in reader.fieldnames and row.get(col):
                        body = str(row[col])
                        break
                
                if not (title or body):
                    continue
                
                text_for_match = f"{title} {body}".lower()
                if keyword_lower not in text_for_match:
                    continue
                
                try:
                    # Run the analyzer
                    analysis_result = analyzer.analyze_article(title, body)
                    
                    # If a framing filter is set, filter by predicted label
                    if label_filter:
                        pred_label = (
                            analysis_result.get("classification_analysis", {}).get("predicted_label")
                            or analysis_result.get("ordinal_analysis", {}).get("predicted_label")
                            or ""
                        )
                        if pred_label.strip().lower() != label_filter:
                            continue
                    
                    result = {
                        "title": title,
                        "body": body,
                        "body_preview": body[:500] if len(body) > 500 else body,
                        "source": row.get("source", row.get("media_name", "")),
                        "source_url": row.get("url", row.get("source_url", "")),
                        "publish_date": row.get("publish_date") or None,
                        "analysis": {
                            "ordinal_analysis": analysis_result.get("ordinal_analysis", {}),
                            "classification_analysis": analysis_result.get("classification_analysis", {})
                        },
                        "ordinal_analysis": analysis_result.get("ordinal_analysis", {}),
                        "classification_analysis": analysis_result.get("classification_analysis", {})
                    }
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to analyze article during topic search: {str(e)}")
                    continue
        
        logger.info(f"Topic search for '{topic}' returning {len(results)} results")
        
        return {
            "success": True,
            "topic": topic,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error searching dataset: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error searching dataset: {str(e)}"
        )

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
