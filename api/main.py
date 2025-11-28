"""
FastAPI backend for EthoScore Article Analysis
"""

import os
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import pandas as pd
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
_dataset_df: Optional[pd.DataFrame] = None
_dataset_path: Optional[Path] = None


def load_dataset() -> Optional[pd.DataFrame]:
    """
    Load the framing annotations dataset into memory (cached global DataFrame).
    """
    global _dataset_df, dataset_loaded, _dataset_path

    base_dir = Path(__file__).parent.parent
    dataset_path = base_dir / "Dataset-framing_annotations-Llama-3.3-70B-Instruct-Turbo.csv"
    _dataset_path = dataset_path

    if _dataset_df is not None:
        return _dataset_df

    if not dataset_path.exists():
        logger.warning(f"Dataset file not found at {dataset_path}")
        dataset_loaded = False
        return None

    try:
        logger.info(f"Loading dataset for topic exploration from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        logger.info(
            f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}"
        )
        _dataset_df = df
        dataset_loaded = True
        return _dataset_df
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
        dataset_loaded = False
        _dataset_df = None
        return None


def _choose_column(columns: List[str], keywords: List[str]) -> Optional[str]:
    """
    Pick the first column whose name contains any of the given keywords.
    """
    lowered = [c.lower() for c in columns]
    for kw in keywords:
        for col, lower_name in zip(columns, lowered):
            if kw in lower_name:
                return col
    return None


def search_dataset(
    topic: str,
    limit: int = 3,
    label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search the precomputed dataset for rows matching the topic (and optional label).

    Uses the specific dataset schema from the Llama-3.3 annotated dataset:
    - 'concept' column: topic category (primary search target)
    - 'title' column: article title (secondary search)
    - 'body' column: article body text (secondary search)  
    - 'FRAMING_CLASS' column: framing label (NEUTRAL, LOADED, ALARMIST)
    - 'source' column: source information
    """
    global _dataset_df, _dataset_path

    if not topic:
        return []

    if _dataset_df is None:
        df = load_dataset()
    else:
        df = _dataset_df

    if df is None or df.empty:
        logger.warning("Dataset is not loaded or is empty; topic search cannot proceed")
        return []

    topic_str = str(topic).strip()
    topic_lower = topic_str.lower()

    dataset_display_path = str(_dataset_path) if _dataset_path is not None else "<in-memory>"
    logger.info(
        f"Searching dataset {dataset_display_path} for topic '{topic_str}'"
    )

    # Build topic filter mask - search in concept, title, and body columns
    mask = pd.Series(False, index=df.index)
    
    # Primary: Search in 'concept' column (topic category)
    if 'concept' in df.columns:
        concept_match = df['concept'].astype(str).str.contains(topic_lower, case=False, na=False)
        mask |= concept_match
        logger.info(f"Found {concept_match.sum()} matches in 'concept' column")
    
    # Secondary: Search in 'title' column
    if 'title' in df.columns:
        title_match = df['title'].astype(str).str.contains(topic_lower, case=False, na=False)
        mask |= title_match
        logger.info(f"Found {title_match.sum()} matches in 'title' column")
    
    # Secondary: Search in 'body' column
    if 'body' in df.columns:
        body_match = df['body'].astype(str).str.contains(topic_lower, case=False, na=False)
        mask |= body_match
        logger.info(f"Found {body_match.sum()} matches in 'body' column")

    # Apply framing label filter using FRAMING_CLASS column
    if label:
        label_upper = str(label).strip().upper()
        if 'FRAMING_CLASS' in df.columns:
            label_mask = df['FRAMING_CLASS'].astype(str).str.upper() == label_upper
            mask &= label_mask
            logger.info(f"Applied FRAMING_CLASS filter: '{label_upper}'")
        else:
            logger.warning("'FRAMING_CLASS' column not found in dataset")

    filtered = df[mask]
    
    if filtered.empty:
        logger.info(f"Topic search for '{topic_str}' returning 0 results")
        return []

    logger.info(f"Found {len(filtered)} total matches, returning top {limit}")

    results: List[Dict[str, Any]] = []

    for _, row in filtered.head(limit).iterrows():
        record: Dict[str, Any] = {}

        # Title
        if 'title' in df.columns and pd.notna(row.get('title')):
            record["title"] = str(row['title']).strip()
        else:
            record["title"] = topic_str

        # Body preview
        if 'body' in df.columns and pd.notna(row.get('body')):
            body_text = str(row['body'])
            record["body_preview"] = body_text[:2000]

        # Source - handle both string and dict formats
        if 'source' in df.columns and pd.notna(row.get('source')):
            source_val = row['source']
            if isinstance(source_val, dict):
                record["source"] = source_val.get('title') or source_val.get('uri', str(source_val))
            else:
                record["source"] = str(source_val)

        # Concept/topic category
        if 'concept' in df.columns and pd.notna(row.get('concept')):
            record["concept"] = str(row['concept'])

        # URL if available
        if 'url' in df.columns and pd.notna(row.get('url')):
            record["source_url"] = str(row['url'])

        # Framing label from FRAMING_CLASS column
        if 'FRAMING_CLASS' in df.columns and pd.notna(row.get('FRAMING_CLASS')):
            framing_label = str(row['FRAMING_CLASS']).strip()
            # Normalize to title case for display
            framing_label_display = framing_label.title()
            record["ordinal_analysis"] = {"predicted_label": framing_label_display}
            record["classification_analysis"] = {"predicted_label": framing_label_display}
            record["framing_class"] = framing_label_display

        results.append(record)

    logger.info(
        f"Topic search for '{topic_str}' returning {len(results)} results (limit={limit})"
    )
    return results

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
            
        # Check & load dataset for topic exploration
        if dataset_path.exists():
            logger.info("[startup] Dataset file found, attempting to load into memory")
            if load_dataset() is not None:
                logger.info("[startup] Dataset loaded successfully")
            else:
                logger.warning("[startup] Failed to load dataset into memory")
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
