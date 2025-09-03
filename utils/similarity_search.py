import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO
import logging

class SimilaritySearch:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = None, None
        self.embeddings_cache = {}  # Cache for image embeddings
        
    def load_model(self):
        """Load the CLIP model"""
        if self.model is None:
            try:
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                logging.info("CLIP model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading CLIP model: {e}")
                raise
    
    def get_image_embedding(self, image_url):
        """Get embedding for an image from URL"""
        if image_url in self.embeddings_cache:
            return self.embeddings_cache[image_url]
            
        try:
            # Download image
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Open and preprocess image
            image = Image.open(BytesIO(response.content))
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.encode_image(image).cpu().numpy()
                
            # Cache the embedding
            self.embeddings_cache[image_url] = embedding
            return embedding
            
        except Exception as e:
            logging.error(f"Error getting embedding for {image_url}: {e}")
            return None
    
    def find_similar_images(self, query_url, image_records, threshold=0.7):
        """Find similar images from a collection of records"""
        self.load_model()
        
        # Get query embedding
        query_embedding = self.get_image_embedding(query_url)
        if query_embedding is None:
            return []
        
        similar_images = []
        
        # Compare with all cropped objects from all records
        for record in image_records:
            # Add the main processed image
            if record.processed_url and record.processed_url != query_url:
                processed_embedding = self.get_image_embedding(record.processed_url)
                if processed_embedding is not None:
                    similarity = cosine_similarity(query_embedding, processed_embedding)[0][0]
                    if similarity >= threshold:
                        similar_images.append({
                            "url": record.processed_url,
                            "similarity": float(similarity),
                            "type": "processed_image",
                            "record_id": record.id
                        })
            
            # Add cropped objects
            for cropped_obj in record.cropped_objects_urls:
                if cropped_obj['url'] != query_url:
                    cropped_embedding = self.get_image_embedding(cropped_obj['url'])
                    if cropped_embedding is not None:
                        similarity = cosine_similarity(query_embedding, cropped_embedding)[0][0]
                        if similarity >= threshold:
                            similar_images.append({
                                "url": cropped_obj['url'],
                                "similarity": float(similarity),
                                "label": cropped_obj.get('label', 'Object'),
                                "confidence": cropped_obj.get('confidence', 0),
                                "type": "cropped_object",
                                "record_id": record.id
                            })
        
        # Sort by similarity (highest first)
        similar_images.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_images[:10]  # Return top 10 results