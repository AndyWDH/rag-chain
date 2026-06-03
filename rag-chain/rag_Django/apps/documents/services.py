from typing import Dict
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from apps.documents.models import Document, Chunk, Collection
from apps.documents.vectorstore import get_vector_store_manager
from apps.documents.bm25 import get_bm25_manager
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE or 512,
            chunk_overlap=settings.CHUNK_OVERLAP or 50,
            length_function=len,
        )
        self.vector_manager = get_vector_store_manager()
        self.bm25_manager = get_bm25_manager()

    def process_document(self, document: Document) -> Dict:
        try:
            document.status = 'processing'
            document.save()

            file_path = document.file_path
            if file_path.endswith('.pdf'):
                texts = self._extract_pdf_text(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts = [f.read()]

            chunks = self.text_splitter.split_text(texts[0] if texts else '')

            created_chunks = []
            for idx, chunk_text in enumerate(chunks):
                chunk = Chunk.objects.create(
                    document=document,
                    content=chunk_text,
                    chunk_index=idx,
                    metadata={
                        'source': document.title,
                        'chunk_size': len(chunk_text),
                    }
                )
                created_chunks.append(chunk)

            document.total_chunks = len(chunks)
            document.processed_chunks = len(chunks)
            document.status = 'completed'
            document.save()

            self._index_chunks(created_chunks)

            return {
                'chunks_created': len(created_chunks),
                'document_id': document.id,
            }

        except Exception as e:
            document.status = 'failed'
            document.error_message = str(e)
            document.save()
            logger.error(f"Failed to process document {document.id}: {e}")
            raise

    def _extract_pdf_text(self, file_path: str) -> list:
        try:
            reader = PdfReader(file_path)
            texts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            return texts
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
            return []

    def _index_chunks(self, chunks: list):
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"chunk_{chunk.id}" for chunk in chunks]

        try:
            self.vector_manager.add_documents(texts, metadatas, ids)
        except Exception as e:
            logger.error(f"Failed to index chunks to vector store: {e}")

        try:
            self.bm25_manager.add_documents(texts)
        except Exception as e:
            logger.error(f"Failed to index chunks to BM25: {e}")

    def delete_document(self, document: Document):
        chunks = document.chunks.all()
        chunk_ids = [f"chunk_{chunk.id}" for chunk in chunks]

        try:
            self.vector_manager.delete(ids=chunk_ids)
        except Exception as e:
            logger.error(f"Failed to delete from vector store: {e}")

        document.delete()

    def rebuild_collection_index(self, collection: Collection):
        all_chunks = Chunk.objects.filter(document__collection=collection)

        texts = [chunk.content for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]
        ids = [f"chunk_{chunk.id}" for chunk in all_chunks]

        self.vector_manager.delete(delete_all=True)
        self.bm25_manager._initialize_bm25()

        if texts:
            self.vector_manager.add_documents(texts, metadatas, ids)
            self.bm25_manager.build_index(texts)