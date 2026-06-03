import os
from pathlib import Path

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'dev-secret-key-change-in-production')

DEBUG = True

ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '*').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third party
    'rest_framework',
    'corsheaders',
    'django_filters',
    # Local apps
    'apps.core',
    'apps.documents',
    'apps.retrieval',
    'apps.chat',
    'apps.analytics',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# CORS
CORS_ALLOW_ALL_ORIGINS = DEBUG
_cors_origins = os.environ.get('CORS_ALLOWED_ORIGINS', '')
CORS_ALLOWED_ORIGINS = [o.strip() for o in _cors_origins.split(',') if o.strip()] if _cors_origins else []

# Cache - 使用本地缓存便于测试
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
        'KEY_PREFIX': 'rag_django',
        'TIMEOUT': 300,
    }
}

# Vector Store
VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent.parent / 'rag_langchain'
CHROMA_PERSIST_DIR = str(VECTOR_STORE_DIR / 'chroma_data')

# LLM Settings
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
LLM_MODEL = os.environ.get('LLM_MODEL', 'qwen-plus')

# Embedding Settings - 必须与 LangChain 框架一致
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'text-embedding-v2')
EMBEDDING_DIMENSION = int(os.environ.get('EMBEDDING_DIMENSION', '1536'))

# Retrieval Settings
RETRIEVAL_K = int(os.environ.get('RETRIEVAL_K', '10'))
RERANK_TOP_N = int(os.environ.get('RERANK_TOP_N', '5'))
RRF_K = int(os.environ.get('RRF_K', '60'))

# Channel Weights by Query Type
CHANNEL_WEIGHTS = {
    'keyword': [0.3, 0.7],    # [vector, bm25]
    'semantic': [0.7, 0.3],
    'balanced': [0.5, 0.5],
}

# Cache TTL Settings (seconds)
CACHE_TTL = {
    'query_result': 300,      # 5 minutes
    'retrieval_result': 1800, # 30 minutes
    'embedding': 86400,       # 24 hours
}

# Chunk Size for Document Processing
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '512'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '50'))

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'apps': {
            'handlers': ['console'],
            'level': 'DEBUG' if DEBUG else 'INFO',
            'propagate': False,
        },
    },
}