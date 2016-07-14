import os
import sys

# ENV VARIABLES

ROOT_DIR = os.environ['ROOT_DIR']
# add the 'src' directory as one where we can import modules
SRC_DIR = os.path.join(os.environ['ROOT_DIR'], 'src')
DAT_DIR = os.path.join(os.environ['ROOT_DIR'], 'data')
MODELS_DIR = os.path.join(os.environ['ROOT_DIR'], 'models')
sys.path.append(SRC_DIR)

RAW = os.path.join(DAT_DIR, 'raw')
PROCESSED = os.path.join(DAT_DIR, 'processed')

# Room models
