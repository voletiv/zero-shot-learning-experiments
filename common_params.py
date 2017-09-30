import os
import sys

#############################################################
# COMMON PARAMETERS
#############################################################

# Root directory with common functions
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Lip reader directory
LIPREADER_DIR = os.path.join(ROOT_DIR, 'Lipreader')
if LIPREADER_DIR not in sys.path:
    sys.path.append(LIPREADER_DIR)
