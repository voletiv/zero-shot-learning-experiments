import os
import sys

#############################################################
# COMMON PARAMETERS
#############################################################

# Root directory with common functions
if 'ROOT_DIR' not in dir():
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Lip reader directory
if 'LIPREADER_DIR' not in dir():
    LIPREADER_DIR = os.path.join(ROOT_DIR, 'lipreader')

if LIPREADER_DIR not in sys.path:
    sys.path.append(LIPREADER_DIR)
