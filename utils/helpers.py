"""
utils/helpers.py
Small, boring, shared utilities that don't deserve their own module.
"""

import uuid


def new_job_id():
    return uuid.uuid4().hex


def human_word_count(text):
    return len(text.split())
