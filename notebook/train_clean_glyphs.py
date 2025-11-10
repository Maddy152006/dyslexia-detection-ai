#!/usr/bin/env python
# coding: utf-8

# In[4]:


# STEP 1
import os, pathlib
print("CWD:", os.getcwd())
print("Items in CWD:", [p.name for p in pathlib.Path.cwd().iterdir()])


# In[3]:


import pathlib

def locate_dataset(start=pathlib.Path.cwd(), max_up=3):
    # try common relatives
    candidates = [
        start / "dataset",
        start.parent / "dataset",
        start.parent.parent / "dataset",
        start / "../dataset",
        start / "../../dataset",
    ]
    # search upward
    ups = [start] + list(start.parents)[:max_up]
    for up in ups:
        candidates += list(up.glob("**/dataset"))

    # validate
    for p in candidates:
        if not p.exists(): 
            continue
        for TT in ("Train","train"):
            for tT in ("Test","test"):
                if (p/TT).is_dir() and (p/tT).is_dir():
                    return p, (TT, tT)
    return None, (None, None)

DATA_ROOT, (TRAIN_NAME, TEST_NAME) = locate_dataset()
print("DATA_ROOT:", DATA_ROOT)
print("Train dir name found:", TRAIN_NAME, "| Test dir name found:", TEST_NAME)


# In[2]:


import pathlib, os
print("CWD:", os.getcwd())
print("Here:", pathlib.Path.cwd())
print("List here:", [p.name for p in pathlib.Path.cwd().iterdir()])


